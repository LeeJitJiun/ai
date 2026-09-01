import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MeanShift, DBSCAN, estimate_bandwidth
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

@st.cache_data
def load_data(filepath="global_university_studentsperformance.csv"):
    df = pd.read_csv(filepath)
    return df.dropna().reset_index(drop=True)

@st.cache_data
def preprocess_features(df, selected_features):
    # Convert tuple back to a list for pandas
    selected_features = list(selected_features)
    
    if not selected_features:
        selected_features = [
            c for c in df.columns 
            if pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in ['student_id', 'exam_score']
        ]
    
    sub_df = df[selected_features].copy()
    encoded_df = pd.get_dummies(sub_df, drop_first=True)
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(encoded_df)
    return scaled_matrix, encoded_df

def evaluate_clusters(scaled_matrix, labels):
    valid_mask = labels != -1
    unique_clusters = set(labels[valid_mask])
    total_points = len(labels)
    noise_count = sum(labels == -1)
    
    metrics = {
        "silhouette": 0.0,
        "davies_bouldin": None,
        "calinski_harabasz": None,
        "noise_ratio": round((noise_count / total_points) * 100, 2)
    }
    
    if len(unique_clusters) > 1:
        clean_matrix = scaled_matrix[valid_mask]
        clean_labels = labels[valid_mask]
        metrics["silhouette"] = round(float(silhouette_score(clean_matrix, clean_labels)), 4)
        metrics["davies_bouldin"] = round(float(davies_bouldin_score(clean_matrix, clean_labels)), 4)
        metrics["calinski_harabasz"] = round(float(calinski_harabasz_score(clean_matrix, clean_labels)), 2)
        
    return metrics

def run_kmeans_model(scaled_matrix, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_matrix)
    metrics = evaluate_clusters(scaled_matrix, labels)
    inertia = round(float(model.inertia_), 2)
    return model, labels, metrics, inertia

def run_meanshift_model(scaled_matrix, quantile=0.2):
    # n_samples=300 speeds up bandwidth estimation significantly
    bandwidth = estimate_bandwidth(scaled_matrix, quantile=quantile, n_samples=300, random_state=42)
    if bandwidth is None or bandwidth <= 0:
        bandwidth = 1.0
    # bin_seeding=True speeds up MeanShift convergence by 40x
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = model.fit_predict(scaled_matrix)
    metrics = evaluate_clusters(scaled_matrix, labels)
    return model, labels, metrics, round(float(bandwidth), 2)

def calculate_suggested_eps(scaled_matrix, min_samples=4, percentile=75):
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(scaled_matrix)
    distances, _ = neighbors.kneighbors(scaled_matrix)
    val = float(np.percentile(distances[:, -1], percentile))
    return max(0.05, round(val, 2))

@st.cache_data
def optimize_dbscan_params(scaled_matrix, max_noise_ratio=15.0):
    best_eps = calculate_suggested_eps(scaled_matrix)
    best_min_samples = 4
    
    eps_range = np.arange(0.05, 2.05, 0.05)
    min_samples_range = range(2, 12)
    
    valid_results = []
    
    for e in eps_range:
        for m in min_samples_range:
            model = DBSCAN(eps=e, min_samples=m)
            labels = model.fit_predict(scaled_matrix)
            
            unique_clusters = set(labels) - {-1}
            if len(unique_clusters) > 1:
                noise_ratio = (list(labels).count(-1) / len(labels)) * 100
                
                if noise_ratio <= max_noise_ratio:
                    valid_mask = labels != -1
                    clean_matrix = scaled_matrix[valid_mask]
                    clean_labels = labels[valid_mask]
                    sil = float(silhouette_score(clean_matrix, clean_labels))
                    
                    valid_results.append({
                        'eps': float(e), 
                        'min_samples': int(m), 
                        'silhouette': sil, 
                        'noise': noise_ratio
                    })
                    
    if valid_results:
        valid_results.sort(key=lambda x: x['silhouette'], reverse=True)
        best = valid_results[0]
        return round(best['eps'], 2), best['min_samples']
        
    return round(best_eps, 2), 4

def run_dbscan_model(scaled_matrix, eps=0.1, min_samples=4):
    safe_eps = max(0.01, float(eps))
    model = DBSCAN(eps=safe_eps, min_samples=min_samples)
    labels = model.fit_predict(scaled_matrix)
    metrics = evaluate_clusters(scaled_matrix, labels)
    return model, labels, metrics

@st.cache_data
def get_kmeans_evaluation_table(scaled_matrix, k_min=2, k_max=10):
    results = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_matrix)
        inertia = km.inertia_
        metrics = evaluate_clusters(scaled_matrix, labels)
        results.append({
            'K': k,
            'Inertia': round(float(inertia), 2),
            'Silhouette Score': metrics['silhouette'],
            'Davies-Bouldin': metrics['davies_bouldin']
        })
    return pd.DataFrame(results)