import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MeanShift, DBSCAN, estimate_bandwidth
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

@st.cache_data
def load_data(filepath="student_habits_performance (1).csv"):
    df = pd.read_csv(filepath)
    return df.dropna().reset_index(drop=True)

@st.cache_data
def preprocess_features(df, selected_features):
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
    
    # FIX: Add 'scaler' to the end of this return statement
    return scaled_matrix, encoded_df, scaler

def evaluate_clusters(scaled_matrix, labels):
    valid_mask = labels != -1
    unique_clusters = set(labels[valid_mask])
    total_points = len(labels)
    noise_count = sum(labels == -1)
    
    metrics = {
        "silhouette": 0.0,
        "davies_bouldin": None,
        "noise_ratio": round((noise_count / total_points) * 100, 2)
    }
    
    if len(unique_clusters) > 1:
        clean_matrix = scaled_matrix[valid_mask]
        clean_labels = labels[valid_mask]
        metrics["silhouette"] = round(float(silhouette_score(clean_matrix, clean_labels)), 4)
        metrics["davies_bouldin"] = round(float(davies_bouldin_score(clean_matrix, clean_labels)), 4)
        
    return metrics

def run_kmeans_model(scaled_matrix, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_matrix)
    metrics = evaluate_clusters(scaled_matrix, labels)
    return model, labels, metrics, round(float(model.inertia_), 2)

def run_meanshift_model(scaled_matrix, quantile=0.2):
    bandwidth = estimate_bandwidth(scaled_matrix, quantile=quantile, n_samples=300, random_state=42)
    if bandwidth is None or bandwidth <= 0: bandwidth = 1.0
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = model.fit_predict(scaled_matrix)
    metrics = evaluate_clusters(scaled_matrix, labels)
    return model, labels, metrics, round(float(bandwidth), 2)

def run_dbscan_model(scaled_matrix, eps=0.1, min_samples=4):
    model = DBSCAN(eps=max(0.01, float(eps)), min_samples=min_samples)
    labels = model.fit_predict(scaled_matrix)
    metrics = evaluate_clusters(scaled_matrix, labels)
    return model, labels, metrics

# --- EVALUATION TABLE GENERATORS ---

@st.cache_data
def get_kmeans_evaluation_table(scaled_matrix, k_min=2, k_max=10):
    results = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_matrix)
        metrics = evaluate_clusters(scaled_matrix, labels)
        results.append({
            'K': k, 'Inertia': round(float(km.inertia_), 2),
            'Silhouette Score': metrics['silhouette'], 'Davies-Bouldin': metrics['davies_bouldin']
        })
    return pd.DataFrame(results)

@st.cache_data
def get_meanshift_evaluation_table(scaled_matrix):
    results = []
    for q in np.arange(0.05, 0.55, 0.05):
        q = round(q, 2)
        bw = estimate_bandwidth(scaled_matrix, quantile=q, n_samples=300, random_state=42)
        if bw is None or bw <= 0: bw = 1.0
        ms = MeanShift(bandwidth=bw, bin_seeding=True)
        labels = ms.fit_predict(scaled_matrix)
        metrics = evaluate_clusters(scaled_matrix, labels)
        n_clusters = len(set(labels) - {-1})
        results.append({
            'Quantile': q, 'Bandwidth': round(bw, 2), 'Clusters Found': n_clusters,
            'Silhouette Score': metrics['silhouette'], 'Davies-Bouldin': metrics['davies_bouldin']
        })
    return pd.DataFrame(results)

@st.cache_data
def get_dbscan_evaluation_data(scaled_matrix, max_noise=20.0):
    results = []
    n_samples = len(scaled_matrix)
    min_cluster_size = int(n_samples * 0.02)  # A "decent" cluster is 2% of the data
    
    for e in np.arange(0.10, 1.55, 0.05):
        for m in range(4, 15): 
            model = DBSCAN(eps=e, min_samples=m)
            labels = model.fit_predict(scaled_matrix)
            
            # Get unique clusters ignoring noise
            unique_clusters = [c for c in set(labels) if c != -1]
            
            if len(unique_clusters) > 1:
                noise_ratio = (list(labels).count(-1) / n_samples) * 100
                
                # Check if noise is acceptable
                if noise_ratio <= max_noise:
                    valid_mask = labels != -1
                    clean_labels = labels[valid_mask]
                    
                    # Count sizes of valid clusters
                    counts = pd.Series(clean_labels).value_counts()
                    
                    # Ensure we have at least TWO decently sized clusters 
                    decent_clusters = sum(counts >= min_cluster_size)
                    
                    if decent_clusters >= 2:
                        sil = float(silhouette_score(scaled_matrix[valid_mask], clean_labels))
                        results.append({
                            'eps': round(e, 2), 
                            'min_samples': m, 
                            'silhouette': sil, 
                            'noise_ratio': noise_ratio,
                            'n_clusters': len(unique_clusters)
                        })
    return pd.DataFrame(results)

# --- IMPROVED AUTO-TUNERS ---

@st.cache_data
def optimize_kmeans_params(scaled_matrix):
    df = get_kmeans_evaluation_table(scaled_matrix)
    # Balance Silhouette Score and Davies-Bouldin Index (higher Sil, lower DB)
    df['Combined_Score'] = df['Silhouette Score'] / (df['Davies-Bouldin'] + 1e-5)
    best_k = df.loc[df['Combined_Score'].idxmax()]['K']
    return int(best_k)

@st.cache_data
def optimize_meanshift_params(scaled_matrix):
    df = get_meanshift_evaluation_table(scaled_matrix)
    n_samples = len(scaled_matrix)
    min_cluster_size = int(n_samples * 0.03)  # Require at least 3% of data per cluster
    
    valid = df[df['Clusters Found'] > 1]
    if not valid.empty:
        return float(valid.loc[valid['Silhouette Score'].idxmax()]['Quantile'])
    return 0.20

@st.cache_data
def optimize_dbscan_params(scaled_matrix):
    # Try to find the best configuration with under 15% noise
    df = get_dbscan_evaluation_data(scaled_matrix, max_noise=15.0)
    
    if not df.empty:
        # Sort by best silhouette score
        best = df.loc[df['silhouette'].idxmax()]
        return float(best['eps']), int(best['min_samples'])
        
    # SMARTER FALLBACK: If grid search fails, mathematically estimate a good EPS based on neighbor distances
    distances, _ = NearestNeighbors(n_neighbors=5).fit(scaled_matrix).kneighbors(scaled_matrix)
    suggested_eps = max(0.1, round(float(np.percentile(distances[:, -1], 85)), 2))
    return suggested_eps, 5