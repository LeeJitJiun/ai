import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans, MeanShift, DBSCAN, estimate_bandwidth
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from helper_functions import (
    load_data, preprocess_features,
    run_kmeans_model, run_meanshift_model, run_dbscan_model,
    get_kmeans_evaluation_table, get_meanshift_evaluation_table, get_dbscan_evaluation_data
)

# ---------------- INLINE SMART AUTO-TUNE FUNCTIONS ----------------
def optimize_kmeans_params(scaled_matrix, max_k=10):
    inertias = []
    k_values = list(range(2, max_k + 1))

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled_matrix)
        inertias.append(km.inertia_)

    k_norm = (np.array(k_values) - min(k_values)) / (max(k_values) - min(k_values))
    i_norm = (np.array(inertias) - min(inertias)) / (max(inertias) - min(inertias))

    p1 = np.array([k_norm[0], i_norm[0]])
    p2 = np.array([k_norm[-1], i_norm[-1]])

    v1 = p2 - p1
    norm_v1 = np.linalg.norm(v1)

    distances = []
    for i in range(len(k_values)):
        p3 = np.array([k_norm[i], i_norm[i]])
        v2 = p1 - p3
        cross_2d = v1[0] * v2[1] - v1[1] * v2[0]
        distance = np.abs(cross_2d) / norm_v1
        distances.append(distance)

    return k_values[np.argmax(distances)]

def optimize_meanshift_params(scaled_matrix):
    quantiles = np.arange(0.02, 0.18, 0.01)
    best_quantile = 0.08
    best_score = -1.0

    for q in quantiles:
        q = float(round(q, 3))
        try:
            bw = estimate_bandwidth(scaled_matrix, quantile=q, n_samples=min(300, len(scaled_matrix)), random_state=42)
            if bw is None or bw <= 0:
                continue
            ms = MeanShift(bandwidth=bw, bin_seeding=True)
            labels = ms.fit_predict(scaled_matrix)
            n_clusters = len(set(labels))
            
            if 2 <= n_clusters <= 10:
                score = silhouette_score(scaled_matrix, labels)
                if score > best_score:
                    best_score = score
                    best_quantile = q
        except Exception:
            continue

    return round(best_quantile, 2)

def optimize_dbscan_params(scaled_matrix):
    eps_values = np.linspace(0.1, 0.8, 15)
    min_samples_values = [3, 4, 5, 6, 8, 10]
    best_eps, best_min, best_score = 0.2, 4, -1.0

    for eps in eps_values:
        for m_samples in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=m_samples)
            labels = db.fit_predict(scaled_matrix)
            n_clusters = len(set(labels) - {-1})
            noise_ratio = np.sum(labels == -1) / len(labels)

            if n_clusters >= 2 and noise_ratio < 0.20:
                core_mask = labels != -1
                if np.sum(core_mask) > 0:
                    score = silhouette_score(scaled_matrix[core_mask], labels[core_mask])
                    if score > best_score:
                        best_score, best_eps, best_min = score, eps, m_samples

    return round(best_eps, 2), int(best_min)


# ---------------- STREAMLIT APP CONFIG ----------------
st.set_page_config(page_title="Student Clustering Dashboard", layout="wide", page_icon="🎓")

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

numeric_cols = [
    c for c in df.columns 
    if pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in ['student_id', 'exam_score']
]

# ---------------- SIDEBAR STRUCTURE & LAYOUT ----------------

# 1. Top Left App Title
st.sidebar.markdown("## 🤖 Unsupervised Machine Learning")
st.sidebar.markdown("---")

# 2. Main Navigation moved to Top of Sidebar
nav_container = st.sidebar.container()
st.sidebar.markdown("---")

# Reserve ordered sidebar containers
algo_container = st.sidebar.container()
st.sidebar.markdown("---")
mode_container = st.sidebar.container()
st.sidebar.markdown("---")
feat_container = st.sidebar.container()

# Render Navigation Radio
with nav_container:
    app_mode = st.radio(
        "Navigation", 
        ["📊 Dataset Analysis", "🔮 User Data Predictor"], 
        label_visibility="collapsed"
    )

# Render Section 2: Model Controls
with mode_container:
    st.header("⚙️ Model Controls")
    view_dim = st.radio("Visualization Mode", ["2D Mode (2 Features)", "3D Mode (3 Features)"])

# Render Section 3: Feature Selection
with feat_container:
    st.header("📊 Feature Selection")
    default_x_idx = numeric_cols.index("attendance_percentage") if "attendance_percentage" in numeric_cols else 0
    feat_x = st.selectbox("Select X-Axis Feature", numeric_cols, index=default_x_idx)
    feat_y = st.selectbox("Select Y-Axis Feature", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

    if view_dim == "2D Mode (2 Features)":
        selected_features = list(dict.fromkeys([feat_x, feat_y]))
    else:
        feat_z = st.selectbox("Select Z-Axis Feature", numeric_cols, index=2 if len(numeric_cols) > 2 else 0)
        selected_features = list(dict.fromkeys([feat_x, feat_y, feat_z]))

# Preprocess matrix based on selection
scaled_matrix, encoded_df, scaler = preprocess_features(df, tuple(selected_features))
current_inertia = None

# Render Section 1: Clustering Algorithm (at top of controls)
with algo_container:
    st.header("🤖 Clustering Algorithm")
    algorithm = st.selectbox("Choose Algorithm", ["K-Means", "MeanShift", "DBSCAN"])

    if algorithm == "K-Means":
        if 'km_k' not in st.session_state:
            st.session_state['km_k'] = 3
            
        if st.button("🤖 Auto-Tune K-Means", use_container_width=True):
            with st.spinner("Finding optimal K using elbow method..."):
                best_k = optimize_kmeans_params(scaled_matrix)
                st.session_state['km_k'] = int(best_k)
                st.rerun()
            
        k = st.slider("Number of Clusters (K)", min_value=2, max_value=10, key='km_k')
        model, labels, metrics, current_inertia = run_kmeans_model(scaled_matrix, n_clusters=k)

    elif algorithm == "MeanShift":
        if 'ms_q' not in st.session_state:
            st.session_state['ms_q'] = 0.08
            
        if st.button("🤖 Auto-Tune MeanShift", use_container_width=True):
            with st.spinner("Finding optimal quantile..."):
                best_q = optimize_meanshift_params(scaled_matrix)
                st.session_state['ms_q'] = float(best_q)
                st.rerun()
            
        quantile = st.slider("Bandwidth Quantile (Density)", min_value=0.01, max_value=0.20, step=0.01, key='ms_q')
        model, labels, metrics, computed_bw = run_meanshift_model(scaled_matrix, quantile=quantile)
        st.info(f"Computed Bandwidth: **{computed_bw:.2f}**")

    elif algorithm == "DBSCAN":
        if 'db_eps' not in st.session_state:
            st.session_state['db_eps'] = 0.10
        if 'db_min' not in st.session_state:
            st.session_state['db_min'] = 4
            
        if st.button("🤖 Auto-Tune DBSCAN", use_container_width=True):
            with st.spinner("Finding optimal EPS and Min Samples..."):
                e, m = optimize_dbscan_params(scaled_matrix)
                st.session_state['db_eps'] = float(round(e / 0.05) * 0.05)
                st.session_state['db_min'] = int(m)
                st.rerun()
            
        eps = st.slider("EPS (Neighborhood Radius)", min_value=0.05, max_value=5.0, step=0.05, key='db_eps')
        min_samples = st.slider("Min Samples", min_value=2, max_value=20, key='db_min')
        model, labels, metrics = run_dbscan_model(scaled_matrix, eps=eps, min_samples=min_samples)

# Prepare DataFrame Labels
df['Cluster'] = labels
sorted_cluster_nums = sorted(df['Cluster'].unique())
ordered_labels = [f"Noise (-1)" if x == -1 else f"Cluster {x:02d}" for x in sorted_cluster_nums]
df['Cluster_Label'] = df['Cluster'].apply(lambda x: f"Noise (-1)" if x == -1 else f"Cluster {x:02d}")
df['Cluster_Label'] = pd.Categorical(df['Cluster_Label'], categories=ordered_labels, ordered=True)

# ---------------- MAIN CONTENT AREA ----------------
st.title("🎓 Student Habits & Performance Clustering")

if app_mode == "📊 Dataset Analysis":
    col1, col2 = st.columns([2.2, 1])

    with col1:
        st.subheader(f"{algorithm} - {view_dim}")
        color_palette = px.colors.qualitative.Bold
        
        n_components = 2 if view_dim == "2D Mode (2 Features)" else 3
        
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_matrix)
        
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(scaled_matrix)-1))
        tsne_result = tsne.fit_transform(scaled_matrix)
        
        df['PCA_1'], df['PCA_2'] = pca_result[:, 0], pca_result[:, 1]
        df['tSNE_1'], df['tSNE_2'] = tsne_result[:, 0], tsne_result[:, 1]
        
        if n_components == 3:
            df['PCA_3'] = pca_result[:, 2]
            df['tSNE_3'] = tsne_result[:, 2]

        vis_tab1, vis_tab2, vis_tab3 = st.tabs(["📊 Original Features", "📉 PCA Projection", "🌌 t-SNE Projection"])

        with vis_tab1:
            if view_dim == "2D Mode (2 Features)":
                fig_orig = px.scatter(df, x=feat_x, y=feat_y, color="Cluster_Label", hover_data=["exam_score"] if "exam_score" in df.columns else None, color_discrete_sequence=color_palette, opacity=0.85)
                fig_orig.update_traces(marker=dict(size=8))
                fig_orig.update_layout(height=500, template="plotly_dark")
            else:
                fig_orig = px.scatter_3d(df, x=feat_x, y=feat_y, z=feat_z, color="Cluster_Label", hover_data=["exam_score"] if "exam_score" in df.columns else None, color_discrete_sequence=color_palette, opacity=0.85)
                fig_orig.update_traces(marker=dict(size=5))
                fig_orig.update_layout(height=550, template="plotly_dark", scene=dict(aspectmode='cube'))
            st.plotly_chart(fig_orig, use_container_width=True)

        with vis_tab2:
            if view_dim == "2D Mode (2 Features)":
                fig_pca = px.scatter(df, x="PCA_1", y="PCA_2", color="Cluster_Label", color_discrete_sequence=color_palette, opacity=0.85, title="PCA 2D Projection")
                fig_pca.update_traces(marker=dict(size=8))
                fig_pca.update_layout(height=500, template="plotly_dark")
            else:
                fig_pca = px.scatter_3d(df, x="PCA_1", y="PCA_2", z="PCA_3", color="Cluster_Label", color_discrete_sequence=color_palette, opacity=0.85, title="PCA 3D Projection")
                fig_pca.update_traces(marker=dict(size=5))
                fig_pca.update_layout(height=550, template="plotly_dark", scene=dict(aspectmode='cube'))
            st.plotly_chart(fig_pca, use_container_width=True)

        with vis_tab3:
            if view_dim == "2D Mode (2 Features)":
                fig_tsne = px.scatter(df, x="tSNE_1", y="tSNE_2", color="Cluster_Label", color_discrete_sequence=color_palette, opacity=0.85, title="t-SNE 2D Projection")
                fig_tsne.update_traces(marker=dict(size=8))
                fig_tsne.update_layout(height=500, template="plotly_dark")
            else:
                fig_tsne = px.scatter_3d(df, x="tSNE_1", y="tSNE_2", z="tSNE_3", color="Cluster_Label", color_discrete_sequence=color_palette, opacity=0.85, title="t-SNE 3D Projection")
                fig_tsne.update_traces(marker=dict(size=5))
                fig_tsne.update_layout(height=550, template="plotly_dark", scene=dict(aspectmode='cube'))
            st.plotly_chart(fig_tsne, use_container_width=True)
            
    with col2:
        st.subheader("Model Evaluation")
        
        n_clusters_found = len([c for c in set(labels) if c != -1])
        
        if n_clusters_found < 2:
            st.warning("⚠️ Bandwidth is too high! All points collapsed into 1 cluster. Lower the quantile slider or click Auto-Tune.")
            
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="Silhouette Score", value=f"{metrics['silhouette']:.3f}", help="Higher is better (-1 to 1). Requires >= 2 clusters.")
            if algorithm == "MeanShift":
                st.metric(label="Bandwidth", value=f"{computed_bw:.2f}", help="Calculated radius for density estimation.")
            else:
                dbi_val = f"{metrics['davies_bouldin']:.3f}" if metrics['davies_bouldin'] is not None else "N/A"
                st.metric(label="Davies-Bouldin Index", value=dbi_val, help="Lower is better (min 0).")
                
        with m_col2:
            if current_inertia is not None:
                st.metric(label="Inertia (WCSS)", value=f"{current_inertia:.1f}", help="Lower is better. K-Means only.")
            else:
                st.metric(label="Inertia", value="N/A", help="Inertia applies only to K-Means.")
                
            st.metric(label="Noise Ratio", value=f"{metrics['noise_ratio']:.1f}%", help="Percentage of outliers.")

        n_noise = sum(labels == -1)
        st.write(f"**Total Clusters Detected:** {n_clusters_found}")
        if -1 in labels:
            st.write(f"**Outliers / Noise Points:** {n_noise}")

        st.subheader("Cluster Summary")
        if 'exam_score' in df.columns:
            summary = df.groupby('Cluster_Label', observed=False)['exam_score'].agg(['count', 'mean']).reset_index()
            summary.columns = ['Cluster', 'Count', 'Avg Exam Score']
            summary['Cluster'] = summary['Cluster'].astype(str) 
            summary['Avg Exam Score'] = summary['Avg Exam Score'].round(2)
            st.dataframe(summary, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.header("📈 Parameter Evaluation & Methods")
    eval_tab1, eval_tab2 = st.tabs([f"📊 {algorithm} Evaluation", "⚔️ Cross-Algorithm Comparison"])

    with eval_tab1:
        col_t1, col_t2 = st.columns([1, 1.2])
        if algorithm == "K-Means":
            eval_table = get_kmeans_evaluation_table(scaled_matrix)
            with col_t1:
                st.markdown("**Evaluation Results Table:**")
                st.dataframe(eval_table, hide_index=True, use_container_width=True)
            with col_t2:
                st.markdown("**Elbow Curve & Silhouette Trend:**")
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(x=eval_table['K'], y=eval_table['Inertia'], mode='lines+markers', name='Inertia (WCSS)', line=dict(color='#ff4b4b', width=3)))
                fig_elbow.add_trace(go.Scatter(x=eval_table['K'], y=eval_table['Silhouette Score'], mode='lines+markers', name='Silhouette Score', yaxis='y2', line=dict(color='#00d26a', width=3)))
                fig_elbow.update_layout(height=380, template="plotly_dark", xaxis=dict(title=dict(text="Number of Clusters (K)")), yaxis=dict(title=dict(text="Inertia (WCSS)", font=dict(color="#ff4b4b")), tickfont=dict(color="#ff4b4b")), yaxis2=dict(title=dict(text="Silhouette Score", font=dict(color="#00d26a")), tickfont=dict(color="#00d26a"), overlaying="y", side="right"), legend=dict(x=0.55, y=0.95))
                st.plotly_chart(fig_elbow, use_container_width=True)

        elif algorithm == "MeanShift":
            eval_table = get_meanshift_evaluation_table(scaled_matrix)
            with col_t1:
                st.markdown("**MeanShift Evaluation Table:**")
                st.dataframe(eval_table, hide_index=True, use_container_width=True)
            with col_t2:
                st.markdown("**Bandwidth & Silhouette Trend:**")
                fig_ms = go.Figure()
                fig_ms.add_trace(go.Scatter(x=eval_table['Quantile'], y=eval_table['Bandwidth'], mode='lines+markers', name='Bandwidth', line=dict(color='#ff4b4b', width=3)))
                fig_ms.add_trace(go.Scatter(x=eval_table['Quantile'], y=eval_table['Silhouette Score'], mode='lines+markers', name='Silhouette Score', yaxis='y2', line=dict(color='#00d26a', width=3)))
                fig_ms.update_layout(height=380, template="plotly_dark", xaxis=dict(title=dict(text="Quantile")), yaxis=dict(title=dict(text="Bandwidth", font=dict(color="#ff4b4b")), tickfont=dict(color="#ff4b4b")), yaxis2=dict(title=dict(text="Silhouette Score", font=dict(color="#00d26a")), tickfont=dict(color="#00d26a"), overlaying="y", side="right"), legend=dict(x=0.05, y=0.95))
                st.plotly_chart(fig_ms, use_container_width=True)

        elif algorithm == "DBSCAN":
            eval_data = get_dbscan_evaluation_data(scaled_matrix)
            with col_t1:
                st.markdown("**Top 10 DBSCAN Configurations:**")
                if not eval_data.empty:
                    top_10 = eval_data.sort_values(by=['silhouette', 'noise_ratio'], ascending=[False, True]).head(10)
                    st.dataframe(top_10, hide_index=True, use_container_width=True)
                else:
                    st.warning("No configurations found with >= 2 clusters under 50% noise threshold.")
            with col_t2:
                st.markdown("**Silhouette Score Heatmap (EPS vs Min Samples):**")
                if not eval_data.empty:
                    pivot_df = eval_data.pivot(index='eps', columns='min_samples', values='silhouette').fillna(0)
                    
                    fig_hm = go.Figure(data=go.Heatmap(
                        z=pivot_df.values, 
                        x=pivot_df.columns, 
                        y=pivot_df.index, 
                        colorscale='Viridis',
                        colorbar=dict(title="Silhouette")
                    ))
                    fig_hm.update_layout(
                        height=380, 
                        template="plotly_dark", 
                        xaxis=dict(title=dict(text="Min Samples")), 
                        yaxis=dict(title=dict(text="EPS"))
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)
                else:
                    st.info("Heatmap unavailable until valid configurations are found.")

    with eval_tab2:
        st.subheader("⚔️ Comparison of All 3 Clustering Algorithms")
        st.caption("Evaluated on the current feature selection. Automatically tuned or manually adjusted for direct comparison.")

        if 'comp_k' not in st.session_state:
            st.session_state['comp_k'] = optimize_kmeans_params(scaled_matrix)
        if 'comp_q' not in st.session_state:
            st.session_state['comp_q'] = optimize_meanshift_params(scaled_matrix)
        if 'comp_eps' not in st.session_state or 'comp_min' not in st.session_state:
            e_opt, m_opt = optimize_dbscan_params(scaled_matrix)
            st.session_state['comp_eps'] = float(round(e_opt / 0.05) * 0.05)
            st.session_state['comp_min'] = int(m_opt)

        col_hdr, col_btn = st.columns([2.2, 1])
        with col_hdr:
            st.markdown("**Adjust Comparison Parameters:**")
        with col_btn:
            if st.button("🤖 Auto-Tune All Models", use_container_width=True, key="autotune_all_comp"):
                with st.spinner("Optimizing parameters for all 3 algorithms..."):
                    st.session_state['comp_k'] = optimize_kmeans_params(scaled_matrix)
                    st.session_state['comp_q'] = optimize_meanshift_params(scaled_matrix)
                    e_opt, m_opt = optimize_dbscan_params(scaled_matrix)
                    st.session_state['comp_eps'] = float(round(e_opt / 0.05) * 0.05)
                    st.session_state['comp_min'] = int(m_opt)
                    st.rerun()

        col_km, col_ms, col_db = st.columns(3)
        
        with col_km:
            st.markdown("**K-Means**")
            comp_k = st.slider("K value for K-Means", min_value=2, max_value=10, key="comp_k")
            
        with col_ms:
            st.markdown("**MeanShift**")
            comp_q = st.slider("Quantile for MeanShift", min_value=0.01, max_value=0.50, step=0.01, key="comp_q")
            
        with col_db:
            st.markdown("**DBSCAN**")
            comp_eps = st.slider("EPS for DBSCAN", min_value=0.05, max_value=2.00, step=0.05, key="comp_eps")
            comp_min = st.slider("Min Samples for DBSCAN", min_value=2, max_value=20, key="comp_min")

        km_m, km_l, km_met, km_i = run_kmeans_model(scaled_matrix, n_clusters=comp_k)
        ms_m, ms_l, ms_met, ms_bw = run_meanshift_model(scaled_matrix, quantile=comp_q)
        db_m, db_l, db_met = run_dbscan_model(scaled_matrix, eps=comp_eps, min_samples=comp_min)

        st.markdown("---")

        st.markdown("📊 **Silhouette Score Comparison**")
        fig_bar = go.Figure()
        
        fig_bar.add_trace(go.Bar(
            name='K-Means', x=['K-Means'], y=[km_met['silhouette']], 
            marker_color='#ff4b4b', text=[f"{km_met['silhouette']:.3f}"], textposition='auto'
        ))
        fig_bar.add_trace(go.Bar(
            name='MeanShift', x=['MeanShift'], y=[ms_met['silhouette']], 
            marker_color='#00d26a', text=[f"{ms_met['silhouette']:.3f}"], textposition='auto'
        ))
        fig_bar.add_trace(go.Bar(
            name='DBSCAN', x=['DBSCAN'], y=[db_met['silhouette']], 
            marker_color='#ffa500', text=[f"{db_met['silhouette']:.3f}"], textposition='auto'
        ))
        
        fig_bar.update_layout(
            height=350, 
            template="plotly_dark", 
            yaxis=dict(title='Silhouette Score', range=[-1, 1.1]),
            xaxis=dict(title='Algorithm'),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        st.markdown("📋 **Detailed Metrics Summary**")
        comp_data = [
            {
                "Algorithm": f"K-Means (K={comp_k})", 
                "Clusters": len([c for c in set(km_l) if c != -1]), 
                "Silhouette Score": km_met['silhouette'], 
                "Davies-Bouldin": km_met['davies_bouldin'] if km_met['davies_bouldin'] is not None else "N/A", 
                "Inertia": km_i, 
                "Noise Ratio (%)": f"{km_met['noise_ratio']}%", 
                "Best For": "Spherical, equal-sized clusters"
            },
            {
                "Algorithm": "MeanShift", 
                "Clusters": len([c for c in set(ms_l) if c != -1]), 
                "Silhouette Score": ms_met['silhouette'], 
                "Davies-Bouldin": ms_met['davies_bouldin'] if ms_met['davies_bouldin'] is not None else "N/A", 
                "Inertia": "N/A", 
                "Noise Ratio (%)": f"{ms_met['noise_ratio']}%", 
                "Best For": "Automatic cluster count finding"
            },
            {
                "Algorithm": "DBSCAN", 
                "Clusters": len([c for c in set(db_l) if c != -1]), 
                "Silhouette Score": db_met['silhouette'], 
                "Davies-Bouldin": db_met['davies_bouldin'] if db_met['davies_bouldin'] is not None else "N/A", 
                "Inertia": "N/A", 
                "Noise Ratio (%)": f"{db_met['noise_ratio']}%", 
                "Best For": "Arbitrary shapes & outlier detection"
            }
        ]
        st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)

elif app_mode == "🔮 User Data Predictor":
    st.header("🔮 Custom Student Cluster Prediction Engine")
    st.write("Input custom student habits to predict their K-Means cluster assignment and visualize where they land relative to the dataset.")

    forced_k = st.session_state.get('km_k', 3)
    km_model_user, km_labels_user, _, _ = run_kmeans_model(scaled_matrix, n_clusters=forced_k)
    
    km_df = df.copy()
    km_df['Cluster_Label'] = [f"Cluster {lbl:02d}" for lbl in km_labels_user]
    km_df['Cluster_Label'] = pd.Categorical(
        km_df['Cluster_Label'], 
        categories=sorted(km_df['Cluster_Label'].unique()), 
        ordered=True
    )

    with st.form(key='main_prediction_form'):
        st.subheader("1. Enter Student Habits")
        
        num_cols = min(len(selected_features), 3)
        form_cols = st.columns(num_cols)
        user_inputs = {}
        
        for idx, feat in enumerate(selected_features):
            col_target = form_cols[idx % num_cols]
            default_val = float(df[feat].mean())
            user_inputs[feat] = col_target.number_input(f"{feat}", value=round(default_val, 2))
            
        submit_button = st.form_submit_button(label="🚀 Predict Cluster & Plot Position", use_container_width=True)

    if submit_button:
        st.markdown("---")
        st.subheader("2. Prediction Results & Position Analysis")
        
        input_df = pd.DataFrame([user_inputs])
        scaled_input = scaler.transform(input_df)
        
        pred_label = km_model_user.predict(scaled_input)[0]
        pred_label_str = f"Cluster {pred_label:02d}"
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.success(f"### Predicted K-Means Assignment: **{pred_label_str}**")
        with res_col2:
            if 'exam_score' in km_df.columns:
                cluster_students = km_df[km_labels_user == pred_label]
                if not cluster_students.empty:
                    avg_score = cluster_students['exam_score'].mean()
                    st.info(f"### Expected Exam Score: **{avg_score:.2f}**")

        st.subheader(f"3. Custom Input Location vs K-Means Clusters (K={forced_k})")
        
        if view_dim == "2D Mode (2 Features)":
            fig_user_plot = px.scatter(
                km_df, x=feat_x, y=feat_y, color="Cluster_Label",
                color_discrete_sequence=px.colors.qualitative.Bold,
                opacity=0.4, title=f"User Position Overlay ({feat_x} vs {feat_y})"
            )
            fig_user_plot.update_traces(marker=dict(size=8))
            
            fig_user_plot.add_trace(go.Scatter(
                x=[user_inputs[feat_x]], y=[user_inputs[feat_y]],
                mode='markers',
                marker=dict(size=18, color='#FFD700', symbol='star', line=dict(width=2, color='black')),
                name='⭐ Your Custom Input'
            ))
            fig_user_plot.update_layout(height=520, template="plotly_dark")
        else:
            fig_user_plot = px.scatter_3d(
                km_df, x=feat_x, y=feat_y, z=feat_z, color="Cluster_Label",
                color_discrete_sequence=px.colors.qualitative.Bold,
                opacity=0.4, title=f"User Position Overlay 3D ({feat_x}, {feat_y}, {feat_z})"
            )
            fig_user_plot.update_traces(marker=dict(size=5))
            
            fig_user_plot.add_trace(go.Scatter3d(
                x=[user_inputs[feat_x]], y=[user_inputs[feat_y]], z=[user_inputs[feat_z]],
                mode='markers',
                marker=dict(size=12, color='#FFD700', symbol='diamond', line=dict(width=2, color='black')),
                name='⭐ Your Custom Input'
            ))
            fig_user_plot.update_layout(height=600, template="plotly_dark", scene=dict(aspectmode='cube'))
            
        st.plotly_chart(fig_user_plot, use_container_width=True)