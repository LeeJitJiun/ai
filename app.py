import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from helper_functions import (
    load_data,
    preprocess_features,
    run_kmeans_model,
    run_meanshift_model,
    calculate_suggested_eps,
    optimize_dbscan_params,
    run_dbscan_model,
    get_kmeans_evaluation_table
)

st.set_page_config(page_title="Student Clustering Dashboard", layout="wide", page_icon="🎓")

st.title("🎓 Student Habits & Performance Clustering")

# Load Data (Cached)
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

numeric_cols = [
    c for c in df.columns 
    if pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in ['student_id', 'exam_score']
]

# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.header("⚙️ Model Controls")

view_dim = st.sidebar.radio("Visualization Mode", ["2D Mode (2 Features)", "3D Mode (3 Features)"])

st.sidebar.markdown("---")
st.sidebar.header("📊 Feature Selection")

if view_dim == "2D Mode (2 Features)":
    feat_x = st.sidebar.selectbox("Select X-Axis Feature", numeric_cols, index=0)
    default_y_idx = 1 if len(numeric_cols) > 1 else 0
    feat_y = st.sidebar.selectbox("Select Y-Axis Feature", numeric_cols, index=default_y_idx)
    selected_features = list(dict.fromkeys([feat_x, feat_y]))
else:
    feat_x = st.sidebar.selectbox("Select X-Axis Feature", numeric_cols, index=0)
    default_y_idx = 1 if len(numeric_cols) > 1 else 0
    feat_y = st.sidebar.selectbox("Select Y-Axis Feature", numeric_cols, index=default_y_idx)
    default_z_idx = 2 if len(numeric_cols) > 2 else 0
    feat_z = st.sidebar.selectbox("Select Z-Axis Feature", numeric_cols, index=default_z_idx)
    selected_features = list(dict.fromkeys([feat_x, feat_y, feat_z]))

st.sidebar.markdown("---")
st.sidebar.header("🤖 Clustering Algorithm")

algorithm = st.sidebar.selectbox("Choose Algorithm", ["K-Means", "MeanShift", "DBSCAN"])

scaled_matrix, encoded_df = preprocess_features(df, tuple(selected_features))

current_inertia = None

if algorithm == "K-Means":
    k = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=8, value=3)
    model, labels, metrics, current_inertia = run_kmeans_model(scaled_matrix, n_clusters=k)

elif algorithm == "MeanShift":
    quantile = st.sidebar.slider("Bandwidth Quantile (Density)", min_value=0.05, max_value=0.50, value=0.20, step=0.05)
    model, labels, metrics, computed_bw = run_meanshift_model(scaled_matrix, quantile=quantile)
    st.sidebar.info(f"Computed Bandwidth: **{computed_bw:.2f}**")

elif algorithm == "DBSCAN":
    # Session State Initialization
    if 'db_eps' not in st.session_state:
        st.session_state.db_eps = 0.10
    if 'db_min_samples' not in st.session_state:
        st.session_state.db_min_samples = 4

    # Single Auto-Tune Button
    if st.sidebar.button("🤖 Auto-Tune DBSCAN"):
        with st.spinner("Finding best parameters..."):
            best_e, best_m = optimize_dbscan_params(scaled_matrix, max_noise_ratio=15.0)
            st.session_state.db_eps = best_e
            st.session_state.db_min_samples = best_m
            st.sidebar.success("Optimal parameters found!")

    # Single set of controls (prevents duplicate rendering)
    eps = st.sidebar.slider(
        "EPS (Neighborhood Radius)", 
        min_value=0.05, max_value=5.0, 
        value=float(st.session_state.db_eps), step=0.05
    )
    min_samples = st.sidebar.slider(
        "Min Samples", 
        min_value=2, max_value=20, 
        value=int(st.session_state.db_min_samples)
    )
    
    st.session_state.db_eps = eps
    st.session_state.db_min_samples = min_samples

    model, labels, metrics = run_dbscan_model(scaled_matrix, eps=eps, min_samples=min_samples)

# ---------------- MAIN CONTENT AREA ----------------
df['Cluster'] = labels
df['Cluster_Label'] = df['Cluster'].apply(lambda x: f"Noise (-1)" if x == -1 else f"Cluster {x}")

col1, col2 = st.columns([2.2, 1])

with col1:
    st.subheader(f"{algorithm} - {view_dim}")
    color_palette = px.colors.qualitative.Bold

    if view_dim == "2D Mode (2 Features)":
        fig = px.scatter(
            df, x=feat_x, y=feat_y, color="Cluster_Label",
            hover_data=["exam_score"] if "exam_score" in df.columns else None,
            color_discrete_sequence=color_palette,
            opacity=0.85
        )
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(
            height=500,
            xaxis_title=feat_x.replace('_', ' ').title(),
            yaxis_title=feat_y.replace('_', ' ').title(),
            legend_title="Clusters",
            template="plotly_dark"
        )
    else:
        fig = px.scatter_3d(
            df, x=feat_x, y=feat_y, z=feat_z, color="Cluster_Label",
            hover_data=["exam_score"] if "exam_score" in df.columns else None,
            color_discrete_sequence=color_palette,
            opacity=0.85
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(
            height=550,
            template="plotly_dark",
            scene=dict(
                xaxis=dict(title=feat_x.replace('_', ' ').title(), showbackground=True, backgroundcolor="#1e1e2f", gridcolor="#444466"),
                yaxis=dict(title=feat_y.replace('_', ' ').title(), showbackground=True, backgroundcolor="#1e1e2f", gridcolor="#444466"),
                zaxis=dict(title=feat_z.replace('_', ' ').title(), showbackground=True, backgroundcolor="#1e1e2f", gridcolor="#444466"),
                aspectmode='cube'
            )
        )
        
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Model Evaluation")
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric(
            label="Silhouette Score", 
            value=f"{metrics['silhouette']:.3f}", 
            help="Higher is better (-1 to 1)."
        )
        dbi_val = f"{metrics['davies_bouldin']:.3f}" if metrics['davies_bouldin'] is not None else "N/A"
        st.metric(
            label="Davies-Bouldin Index", 
            value=dbi_val, 
            help="Lower is better (min 0). Ideal for MeanShift & K-Means."
        )
    with m_col2:
        if current_inertia is not None:
            st.metric(
                label="Inertia (WCSS)", 
                value=f"{current_inertia:.1f}", 
                help="Lower is better. Specific to K-Means."
            )
        else:
            st.metric(label="Inertia", value="N/A", help="Inertia applies only to K-Means.")
            
        st.metric(
            label="Noise Ratio", 
            value=f"{metrics['noise_ratio']:.1f}%", 
            help="Percentage of outliers/noise points. Essential for DBSCAN."
        )

    n_clusters_found = len([c for c in set(labels) if c != -1])
    n_noise = sum(labels == -1)
    
    st.write(f"**Total Clusters Detected:** {n_clusters_found}")
    if -1 in labels:
        st.write(f"**Outliers / Noise Points:** {n_noise}")

    st.subheader("Cluster Summary")
    if 'exam_score' in df.columns:
        summary = df.groupby('Cluster_Label')['exam_score'].agg(['count', 'mean']).reset_index()
        summary.columns = ['Cluster', 'Count', 'Avg Exam Score']
        summary['Avg Exam Score'] = summary['Avg Exam Score'].round(2)
        st.dataframe(summary, hide_index=True, use_container_width=True)

st.markdown("---")

# ---------------- EVALUATION & COMPARISON SECTION ----------------
st.header("📈 Model Evaluation & Method Comparison")

tab1, tab2 = st.tabs(["📊 K-Means Evaluation Table & Elbow Curve", "⚔️ Cross-Algorithm Comparison"])

with tab1:
    st.subheader("Evaluation Results (K = 2 to 10)")
    eval_table = get_kmeans_evaluation_table(scaled_matrix)
    
    col_t1, col_t2 = st.columns([1, 1.2])
    
    with col_t1:
        st.markdown("**Evaluation Results Table:**")
        st.dataframe(eval_table, hide_index=True, use_container_width=True)
        
    with col_t2:
        st.markdown("**Elbow Curve & Silhouette Trend:**")
        fig_elbow = go.Figure()
        
        fig_elbow.add_trace(go.Scatter(
            x=eval_table['K'], y=eval_table['Inertia'],
            mode='lines+markers', name='Inertia (WCSS)',
            line=dict(color='#ff4b4b', width=3)
        ))
        
        fig_elbow.add_trace(go.Scatter(
            x=eval_table['K'], y=eval_table['Silhouette Score'],
            mode='lines+markers', name='Silhouette Score',
            yaxis='y2', line=dict(color='#00d26a', width=3)
        ))
        
        fig_elbow.update_layout(
            height=380,
            template="plotly_dark",
            xaxis=dict(title="Number of Clusters (K)"),
            yaxis=dict(title="Inertia (WCSS)", title_font=dict(color="#ff4b4b"), tickfont=dict(color="#ff4b4b")),
            yaxis2=dict(title="Silhouette Score", title_font=dict(color="#00d26a"), tickfont=dict(color="#00d26a"), overlaying="y", side="right"),
            legend=dict(x=0.55, y=0.95)
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

with tab2:
    st.subheader("Comparison of All 3 Clustering Algorithms")
    st.caption("Evaluated on the current feature selection.")
    
    km_m, km_l, km_met, km_i = run_kmeans_model(scaled_matrix, n_clusters=3)
    ms_m, ms_l, ms_met, ms_bw = run_meanshift_model(scaled_matrix, quantile=0.20)
    db_m, db_l, db_met = run_dbscan_model(scaled_matrix, eps=0.10, min_samples=4)
    
    comp_data = [
        {
            "Algorithm": "K-Means (K=3)",
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
    
    comp_df = pd.DataFrame(comp_data)
    st.dataframe(comp_df, hide_index=True, use_container_width=True)