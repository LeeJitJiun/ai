import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from helper_functions import (
    load_data, preprocess_features,
    run_kmeans_model, run_meanshift_model, run_dbscan_model,
    get_kmeans_evaluation_table, get_meanshift_evaluation_table, get_dbscan_evaluation_data,
    optimize_kmeans_params, optimize_meanshift_params, optimize_dbscan_params
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

feat_x = st.sidebar.selectbox("Select X-Axis Feature", numeric_cols, index=0)
feat_y = st.sidebar.selectbox("Select Y-Axis Feature", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

if view_dim == "2D Mode (2 Features)":
    selected_features = list(dict.fromkeys([feat_x, feat_y]))
else:
    feat_z = st.sidebar.selectbox("Select Z-Axis Feature", numeric_cols, index=2 if len(numeric_cols) > 2 else 0)
    selected_features = list(dict.fromkeys([feat_x, feat_y, feat_z]))

st.sidebar.markdown("---")
st.sidebar.header("🤖 Clustering Algorithm")

algorithm = st.sidebar.selectbox("Choose Algorithm", ["K-Means", "MeanShift", "DBSCAN"])

scaled_matrix, encoded_df, scaler = preprocess_features(df, tuple(selected_features))
current_inertia = None

if algorithm == "K-Means":
    if 'km_k' not in st.session_state:
        st.session_state['km_k'] = 3
        
    if st.sidebar.button("🤖 Auto-Tune K-Means"):
        best_k = optimize_kmeans_params(scaled_matrix)
        st.session_state['km_k'] = int(best_k)
        st.sidebar.success(f"Optimal K found: {best_k}")
        
    k = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, key='km_k')
    model, labels, metrics, current_inertia = run_kmeans_model(scaled_matrix, n_clusters=k)

elif algorithm == "MeanShift":
    if 'ms_q' not in st.session_state:
        st.session_state['ms_q'] = 0.20
        
    if st.sidebar.button("🤖 Auto-Tune MeanShift"):
        best_q = optimize_meanshift_params(scaled_matrix)
        st.session_state['ms_q'] = float(best_q)
        st.sidebar.success(f"Optimal Quantile found: {best_q:.2f}")
        
    quantile = st.sidebar.slider("Bandwidth Quantile (Density)", min_value=0.05, max_value=0.50, step=0.05, key='ms_q')
    model, labels, metrics, computed_bw = run_meanshift_model(scaled_matrix, quantile=quantile)
    st.sidebar.info(f"Computed Bandwidth: **{computed_bw:.2f}**")

elif algorithm == "DBSCAN":
    if 'db_eps' not in st.session_state:
        st.session_state['db_eps'] = 0.10
    if 'db_min' not in st.session_state:
        st.session_state['db_min'] = 4
        
    if st.sidebar.button("🤖 Auto-Tune DBSCAN"):
        e, m = optimize_dbscan_params(scaled_matrix)
        st.session_state['db_eps'] = float(e)
        st.session_state['db_min'] = int(m)
        st.sidebar.success(f"Optimal EPS: {e:.2f}, Min Samples: {m}")
        
    eps = st.sidebar.slider("EPS (Neighborhood Radius)", min_value=0.05, max_value=5.0, step=0.05, key='db_eps')
    min_samples = st.sidebar.slider("Min Samples", min_value=2, max_value=20, key='db_min')
    model, labels, metrics = run_dbscan_model(scaled_matrix, eps=eps, min_samples=min_samples)
    
st.sidebar.markdown("---")
st.sidebar.header("🔮 Custom Prediction Engine")
st.sidebar.write("Input student habits to predict their cluster.")

# Wrap the inputs in a form to prevent auto-rerunning
with st.sidebar.form(key='prediction_form'):
    user_inputs = {}
    for feat in selected_features:
        default_val = float(df[feat].mean()) # Default to dataset average
        user_inputs[feat] = st.number_input(f"{feat}", value=default_val)
    
    # The submit button replaces your old st.button
    submit_button = st.form_submit_button(label="🚀 Run Prediction")

# ---------------- MAIN CONTENT AREA ----------------
df['Cluster'] = labels

# 1. Get numerically sorted unique IDs
sorted_cluster_nums = sorted(df['Cluster'].unique())

# 2. Create labels WITH A LEADING ZERO (Cluster 01, Cluster 02, etc.)
ordered_labels = [f"Noise (-1)" if x == -1 else f"Cluster {x:02d}" for x in sorted_cluster_nums]

# 3. Apply to dataframe
df['Cluster_Label'] = df['Cluster'].apply(lambda x: f"Noise (-1)" if x == -1 else f"Cluster {x:02d}")

# 4. Lock for Plotly graphs
df['Cluster_Label'] = pd.Categorical(df['Cluster_Label'], categories=ordered_labels, ordered=True)

col1, col2 = st.columns([2.2, 1])

with col1:
    st.subheader(f"{algorithm} - {view_dim}")
    color_palette = px.colors.qualitative.Bold
    
    # --- DIMENSIONALITY REDUCTION (PCA & t-SNE) ---
    n_components = 2 if view_dim == "2D Mode (2 Features)" else 3
    
    # Calculate PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_matrix)
    
    # Calculate t-SNE (adjust perplexity to avoid errors on small datasets)
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(scaled_matrix)-1))
    tsne_result = tsne.fit_transform(scaled_matrix)
    
    # Add coordinates to dataframe for easy plotting
    df['PCA_1'], df['PCA_2'] = pca_result[:, 0], pca_result[:, 1]
    df['tSNE_1'], df['tSNE_2'] = tsne_result[:, 0], tsne_result[:, 1]
    
    if n_components == 3:
        df['PCA_3'] = pca_result[:, 2]
        df['tSNE_3'] = tsne_result[:, 2]

    # --- TABS FOR VISUALIZATION ---
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
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric(
            label="Silhouette Score", 
            value=f"{metrics['silhouette']:.3f}", 
            help="Higher is better (-1 to 1)."
        )
        
        # --- DYNAMIC METRIC SWAP ---
        if algorithm == "MeanShift":
            st.metric(
                label="Bandwidth", 
                value=f"{computed_bw:.2f}", 
                help="The calculated radius for density estimation based on the quantile."
            )
        else:
            dbi_val = f"{metrics['davies_bouldin']:.3f}" if metrics['davies_bouldin'] is not None else "N/A"
            st.metric(
                label="Davies-Bouldin Index", 
                value=dbi_val, 
                help="Lower is better (min 0). Ideal for K-Means."
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
        
        # Convert back to standard string so Streamlit's interactive sorting works
        summary['Cluster'] = summary['Cluster'].astype(str) 
        
        summary['Avg Exam Score'] = summary['Avg Exam Score'].round(2)
        st.dataframe(summary, hide_index=True, use_container_width=True)

st.markdown("---")

# ---------------- DYNAMIC EVALUATION SECTION ----------------
st.header("📈 Parameter Evaluation & Methods")

tab1, tab2 = st.tabs([f"📊 {algorithm} Evaluation", "⚔️ Cross-Algorithm Comparison"])

with tab1:
    col_t1, col_t2 = st.columns([1, 1.2])
    
    if algorithm == "K-Means":
        eval_table = get_kmeans_evaluation_table(scaled_matrix)
        
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
            
            # Modern Plotly syntax for title fonts
            fig_elbow.update_layout(
                height=380,
                template="plotly_dark",
                xaxis=dict(title=dict(text="Number of Clusters (K)")),
                yaxis=dict(
                    title=dict(text="Inertia (WCSS)", font=dict(color="#ff4b4b")),
                    tickfont=dict(color="#ff4b4b")
                ),
                yaxis2=dict(
                    title=dict(text="Silhouette Score", font=dict(color="#00d26a")),
                    tickfont=dict(color="#00d26a"),
                    overlaying="y",
                    side="right"
                ),
                legend=dict(x=0.55, y=0.95)
            )
            st.plotly_chart(fig_elbow, use_container_width=True)

    elif algorithm == "MeanShift":
        eval_table = get_meanshift_evaluation_table(scaled_matrix)
        
        with col_t1:
            st.markdown("**MeanShift Evaluation Table:**")
            st.dataframe(eval_table, hide_index=True, use_container_width=True)
            
        with col_t2:
            st.markdown("**Bandwidth & Silhouette Trend:**")
            fig_ms = go.Figure()
            
            fig_ms.add_trace(go.Scatter(
                x=eval_table['Quantile'], y=eval_table['Bandwidth'],
                mode='lines+markers', name='Bandwidth',
                line=dict(color='#ff4b4b', width=3)
            ))
            
            fig_ms.add_trace(go.Scatter(
                x=eval_table['Quantile'], y=eval_table['Silhouette Score'],
                mode='lines+markers', name='Silhouette Score',
                yaxis='y2', line=dict(color='#00d26a', width=3)
            ))
            
            fig_ms.update_layout(
                height=380,
                template="plotly_dark",
                xaxis=dict(title=dict(text="Quantile")),
                yaxis=dict(
                    title=dict(text="Bandwidth", font=dict(color="#ff4b4b")),
                    tickfont=dict(color="#ff4b4b")
                ),
                yaxis2=dict(
                    title=dict(text="Silhouette Score", font=dict(color="#00d26a")),
                    tickfont=dict(color="#00d26a"),
                    overlaying="y",
                    side="right"
                ),
                legend=dict(x=0.05, y=0.95)
            )
            st.plotly_chart(fig_ms, use_container_width=True)

    elif algorithm == "DBSCAN":
        eval_data = get_dbscan_evaluation_data(scaled_matrix)
        
        with col_t1:
            st.markdown("**Top 10 DBSCAN Configurations:**")
            if not eval_data.empty:
                top_10 = eval_data.sort_values(by=['silhouette', 'noise_ratio'], ascending=[False, True]).head(10)
                st.dataframe(top_10, hide_index=True, use_container_width=True)
            else:
                st.write("No valid configurations found within the noise threshold.")
                
        with col_t2:
            st.markdown("**Silhouette Score Heatmap (EPS vs Min Samples):**")
            if not eval_data.empty:
                pivot_df = eval_data.pivot(index='eps', columns='min_samples', values='silhouette')
                fig_hm = go.Figure(data=go.Heatmap(
                    z=pivot_df.values,
                    x=pivot_df.columns,
                    y=pivot_df.index,
                    colorscale='Viridis'
                ))
                fig_hm.update_layout(
                    height=380,
                    template="plotly_dark",
                    xaxis=dict(title=dict(text="Min Samples")),
                    yaxis=dict(title=dict(text="EPS"))
                )
                st.plotly_chart(fig_hm, use_container_width=True)

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

# ---------------- PREDICTION RESULTS AREA ----------------
# Only run this logic IF the user clicks the submit button
if submit_button:
    st.markdown("---")
    st.header("🎯 Custom Prediction Results")
    
    # Convert input to dataframe
    input_df = pd.DataFrame([user_inputs])
    
    # Scale the custom input using the exact same scaler
    scaled_input = scaler.transform(input_df)
    
    # Predict based on algorithm
    if algorithm in ["K-Means", "MeanShift"]:
        pred_label = model.predict(scaled_input)[0]
    elif algorithm == "DBSCAN":
        # DBSCAN Workaround: Find the Nearest Neighbor
        nn = NearestNeighbors(n_neighbors=1).fit(scaled_matrix)
        _, indices = nn.kneighbors(scaled_input)
        pred_label = labels[indices[0][0]]
        
    # Format the output name
    pred_label_str = f"Cluster {pred_label:02d}" if pred_label != -1 else "Noise (-1)"
    st.success(f"**Predicted Assignment:** {pred_label_str}")
    
    # Pull contextual data for that cluster
    if 'exam_score' in df.columns:
        cluster_students = df[labels == pred_label]
        if not cluster_students.empty:
            avg_score = cluster_students['exam_score'].mean()
            st.info(f"Students in {pred_label_str} average an exam score of **{avg_score:.2f}**.")
        elif pred_label == -1:
            st.warning("This student's habits are highly unique and considered Outliers (Noise).")