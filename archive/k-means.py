import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# 1. DATASET FILES
# ============================================================

DATASETS = {
    "Dataset 1": "Data/student_habits_performance.csv",
}


# ============================================================
# 2. CLUSTERING FEATURES
# ============================================================
# These features are used to CREATE the clusters.
#
# IMPORTANT:
# exam_score is NOT included here.
# This keeps the clustering unsupervised and allows
# exam_score to be used only for interpretation later.
# ============================================================

DATASET_FEATURES = {

    "Dataset 1": [
        "study_hours_per_day",
        "social_media_hours",
        "attendance_percentage",
        "sleep_hours"
    ],

}


# ============================================================
# 3. INTERPRETATION FEATURES
# ============================================================
# These features are NOT used to create clusters.
# They are only used after clustering to understand
# the characteristics of each cluster.
# ============================================================

INTERPRETATION_FEATURES = {

    "Dataset 1": [
        "exam_score"
    ],

}


# ============================================================
# 4. LOAD DATASET
# ============================================================

def load_dataset(filepath):

    df = pd.read_csv(filepath)

    return df


# ============================================================
# 5. SELECT CLUSTERING FEATURES
# ============================================================

def select_features(df, features):

    X = df[features].copy()

    return X


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

def check_missing_values(X):

    print("\nMissing values:")

    missing_values = X.isnull().sum()

    print(missing_values)


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(X):

    X = X.copy()

    # --------------------------------------------------------
    # Numerical columns
    # --------------------------------------------------------

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns


    # Fill numerical missing values with median

    for column in numerical_columns:

        if X[column].isnull().sum() > 0:

            X[column] = X[column].fillna(
                X[column].median()
            )


    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns


    # Fill categorical missing values with mode

    for column in categorical_columns:

        if X[column].isnull().sum() > 0:

            X[column] = X[column].fillna(
                X[column].mode()[0]
            )


    return X


# ============================================================
# 8. ENCODE CATEGORICAL FEATURES
# ============================================================

def encode_features(X):

    X_encoded = pd.get_dummies(
        X,
        drop_first=True
    )

    return X_encoded


# ============================================================
# 9. FEATURE SCALING
# ============================================================

def scale_features(X):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


# ============================================================
# 10. ELBOW METHOD + SILHOUETTE SCORE
# ============================================================

def evaluate_k_values(
    X_scaled,
    max_k=10
):

    inertia = []

    silhouette_scores = []

    k_values = range(
        2,
        max_k + 1
    )


    for k in k_values:

        # ----------------------------------------------------
        # Create K-Means model
        # ----------------------------------------------------

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )


        # ----------------------------------------------------
        # Train K-Means
        # ----------------------------------------------------

        labels = kmeans.fit_predict(
            X_scaled
        )


        # ----------------------------------------------------
        # Calculate Inertia
        # ----------------------------------------------------

        inertia.append(
            kmeans.inertia_
        )


        # ----------------------------------------------------
        # Calculate Silhouette Score
        # ----------------------------------------------------

        score = silhouette_score(
            X_scaled,
            labels
        )

        silhouette_scores.append(
            score
        )


    return (
        list(k_values),
        inertia,
        silhouette_scores
    )


# ============================================================
# 11. DISPLAY ELBOW GRAPH
# ============================================================

def plot_elbow(
    k_values,
    inertia,
    dataset_name
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        inertia,
        marker="o"
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(
        "Inertia"
    )

    plt.title(
        f"Elbow Method - {dataset_name}"
    )

    plt.xticks(
        k_values
    )

    plt.grid(
        True
    )

    plt.show()


# ============================================================
# 12. DISPLAY SILHOUETTE GRAPH
# ============================================================

def plot_silhouette(
    k_values,
    silhouette_scores,
    dataset_name
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        silhouette_scores,
        marker="o"
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(
        "Silhouette Score"
    )

    plt.title(
        f"Silhouette Score - {dataset_name}"
    )

    plt.xticks(
        k_values
    )

    plt.grid(
        True
    )

    plt.show()


# ============================================================
# 13. TRAIN FINAL K-MEANS
# ============================================================

def train_final_kmeans(
    X_scaled,
    best_k
):

    print("\n" + "-" * 60)

    print("TRAINING FINAL K-MEANS")

    print("-" * 60)


    # --------------------------------------------------------
    # Create final K-Means model
    # --------------------------------------------------------

    final_kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10
    )


    # --------------------------------------------------------
    # Train final K-Means
    # --------------------------------------------------------

    cluster_labels = final_kmeans.fit_predict(
        X_scaled
    )


    print(
        "Final K-Means training completed."
    )

    print(
        "Final K:",
        best_k
    )


    return (
        final_kmeans,
        cluster_labels
    )


# ============================================================
# 14. GET CLUSTER RESULTS
# ============================================================

def get_cluster_results(
    df,
    cluster_labels
):

    df_result = df.copy()


    # Add cluster labels to original dataset

    df_result["Cluster"] = cluster_labels


    print("\n" + "-" * 60)

    print("CLUSTER RESULTS")

    print("-" * 60)


    # --------------------------------------------------------
    # Number of students in each cluster
    # --------------------------------------------------------

    print(
        "\nNumber of students in each cluster:"
    )


    cluster_counts = (
        df_result["Cluster"]
        .value_counts()
        .sort_index()
    )


    print(
        cluster_counts
    )


    return df_result


# ============================================================
# 15. INTERPRET CLUSTERS
# ============================================================

def interpret_clusters(
    df_result,
    clustering_features,
    interpretation_features
):

    print("\n" + "-" * 60)

    print("CLUSTER CHARACTERISTICS")

    print("-" * 60)


    # --------------------------------------------------------
    # Calculate mean of clustering features
    # --------------------------------------------------------

    clustering_summary = (
        df_result
        .groupby("Cluster")[clustering_features]
        .mean()
    )


    # --------------------------------------------------------
    # Calculate mean of interpretation features
    # --------------------------------------------------------

    interpretation_summary = (
        df_result
        .groupby("Cluster")[interpretation_features]
        .mean()
    )


    # --------------------------------------------------------
    # Combine both summaries
    # --------------------------------------------------------

    cluster_summary = pd.concat(
        [
            clustering_summary,
            interpretation_summary
        ],
        axis=1
    )


    print(
        cluster_summary.to_string()
    )


    return cluster_summary


# ============================================================
# 16. DISPLAY CLUSTERING GRAPH
# ============================================================
# The graph shows:
#
# X-axis = Study Hours
# Y-axis = Attendance
# Colour/group = K-Means Cluster
#
# IMPORTANT:
# This graph is only a 2D visualisation.
# K-Means itself uses all clustering features.
# ============================================================

def plot_clusters(
    df_result,
    dataset_name
):

    plt.figure(
        figsize=(9, 6)
    )


    # --------------------------------------------------------
    # Get all clusters
    # --------------------------------------------------------

    clusters = sorted(
        df_result["Cluster"].unique()
    )


    # --------------------------------------------------------
    # Plot each cluster
    # --------------------------------------------------------

    for cluster in clusters:

        cluster_data = df_result[
            df_result["Cluster"] == cluster
        ]


        plt.scatter(
            cluster_data[
                "study_hours_per_day"
            ],
            cluster_data[
                "attendance_percentage"
            ],
            label=f"Cluster {cluster}",
            alpha=0.6
        )


    # --------------------------------------------------------
    # Labels and title
    # --------------------------------------------------------

    plt.xlabel(
        "Study Hours per Day"
    )

    plt.ylabel(
        "Attendance Percentage"
    )

    plt.title(
        f"K-Means Clusters: Study Hours vs Attendance - {dataset_name}"
    )


    plt.legend()

    plt.grid(
        True
    )

    plt.show()


# ============================================================
# 17. PROCESS ONE DATASET
# ============================================================

def process_dataset(
    dataset_name,
    file_path,
    clustering_features,
    interpretation_features
):

    print("\n" + "=" * 60)

    print(
        dataset_name
    )

    print("=" * 60)


    # ========================================================
    # LOAD DATASET
    # ========================================================

    df = load_dataset(
        file_path
    )


    print("\nDataset shape:")

    print(
        df.shape
    )


    # ========================================================
    # SELECT CLUSTERING FEATURES
    # ========================================================

    X = select_features(
        df,
        clustering_features
    )


    print("\nClustering features:")

    print(
        X.columns.tolist()
    )


    print(
        "\nThese features will be used to create the clusters."
    )


    print(
        "\nInterpretation feature(s):"
    )

    print(
        interpretation_features
    )


    print(
        "\nExam score is NOT used to create clusters."
    )


    # ========================================================
    # CHECK DATA TYPES
    # ========================================================

    print("\nData types:")

    print(
        X.dtypes
    )


    # ========================================================
    # CHECK MISSING VALUES
    # ========================================================

    check_missing_values(
        X
    )


    # ========================================================
    # HANDLE MISSING VALUES
    # ========================================================

    X = handle_missing_values(
        X
    )


    print(
        "\nMissing values handled."
    )


    # ========================================================
    # ENCODE CATEGORICAL FEATURES
    # ========================================================

    X = encode_features(
        X
    )


    print(
        "\nEncoding completed."
    )


    print(
        "Number of clustering features after encoding:",
        X.shape[1]
    )


    # ========================================================
    # FEATURE SCALING
    # ========================================================

    (
        X_scaled,
        scaler
    ) = scale_features(
        X
    )


    print(
        "\nScaling completed."
    )


    # ========================================================
    # EVALUATE K VALUES
    # ========================================================

    (
        k_values,
        inertia,
        silhouette_scores
    ) = evaluate_k_values(
        X_scaled,
        max_k=10
    )


    # ========================================================
    # DISPLAY EVALUATION RESULTS
    # ========================================================

    print(
        "\nEvaluation Results:"
    )

    print(
        "-" * 50
    )


    print(
        f"{'K':<10}"
        f"{'Inertia':<20}"
        f"{'Silhouette Score':<20}"
    )


    print(
        "-" * 50
    )


    for i in range(
        len(k_values)
    ):

        print(
            f"{k_values[i]:<10}"
            f"{inertia[i]:<20.2f}"
            f"{silhouette_scores[i]:<20.4f}"
        )


    # ========================================================
    # FIND BEST K
    # ========================================================

    best_index = silhouette_scores.index(
        max(silhouette_scores)
    )


    best_k = k_values[
        best_index
    ]


    best_score = silhouette_scores[
        best_index
    ]


    print(
        "\nBest K based on Silhouette Score:",
        best_k
    )


    print(
        "Best Silhouette Score:",
        round(
            best_score,
            4
        )
    )


    # ========================================================
    # DISPLAY ELBOW GRAPH
    # ========================================================

    plot_elbow(
        k_values,
        inertia,
        dataset_name
    )


    # ========================================================
    # DISPLAY SILHOUETTE GRAPH
    # ========================================================

    plot_silhouette(
        k_values,
        silhouette_scores,
        dataset_name
    )


    # ========================================================
    # TRAIN FINAL K-MEANS
    # ========================================================

    (
        final_kmeans,
        cluster_labels
    ) = train_final_kmeans(
        X_scaled,
        best_k
    )


    # ========================================================
    # GET CLUSTER RESULTS
    # ========================================================

    df_result = get_cluster_results(
        df,
        cluster_labels
    )


    # ========================================================
    # INTERPRET CLUSTERS
    # ========================================================

    cluster_summary = interpret_clusters(
        df_result,
        clustering_features,
        interpretation_features
    )


    # ========================================================
    # DISPLAY CLUSTERING GRAPH
    # ========================================================

    plot_clusters(
        df_result,
        dataset_name
    )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return (
        df_result,
        final_kmeans,
        scaler,
        best_k,
        best_score,
        cluster_summary
    )


# ============================================================
# 18. MAIN PROGRAM
# ============================================================

def main():

    for dataset_name, csvfile in DATASETS.items():


        # ----------------------------------------------------
        # Get clustering features
        # ----------------------------------------------------

        clustering_features = DATASET_FEATURES[
            dataset_name
        ]


        # ----------------------------------------------------
        # Get interpretation features
        # ----------------------------------------------------

        interpretation_features = INTERPRETATION_FEATURES[
            dataset_name
        ]


        # ----------------------------------------------------
        # Process dataset
        # ----------------------------------------------------

        (
            df_result,
            final_kmeans,
            scaler,
            best_k,
            best_score,
            cluster_summary
        ) = process_dataset(
            dataset_name,
            csvfile,
            clustering_features,
            interpretation_features
        )


# ============================================================
# 19. RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()