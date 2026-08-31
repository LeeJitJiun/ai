import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# 1. DATASET FILES
# ============================================================

DATASETS = {
    "Dataset 1": "Data/StudentsPerformance.csv",
    "Dataset 2": "Data/student_habits_performance.csv",
    "Dataset 3": "Data/student_performance.csv",
    "Dataset 4": "Data/student-por.csv",
    "Dataset 5": "Data/StudentPerformanceFactors.csv"
}


# ============================================================
# 2. DATASET FEATURES
# ============================================================

DATASET_FEATURES = {

    "Dataset 1": [
        "test preparation course",
        "math score",
        "reading score",
        "writing score"
    ],

    "Dataset 2": [
        "study_hours_per_day",
        "social_media_hours",
        "attendance_percentage",
        "sleep_hours",
        "exam_score"
    ],

    "Dataset 3": [
        "Math_Score",
        "Science_Score",
        "English_Score"
    ],

    "Dataset 4": [
        "age",
        "famsize",
        "Medu",
        "Fedu",
        "guardian",
        "traveltime",
        "studytime",
        "Failures",
        "absences",
        "G1",
        "G2",
        "G3"
    ],

    "Dataset 5": [
        "Hours_Studied",
        "Attendance",
        "Parental_Involvement",
        "Access_to_Resources",
        "Sleep_Hours",
        "Motivation_Level",
        "Tutoring_Sessions",
        "Exam_Score"
    ]
}


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_dataset(filepath):

    df = pd.read_csv(filepath)

    return df


# ============================================================
# 4. SELECT FEATURES
# ============================================================

def select_features(df, features):

    X = df[features].copy()

    return X


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

def check_missing_values(X):

    print("\nMissing values:")

    missing_values = X.isnull().sum()

    print(missing_values)


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(X):

    X = X.copy()

    # Numerical columns
    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    # Fill numerical missing values with median
    for column in numerical_columns:

        if X[column].isnull().sum() > 0:

            X[column] = X[column].fillna(
                X[column].median()
            )


    # Categorical columns
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
# 7. ENCODE CATEGORICAL FEATURES
# ============================================================

def encode_features(X):

    X_encoded = pd.get_dummies(
        X,
        drop_first=True
    )

    return X_encoded


# ============================================================
# 8. FEATURE SCALING
# ============================================================

def scale_features(X):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled


# ============================================================
# 9. ELBOW METHOD + SILHOUETTE SCORE
# ============================================================

def evaluate_k_values(X_scaled, max_k=10):

    inertia = []
    silhouette_scores = []

    k_values = range(2, max_k + 1)


    for k in k_values:

        # Create K-means model
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )


        # Train K-means
        labels = kmeans.fit_predict(
            X_scaled
        )


        # Inertia
        inertia.append(
            kmeans.inertia_
        )


        # Silhouette Score
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
# 10. DISPLAY ELBOW GRAPH
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

    plt.xticks(k_values)

    plt.grid(True)

    plt.show()


# ============================================================
# 11. DISPLAY SILHOUETTE GRAPH
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

    plt.xticks(k_values)

    plt.grid(True)

    plt.show()


# ============================================================
# 12. PROCESS ONE DATASET
# ============================================================

def process_dataset(
    dataset_name,
    file_path,
    features
):

    print("\n" + "=" * 60)

    print(dataset_name)

    print("=" * 60)


    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset(
        file_path
    )

    print("\nDataset shape:")

    print(df.shape)


    # --------------------------------------------------------
    # Select features
    # --------------------------------------------------------

    X = select_features(
        df,
        features
    )

    print("\nSelected features:")

    print(
        X.columns.tolist()
    )


    # --------------------------------------------------------
    # Check data types
    # --------------------------------------------------------

    print("\nData types:")

    print(
        X.dtypes
    )


    # --------------------------------------------------------
    # Check missing values
    # --------------------------------------------------------

    check_missing_values(
        X
    )


    # --------------------------------------------------------
    # Handle missing values
    # --------------------------------------------------------

    X = handle_missing_values(
        X
    )

    print(
        "\nMissing values handled."
    )


    # --------------------------------------------------------
    # Encode categorical features
    # --------------------------------------------------------

    X = encode_features(
        X
    )

    print(
        "\nEncoding completed."
    )

    print(
        "Number of features after encoding:",
        X.shape[1]
    )


    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    X_scaled = scale_features(
        X
    )

    print(
        "\nScaling completed."
    )


    # --------------------------------------------------------
    # Evaluate K values
    # --------------------------------------------------------

    (
        k_values,
        inertia,
        silhouette_scores
    ) = evaluate_k_values(
        X_scaled,
        max_k=10
    )


    # --------------------------------------------------------
    # Display evaluation results
    # --------------------------------------------------------

    print("\nEvaluation Results:")

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


    # --------------------------------------------------------
    # Find best K based on Silhouette Score
    # --------------------------------------------------------

    best_index = silhouette_scores.index(
        max(silhouette_scores)
    )

    best_k = k_values[best_index]

    best_score = silhouette_scores[best_index]


    print(
        "\nBest K based on Silhouette Score:",
        best_k
    )

    print(
        "Best Silhouette Score:",
        round(best_score, 4)
    )


    # --------------------------------------------------------
    # Display Elbow Graph
    # --------------------------------------------------------

    plot_elbow(
        k_values,
        inertia,
        dataset_name
    )


    # --------------------------------------------------------
    # Display Silhouette Graph
    # --------------------------------------------------------

    plot_silhouette(
        k_values,
        silhouette_scores,
        dataset_name
    )


    return (
        X_scaled,
        best_k,
        best_score
    )


# ============================================================
# 13. MAIN PROGRAM
# ============================================================

def main():

    for dataset_name, csvfile in DATASETS.items():

        # Get features for current dataset
        features = DATASET_FEATURES[
            dataset_name
        ]


        # Process dataset
        (
            X_scaled,
            best_k,
            best_score
        ) = process_dataset(
            dataset_name,
            csvfile,
            features
        )


# ============================================================
# 14. RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()