"""DBSCAN analysis for the student habits dataset.

Run:
    python dbscan_student_analysis.py

The exam score is kept for interpretation only. It is not used to create
clusters, because DBSCAN is an unsupervised method.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


DATA_FILE = Path(__file__).with_name("student_habits_performance (1).csv")
MIN_SAMPLES = 8
EPS_PERCENTILE = 70


def load_and_prepare_data(
    path: Path,
) -> tuple[pd.DataFrame, pd.Series, StandardScaler, pd.DataFrame]:
    data = pd.read_csv(path)
    data = data.dropna().reset_index(drop=True)

    target_column = next(
        (column for column in ("exam_score", "Exam_Score") if column in data),
        None,
    )
    target = data[target_column] if target_column else pd.Series(np.nan, index=data.index)
    id_columns = [column for column in ("student_id", "Student_ID", "Name") if column in data]
    target_columns = [column for column in ("exam_score", "Exam_Score") if column in data]
    features = data.drop(columns=id_columns + target_columns)
    numeric_features = features.select_dtypes(include="number")
    return numeric_features, target, StandardScaler(), data


def choose_eps(matrix: np.ndarray, min_samples: int, percentile: float) -> float:
    """Choose a reproducible starting eps from the k-neighbor distances."""
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(matrix)
    distances, _ = neighbors.kneighbors(matrix)
    return float(np.percentile(distances[:, -1], percentile))


def choose_plot_columns(data: pd.DataFrame, numeric_features: pd.DataFrame) -> tuple[str, str]:
    preferred_pairs = [
        ("study_hours_per_day", "attendance_percentage"),
        ("Hours_Studied", "Attendance"),
        ("math score", "reading score"),
        ("Math_Score", "Science_Score"),
    ]
    for x_column, y_column in preferred_pairs:
        if x_column in data and y_column in data:
            return x_column, y_column
    return tuple(numeric_features.columns[:2])


def evaluate_dbscan(matrix: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    cluster_labels = labels[labels != -1]
    clustered_rows = int(cluster_labels.size)
    noise_rows = int(np.sum(labels == -1))
    cluster_count = len(set(cluster_labels))

    result: dict[str, float | int] = {
        "clusters": cluster_count,
        "clustered_rows": clustered_rows,
        "noise_rows": noise_rows,
    }
    if cluster_count >= 2 and clustered_rows > cluster_count:
        clustered_mask = labels != -1
        result["silhouette"] = float(
            silhouette_score(matrix[clustered_mask], labels[clustered_mask])
        )
        result["davies_bouldin"] = float(
            davies_bouldin_score(matrix[clustered_mask], labels[clustered_mask])
        )
    else:
        result["silhouette"] = float("nan")
        result["davies_bouldin"] = float("nan")
    return result


def run_analysis(
    eps: float,
    min_samples: int,
    data_file: Path = DATA_FILE,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, dict[str, float | int]]:
    features, exam_scores, preprocessor, raw_data = load_and_prepare_data(data_file)
    matrix = preprocessor.fit_transform(features)

    model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = model.fit_predict(matrix)
    metrics = evaluate_dbscan(matrix, labels)
    return raw_data, exam_scores, labels, metrics


def launch_dashboard(data_file: Path = DATA_FILE) -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    raw_data = pd.read_csv(data_file).dropna().reset_index(drop=True)
    excluded_columns = [column for column in ("student_id", "Student_ID", "Name", "exam_score", "Exam_Score") if column in raw_data]
    numeric_features = raw_data.drop(columns=excluded_columns).select_dtypes(include="number")
    if len(numeric_features.columns) < 2:
        raise ValueError("The CSV needs at least two numeric columns for DBSCAN and the graph.")
    x_column, y_column = choose_plot_columns(raw_data, numeric_features)
    matrix = StandardScaler().fit_transform(numeric_features)
    suggested_eps = choose_eps(matrix, MIN_SAMPLES, EPS_PERCENTILE)

    root = tk.Tk()
    root.title("Student Habits - DBSCAN Explorer")
    root.geometry("1100x720")
    root.minsize(850, 600)

    controls = ttk.Frame(root, padding=10)
    controls.pack(fill="x")
    ttk.Label(controls, text="eps").pack(side="left")
    eps_var = tk.StringVar(value=f"{suggested_eps:.3f}")
    ttk.Entry(controls, textvariable=eps_var, width=10).pack(side="left", padx=(5, 15))
    ttk.Label(controls, text="min_samples").pack(side="left")
    min_samples_var = tk.StringVar(value=str(MIN_SAMPLES))
    ttk.Entry(controls, textvariable=min_samples_var, width=10).pack(side="left", padx=(5, 15))
    status_var = tk.StringVar()
    ttk.Label(controls, textvariable=status_var).pack(side="left", padx=15)

    body = ttk.Frame(root, padding=(10, 0, 10, 10))
    body.pack(fill="both", expand=True)
    figure = Figure(figsize=(7, 5), dpi=100)
    axis = figure.add_subplot(111)
    canvas = FigureCanvasTkAgg(figure, master=body)
    canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

    side = ttk.Frame(body, padding=(15, 0, 0, 0))
    side.pack(side="right", fill="both", expand=False)
    ttk.Label(side, text="Cluster summary", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
    columns = ("cluster", "students", "mean")
    table = ttk.Treeview(side, columns=columns, show="headings", height=12)
    for column, heading in zip(columns, ("Cluster", "Students", "Mean exam score")):
        table.heading(column, text=heading)
        table.column(column, width=105, anchor="center")
    table.pack(fill="x", pady=(8, 15))
    ttk.Label(
        side,
        text="Exam score is shown for interpretation only.\nIt is not used to create clusters.",
        wraplength=300,
    ).pack(anchor="w")

    def update_plot() -> None:
        try:
            eps = float(eps_var.get())
            min_samples = int(min_samples_var.get())
            if eps <= 0 or min_samples < 2:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid parameters", "eps must be positive and min_samples must be at least 2.")
            return

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(matrix)
        metrics = evaluate_dbscan(matrix, labels)
        axis.clear()
        for cluster in sorted(set(labels)):
            mask = labels == cluster
            name = "Noise" if cluster == -1 else f"Cluster {cluster}"
            axis.scatter(
                raw_data.loc[mask, x_column],
                raw_data.loc[mask, y_column],
                s=24 if cluster != -1 else 18,
                alpha=0.7,
                label=name,
            )
        axis.set_title(f"DBSCAN clusters: {x_column} vs {y_column}")
        axis.set_xlabel(x_column)
        axis.set_ylabel(y_column)
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        canvas.draw()

        for item in table.get_children():
            table.delete(item)
        clustered = labels != -1
        if clustered.any():
            summary = (
                pd.DataFrame({"cluster": labels[clustered], "exam_score": raw_data.loc[clustered, "exam_score"] if "exam_score" in raw_data else np.nan})
                .groupby("cluster")
                .agg(students=("exam_score", "size"), mean=("exam_score", "mean"))
                .sort_values("mean", ascending=False)
            )
            for cluster, row in summary.iterrows():
                table.insert("", "end", values=(cluster, int(row["students"]), f"{row['mean']:.2f}"))
        status_var.set(
            f"{metrics['clusters']} clusters | {metrics['noise_rows']} noise | "
            f"silhouette: {metrics['silhouette']:.3f}"
        )

    ttk.Button(controls, text="Run DBSCAN", command=update_plot).pack(side="left")
    update_plot()
    root.mainloop()


def run_cli(data_file: Path) -> None:
    raw_data = pd.read_csv(data_file).dropna().reset_index(drop=True)
    excluded_columns = [column for column in ("student_id", "Student_ID", "Name", "exam_score", "Exam_Score") if column in raw_data]
    numeric_features = raw_data.drop(columns=excluded_columns).select_dtypes(include="number")
    matrix = StandardScaler().fit_transform(numeric_features)
    eps = choose_eps(matrix, MIN_SAMPLES, EPS_PERCENTILE)
    _, exam_scores, labels, metrics = run_analysis(eps, MIN_SAMPLES, data_file)
    print(f"Dataset: {data_file.name}")
    print(f"Rows used: {len(raw_data)}")
    print(
        f"DBSCAN parameters: eps={eps:.3f} (the {EPS_PERCENTILE}th percentile "
        f"of k-neighbor distances), min_samples={MIN_SAMPLES}"
    )
    print(f"Clusters found: {metrics['clusters']}")
    print(f"Noise points: {metrics['noise_rows']}")
    print(f"Silhouette score: {metrics['silhouette']:.3f}")
    print(f"Davies-Bouldin score: {metrics['davies_bouldin']:.3f}")

    clustered = labels != -1
    if clustered.any():
        cluster_means = (
            pd.DataFrame({"cluster": labels[clustered], "exam_score": exam_scores[clustered]})
            .groupby("cluster", as_index=True)
            .agg(students=("exam_score", "size"), mean_exam_score=("exam_score", "mean"))
            .sort_values("mean_exam_score", ascending=False)
        )
        print("\nCluster summary (exam_score was not used for clustering):")
        print(cluster_means.round(2).to_string())

    # Optional diagnostic only: compare clusters with a 60-point pass label.
    pass_labels = (exam_scores >= 60).astype(int)
    valid = labels != -1
    if valid.any() and len(set(labels[valid])) >= 2:
        print(
            "\nAdjusted Rand score against pass/fail (interpretation only): "
            f"{adjusted_rand_score(pass_labels[valid], labels[valid]):.3f}"
        )

    identifier = next(
        (column for column in ("student_id", "Student_ID", "Name") if column in raw_data),
        None,
    )
    output = pd.DataFrame(
        {identifier or "row_number": raw_data[identifier] if identifier else raw_data.index, "cluster": labels}
    )
    output_file = data_file.with_name(f"{data_file.stem}_dbscan_clusters.csv")
    output.to_csv(output_file, index=False)
    print(f"\nSaved cluster assignments to {output_file.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DBSCAN on a student CSV file.")
    parser.add_argument("--csv", type=Path, default=DATA_FILE, help="CSV file to analyze")
    parser.add_argument("--cli", action="store_true", help="Print results instead of opening Tkinter")
    args = parser.parse_args()
    if not args.csv.exists():
        parser.error(f"CSV file not found: {args.csv}")
    if args.cli:
        run_cli(args.csv)
    else:
        launch_dashboard(args.csv)


if __name__ == "__main__":
    main()