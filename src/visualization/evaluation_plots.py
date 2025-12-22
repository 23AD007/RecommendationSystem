import mlflow
import pandas as pd
import matplotlib.pyplot as plt


def plot_model_comparison():
    experiments = mlflow.search_experiments()
    records = []

    for exp in experiments:
        runs = mlflow.search_runs(
            exp.experiment_id,
            order_by=["start_time DESC"],
            max_results=1
        )

        if not runs.empty:
            run = runs.iloc[0]
            records.append({
                "Model": exp.name.replace("Packaging_", ""),
                "Accuracy": run.get("metrics.accuracy"),
                "F1-score": run.get("metrics.f1_score"),
                "ROC-AUC": run.get("metrics.roc_auc"),
            })

    df = pd.DataFrame(records).dropna()

    df.set_index("Model").plot(kind="bar", figsize=(10, 6))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_model_comparison()
