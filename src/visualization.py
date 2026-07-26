from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def generate_visualizations():

    project_root = Path(__file__).resolve().parent.parent

    metrics_path = (
        project_root
        / "results"
        / "metrics"
        / "all_seed_results.csv"
    )

    figures_path = (
        project_root
        / "results"
        / "figures"
    )

    figures_path.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(metrics_path)

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "Specificity",
        "CV Mean",
        "CV Std",
        "Train Time",
        "Predict Time"
    ]

    mean_results = (
        df
        .groupby("Model")[metrics]
        .mean()
    )

    # ==========================================================
    # 1. Individual Metric Comparisons
    # ==========================================================

    for metric in metrics:

        plt.figure(
            figsize=(10, 6)
        )

        mean_results[metric].sort_values().plot(
            kind="bar"
        )

        plt.title(
            f"{metric} Comparison Across Models"
        )

        plt.xlabel(
            "Model"
        )

        plt.ylabel(
            metric
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        filename = (
            metric
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
            + "_comparison.png"
        )

        plt.savefig(
            figures_path / filename,
            dpi=300
        )

        plt.show()

        plt.close()

    # ==========================================================
    # 2. Overall Model Performance Comparison
    # ==========================================================

    performance_metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "Specificity"
    ]

    overall_results = (
        mean_results[performance_metrics]
        .sort_values(
            by="F1"
        )
    )

    overall_results.plot(
        kind="bar",
        figsize=(14, 8)
    )

    plt.title(
        "Overall Model Performance Comparison"
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Mean Score"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.legend(
        title="Metrics"
    )

    plt.tight_layout()

    plt.savefig(
        figures_path
        / "model_performance_comparison.png",
        dpi=300
    )

    plt.show()

    plt.close()

    # ==========================================================
    # 3. Cross-Validation Comparison
    # ==========================================================

    cv_results = (
        mean_results[
            [
                "CV Mean",
                "CV Std"
            ]
        ]
        .sort_values(
            by="CV Mean"
        )
    )

    cv_results.plot(
        kind="bar",
        figsize=(12, 7)
    )

    plt.title(
        "Cross-Validation Performance and Stability"
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Cross-Validation Score"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.legend(
        [
            "CV Mean",
            "CV Standard Deviation"
        ]
    )

    plt.tight_layout()

    plt.savefig(
        figures_path
        / "cross_validation_comparison.png",
        dpi=300
    )

    plt.show()

    plt.close()

    # ==========================================================
    # 4. Random Seed Stability
    # ==========================================================

    stability = (
        df
        .groupby("Model")["Accuracy"]
        .std()
        .sort_values()
    )

    plt.figure(
        figsize=(10, 6)
    )

    stability.plot(
        kind="bar"
    )

    plt.title(
        "Accuracy Stability Across Random Seeds"
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Accuracy Standard Deviation"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        figures_path
        / "seed_stability.png",
        dpi=300
    )

    plt.show()

    plt.close()

    # ==========================================================
    # 5. Training Time Comparison
    # ==========================================================

    train_time = (
        mean_results[
            "Train Time"
        ]
        .sort_values()
    )

    plt.figure(
        figsize=(10, 6)
    )

    train_time.plot(
        kind="bar"
    )

    plt.title(
        "Training Time Comparison Across Models"
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Training Time (seconds)"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        figures_path
        / "training_time_comparison.png",
        dpi=300
    )

    plt.show()

    plt.close()

    # ==========================================================
    # 6. Prediction Time Comparison
    # ==========================================================

    predict_time = (
        mean_results[
            "Predict Time"
        ]
        .sort_values()
    )

    plt.figure(
        figsize=(10, 6)
    )

    predict_time.plot(
        kind="bar"
    )

    plt.title(
        "Prediction Time Comparison Across Models"
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Prediction Time (seconds)"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        figures_path
        / "prediction_time_comparison.png",
        dpi=300
    )

    plt.show()

    plt.close()

    print(
        "All visualizations generated successfully."
    )


if __name__ == "__main__":

    generate_visualizations()