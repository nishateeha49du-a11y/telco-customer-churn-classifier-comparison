from pathlib import Path

import pandas as pd


def create_final_comparison():

    project_root = Path(__file__).resolve().parent.parent

    input_path = (
        project_root
        / "results"
        / "metrics"
        / "all_seed_results.csv"
    )

    output_path = (
        project_root
        / "results"
        / "metrics"
        / "final_model_comparison.csv"
    )

    df = pd.read_csv(
        input_path
    )

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

    final_results = (
        df
        .groupby("Model")[metrics]
        .mean()
        .sort_values(
            by="F1",
            ascending=False
        )
    )

    final_results.to_csv(
        output_path
    )

    print(
        "\nFinal Model Comparison:"
    )

    print(
        final_results.round(4)
    )

    print(
        f"\nFinal comparison saved to: {output_path}"
    )


if __name__ == "__main__":

    create_final_comparison()