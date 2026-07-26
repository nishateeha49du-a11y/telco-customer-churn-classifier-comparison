from pathlib import Path

import pandas as pd


def create_model_ranking():

    project_root = Path(__file__).resolve().parent.parent

    input_path = (
        project_root
        / "results"
        / "metrics"
        / "final_model_comparison.csv"
    )

    output_path = (
        project_root
        / "results"
        / "metrics"
        / "model_ranking.csv"
    )

    df = pd.read_csv(
        input_path
    )

    ranking = (
        df
        .sort_values(
            by="F1",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    ranking.insert(
        0,
        "Rank",
        range(
            1,
            len(ranking) + 1
        )
    )

    ranking.to_csv(
        output_path,
        index=False
    )

    print(
        "\nModel Ranking Based on F1-Score:"
    )

    print(
        ranking[
            [
                "Rank",
                "Model",
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
        ].round(4)
    )

    print(
        f"\nModel ranking saved to: {output_path}"
    )


if __name__ == "__main__":

    create_model_ranking()