import pandas as pd
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent

results_path = (
    project_root
    / "results"
    / "metrics"
    / "all_seed_results.csv"
)

results_df = pd.read_csv(results_path)


metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC"
]


seed_summary = (
    results_df
    .groupby("Model")[metric_columns]
    .agg(["mean", "std", "min", "max"])
)


output_path = (
    project_root
    / "results"
    / "metrics"
    / "seed_comparison_summary.csv"
)

seed_summary.to_csv(output_path)

print("Seed comparison completed.")
print(f"Summary saved to: {output_path}")

print("\nSeed comparison summary:")
print(seed_summary)