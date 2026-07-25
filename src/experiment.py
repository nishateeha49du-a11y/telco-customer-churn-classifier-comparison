import pandas as pd
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

from data_loader import load_dataset
from preprocessing import clean_dataset, prepare_features
from splitting import split_data
from models import get_models
from evaluation import evaluate_model


SEEDS = [42, 100, 2026]


df = load_dataset()
df = clean_dataset(df)

X, y = prepare_features(df)

results = []


for seed in SEEDS:

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X,
        y,
        random_state=seed
    )

    models = get_models(
        random_state=seed
    )

    for model_name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        results.append({
            "Model": model_name,
            "Seed": seed,
            **metrics
        })


results_df = pd.DataFrame(results)

output_path = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "metrics"
    / "all_seed_results.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    output_path,
    index=False
)

print("\nExperiment completed.")
print(f"Results saved to: {output_path}")