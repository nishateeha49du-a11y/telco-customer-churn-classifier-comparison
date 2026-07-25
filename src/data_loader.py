from pathlib import Path
import pandas as pd


def load_dataset():
    """
    Load the Telco Customer Churn dataset.
    """

    project_root = Path(__file__).resolve().parent.parent

    data_path = (
        project_root
        / "data"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    df = pd.read_csv(data_path)

    return df