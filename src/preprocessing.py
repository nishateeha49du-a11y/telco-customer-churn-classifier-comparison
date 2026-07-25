import pandas as pd


def clean_dataset(df):
    df = df.drop("customerID", axis=1)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df = df.dropna()

    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return df


def prepare_features(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X = pd.get_dummies(
        X,
        drop_first=True,
        dtype=int
    )

    return X, y