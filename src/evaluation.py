import time

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test
):

    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    train_time = (
        time.perf_counter()
        - start_time
    )

    start_time = time.perf_counter()

    y_pred = model.predict(
        X_test
    )

    predict_time = (
        time.perf_counter()
        - start_time
    )

    y_prob = (
        model.predict_proba(X_test)[:, 1]
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    specificity = (
        tn / (tn + fp)
    )

    return {

        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_test,
            y_prob
        ),

        "Specificity": specificity,

        "Train Time": train_time,

        "Predict Time": predict_time
    }