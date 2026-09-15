import os

import joblib
import mlflow
import mlflow.xgboost

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# 1. Set MLflow experiment
mlflow.set_experiment("Credit Card Fraud Detection")


# 2. Start MLflow run
with mlflow.start_run(run_name="XGBoost_Final_Model"):

    # Load the saved model
    model = joblib.load(
        "models/xgboost_fraud_model.pkl"
    )

    # Load the scaler
    scaler = joblib.load(
        "models/scaler.pkl"
    )

    # Load the selected threshold
    threshold = joblib.load(
        "models/threshold.pkl"
    )

    print("Loaded saved XGBoost model.")
    print("Threshold:", threshold)


    # ------------------------------------------------
    # Load test data
    # ------------------------------------------------

    import pandas as pd

    from sklearn.model_selection import train_test_split

    df = pd.read_csv(
        "data/creditcard.csv"
    )

    df = df.drop_duplicates()

    X = df.drop(
        "Class",
        axis=1
    )

    y = df["Class"]


    # Use the same test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # Scale test data
    X_test[["Time", "Amount"]] = scaler.transform(
        X_test[["Time", "Amount"]]
    )


    # ------------------------------------------------
    # Predictions
    # ------------------------------------------------

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    y_pred = (
        y_probability >= threshold
    ).astype(int)


    # ------------------------------------------------
    # Calculate metrics
    # ------------------------------------------------

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    pr_auc = average_precision_score(
        y_test,
        y_probability
    )


    # ------------------------------------------------
    # Log parameters
    # ------------------------------------------------

    mlflow.log_param(
        "model",
        "XGBoost"
    )

    mlflow.log_param(
        "n_estimators",
        200
    )

    mlflow.log_param(
        "max_depth",
        6
    )

    mlflow.log_param(
        "learning_rate",
        0.1
    )

    mlflow.log_param(
        "subsample",
        0.8
    )

    mlflow.log_param(
        "colsample_bytree",
        0.8
    )

    mlflow.log_param(
        "threshold",
        threshold
    )

    mlflow.log_param(
        "smote",
        True
    )


    # ------------------------------------------------
    # Log metrics
    # ------------------------------------------------

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )

    mlflow.log_metric(
        "pr_auc",
        pr_auc
    )


    # ------------------------------------------------
    # Log SHAP plot
    # ------------------------------------------------

    if os.path.exists(
        "models/shap_summary.png"
    ):
        mlflow.log_artifact(
            "models/shap_summary.png"
        )


    # ------------------------------------------------
    # Print results
    # ------------------------------------------------

    print("\n" + "=" * 60)
    print("MLFLOW RESULTS")
    print("=" * 60)

    print(
        "Precision:",
        precision
    )

    print(
        "Recall:",
        recall
    )

    print(
        "F1-Score:",
        f1
    )

    print(
        "ROC-AUC:",
        roc_auc
    )

    print(
        "PR-AUC:",
        pr_auc
    )

    print("\nMLflow tracking completed successfully.")