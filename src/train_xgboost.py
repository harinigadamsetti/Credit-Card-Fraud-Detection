import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# 1. Load dataset
df = pd.read_csv("data/creditcard.csv")


# 2. Remove duplicates
df = df.drop_duplicates()


# 3. Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]


# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 5. Scale Time and Amount
scaler = StandardScaler()

X_train[["Time", "Amount"]] = scaler.fit_transform(
    X_train[["Time", "Amount"]]
)

X_test[["Time", "Amount"]] = scaler.transform(
    X_test[["Time", "Amount"]]
)


# 6. Apply SMOTE only to training data
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("Training data after SMOTE:")
print(X_train_smote.shape)


# 7. Create XGBoost model
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)


# 8. Train model
print("\nTraining XGBoost...")

model.fit(
    X_train_smote,
    y_train_smote
)


# 9. Predictions using default threshold 0.5
y_pred = model.predict(X_test)


# 10. Fraud probabilities
y_probability = model.predict_proba(X_test)[:, 1]


# 11. Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 12. Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 13. ROC-AUC
roc_auc = roc_auc_score(y_test, y_probability)

print("\nROC-AUC Score:", roc_auc)


# 14. PR-AUC
pr_auc = average_precision_score(y_test, y_probability)

print("PR-AUC Score:", pr_auc)


# 15. Threshold Tuning
print("\n" + "=" * 60)
print("THRESHOLD TUNING")
print("=" * 60)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for threshold in thresholds:

    # Convert probability into class prediction
    y_threshold_pred = (
        y_probability >= threshold
    ).astype(int)

    print(f"\nThreshold: {threshold}")

    print(
        classification_report(
            y_test,
            y_threshold_pred,
            digits=2
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_threshold_pred
        )
    )