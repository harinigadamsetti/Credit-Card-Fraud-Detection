import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# 1. Load dataset
df = pd.read_csv("data/creditcard.csv")


# 2. Remove duplicates
df = df.drop_duplicates()


# 3. Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]


# 4. Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 5. Split training data into training and validation data
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)


print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Test data:", X_test.shape)


# 6. Scale Time and Amount
scaler = StandardScaler()

X_train[["Time", "Amount"]] = scaler.fit_transform(
    X_train[["Time", "Amount"]]
)

X_val[["Time", "Amount"]] = scaler.transform(
    X_val[["Time", "Amount"]]
)

X_test[["Time", "Amount"]] = scaler.transform(
    X_test[["Time", "Amount"]]
)


# 7. Apply SMOTE only to training data
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nTraining data after SMOTE:")
print(X_train_smote.shape)


# 8. Create XGBoost model
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


# 9. Train model
print("\nTraining XGBoost...")

model.fit(
    X_train_smote,
    y_train_smote
)


# 10. Get validation probabilities
y_val_probability = model.predict_proba(X_val)[:, 1]


# 11. Find best threshold using validation data
print("\n" + "=" * 60)
print("VALIDATION THRESHOLD TUNING")
print("=" * 60)

thresholds = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]

best_threshold = 0
best_f1 = 0

for threshold in thresholds:

    y_val_pred = (
        y_val_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.1f} | "
        f"Precision: {precision:.2f} | "
        f"Recall: {recall:.2f} | "
        f"F1: {f1:.2f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


print("\nBest Threshold:", best_threshold)
print("Best Validation F1:", best_f1)


# 12. Final evaluation on untouched test data
y_test_probability = model.predict_proba(X_test)[:, 1]

y_test_pred = (
    y_test_probability >= best_threshold
).astype(int)


print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)


# 13. Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_test_pred,
        digits=2
    )
)


# 14. Confusion matrix
print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_test_pred
    )
)