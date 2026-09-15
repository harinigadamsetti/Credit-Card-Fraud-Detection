import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier


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


# 7. Create final XGBoost model
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


# 8. Train final model
print("\nTraining final XGBoost model...")

model.fit(
    X_train_smote,
    y_train_smote
)


# 9. Create models directory if it doesn't exist
import os

os.makedirs("models", exist_ok=True)


# 10. Save XGBoost model
joblib.dump(
    model,
    "models/xgboost_fraud_model.pkl"
)

print("\nXGBoost model saved successfully!")


# 11. Save scaler
joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Scaler saved successfully!")


# 12. Save selected threshold
threshold = 0.9

joblib.dump(
    threshold,
    "models/threshold.pkl"
)

print("Threshold saved successfully!")


# 13. Display saved files
print("\nSaved files:")

print("models/xgboost_fraud_model.pkl")
print("models/scaler.pkl")
print("models/threshold.pkl")