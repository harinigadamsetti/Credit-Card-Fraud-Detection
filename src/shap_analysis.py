import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


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


# 5. Load saved model and scaler
model = joblib.load(
    "models/xgboost_fraud_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)


# 6. Scale Time and Amount
X_test[["Time", "Amount"]] = scaler.transform(
    X_test[["Time", "Amount"]]
)


# 7. Select a sample for SHAP analysis
# Using a sample makes SHAP faster
X_sample = X_test.sample(
    n=2000,
    random_state=42
)


# 8. Create SHAP Tree Explainer
print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(model)


# 9. Calculate SHAP values
print("Calculating SHAP values...")

shap_values = explainer.shap_values(X_sample)


# 10. SHAP Summary Plot
print("\nGenerating SHAP summary plot...")

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.tight_layout()

plt.savefig(
    "models/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nSHAP analysis completed successfully!")

print(
    "SHAP plot saved to: models/shap_summary.png"
)