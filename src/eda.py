import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# -----------------------------
# 1. Basic Dataset Information
# -----------------------------

print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# -----------------------------
# 2. Class Distribution
# -----------------------------

print("\nClass Distribution:")
print(df["Class"].value_counts())

print("\nClass Percentage:")
print(df["Class"].value_counts(normalize=True) * 100)

# -----------------------------
# 3. Transaction Amount
# -----------------------------

print("\nTransaction Amount Statistics:")
print(df["Amount"].describe())

# -----------------------------
# 4. Fraud Amount Statistics
# -----------------------------

print("\nFraud Transaction Amount:")
print(df[df["Class"] == 1]["Amount"].describe())

# -----------------------------
# 5. Normal Transaction Amount
# -----------------------------

print("\nNormal Transaction Amount:")
print(df[df["Class"] == 0]["Amount"].describe())

# -----------------------------
# 6. Visualization
# -----------------------------

plt.figure(figsize=(8, 5))

sns.countplot(x="Class", data=df)

plt.title("Normal vs Fraud Transactions")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")

plt.show()

# -----------------------------
# 7. Amount Distribution
# -----------------------------

plt.figure(figsize=(10, 5))

sns.boxplot(x="Class", y="Amount", data=df)

plt.title("Transaction Amount by Class")
plt.xlabel("Class")
plt.ylabel("Amount")

plt.show()