# 💳 Credit Card Fraud Detection Using Machine Learning

## 📌 Project Overview

This project develops an end-to-end machine learning system for detecting fraudulent credit card transactions.

The system addresses the highly imbalanced nature of fraud detection using SMOTE and compares multiple machine learning algorithms including Logistic Regression, Random Forest, and XGBoost.

XGBoost was selected as the final model based on its ROC-AUC performance. Threshold tuning was performed to improve fraud classification performance. SHAP was used for model explainability and MLflow was used for experiment tracking.

The final model is deployed through an interactive Streamlit web application.

---

## 🎯 Objectives

- Detect fraudulent credit card transactions.
- Handle severe class imbalance using SMOTE.
- Compare multiple machine learning models.
- Tune the classification threshold.
- Evaluate the model using Precision, Recall, F1-Score, ROC-AUC and PR-AUC.
- Explain predictions using SHAP.
- Track experiments using MLflow.
- Deploy the model using Streamlit.

---

## 📊 Dataset

The project uses the Credit Card Fraud Detection dataset containing anonymized transaction features.

### Dataset Information

- Total transactions: 284,807
- Features: 30
- Target: `Class`
- `Time`: Transaction time
- `V1`–`V28`: Anonymized features
- `Amount`: Transaction amount

### Target Classes

| Class | Meaning |
|------:|---------|
| 0 | Normal Transaction |
| 1 | Fraudulent Transaction |

The dataset is highly imbalanced because fraudulent transactions represent only a very small portion of all transactions.

---

## 🔄 Machine Learning Pipeline

```text
Credit Card Dataset
        ↓
Exploratory Data Analysis
        ↓
Duplicate Removal
        ↓
Train-Test Split
        ↓
Feature Scaling
        ↓
SMOTE
        ↓
Model Training
        ↓
Logistic Regression
Random Forest
XGBoost
        ↓
Model Evaluation
        ↓
Threshold Tuning
        ↓
Final XGBoost Model
        ↓
SHAP Explainability
        ↓
MLflow Tracking
        ↓
Streamlit Deployment