import pandas as pd


# Results from our trained models

results = {
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],

    "Precision": [
        0.05,
        0.91,
        0.92
    ],

    "Recall": [
        0.87,
        0.76,
        0.75
    ],

    "F1-Score": [
        0.10,
        0.83,
        0.83
    ],

    "ROC-AUC": [
        0.9619,
        0.9662,
        0.9704
    ],

    "PR-AUC": [
        None,
        None,
        0.8137
    ]
}


# Create DataFrame
comparison = pd.DataFrame(results)


# Display comparison
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(comparison.to_string(index=False))


# Find best model based on F1-score
best_f1_model = comparison.loc[
    comparison["F1-Score"].idxmax()
]

print("\n" + "=" * 70)
print("BEST MODEL BASED ON F1-SCORE")
print("=" * 70)

print(
    "Model:",
    best_f1_model["Model"]
)

print(
    "F1-Score:",
    best_f1_model["F1-Score"]
)


# Find best model based on ROC-AUC
best_roc_model = comparison.loc[
    comparison["ROC-AUC"].idxmax()
]

print("\n" + "=" * 70)
print("BEST MODEL BASED ON ROC-AUC")
print("=" * 70)

print(
    "Model:",
    best_roc_model["Model"]
)

print(
    "ROC-AUC:",
    best_roc_model["ROC-AUC"]
)