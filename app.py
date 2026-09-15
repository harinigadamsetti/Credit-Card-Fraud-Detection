import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# LOAD MODEL, SCALER AND THRESHOLD
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/xgboost_fraud_model.pkl"
    )

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    threshold = joblib.load(
        "models/threshold.pkl"
    )

    return model, scaler, threshold


model, scaler, threshold = load_model()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    data = pd.read_csv(
        "data/creditcard.csv"
    )

    return data


data = load_dataset()


# ============================================================
# TITLE
# ============================================================

st.title("💳 Credit Card Fraud Detection")

st.write(
    "An XGBoost-based machine learning system "
    "for detecting potentially fraudulent credit card transactions."
)

st.divider()


# ============================================================
# SIDEBAR - MODEL INFORMATION
# ============================================================

st.sidebar.title("🔍 Model Information")

st.sidebar.write("**Model:** XGBoost")
st.sidebar.write("**Sampling:** SMOTE")
st.sidebar.write(
    f"**Decision Threshold:** {threshold:.2f}"
)
st.sidebar.write("**ROC-AUC:** 0.9704")
st.sidebar.write("**PR-AUC:** 0.8137")
st.sidebar.write("**F1-Score:** 0.83")

st.sidebar.divider()

st.sidebar.info(
    "V1-V28 are anonymized features from "
    "the original credit card fraud dataset."
)


# ============================================================
# DEMO TRANSACTIONS
# ============================================================

st.subheader("🧪 Demo Transactions")

st.write(
    "Use the buttons below to load real transactions "
    "from the dataset for demonstration."
)


sample_col1, sample_col2 = st.columns(2)


# ------------------------------------------------------------
# Fraud Sample
# ------------------------------------------------------------

with sample_col1:

    if st.button(
        "🚨 Load Sample Fraud Transaction",
        use_container_width=True
    ):

        fraud_transaction = data[
            data["Class"] == 1
        ].iloc[0]

        st.session_state["sample_transaction"] = (
            fraud_transaction.to_dict()
        )

        st.success(
            "Fraud transaction loaded!"
        )


# ------------------------------------------------------------
# Normal Sample
# ------------------------------------------------------------

with sample_col2:

    if st.button(
        "✅ Load Sample Normal Transaction",
        use_container_width=True
    ):

        normal_transaction = data[
            data["Class"] == 0
        ].iloc[0]

        st.session_state["sample_transaction"] = (
            normal_transaction.to_dict()
        )

        st.success(
            "Normal transaction loaded!"
        )


# ============================================================
# CURRENT SAMPLE
# ============================================================

sample = st.session_state.get(
    "sample_transaction",
    {}
)


# ============================================================
# TRANSACTION DETAILS
# ============================================================

st.subheader("💰 Transaction Details")

col1, col2 = st.columns(2)


with col1:

    time = st.number_input(
        "Transaction Time",
        value=float(
            sample.get(
                "Time",
                0.0
            )
        )
    )


with col2:

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=float(
            sample.get(
                "Amount",
                100.0
            )
        )
    )


# ============================================================
# ANONYMIZED FEATURES
# ============================================================

st.subheader("🔢 Anonymized Features")

st.write(
    "V1-V28 are anonymized features used by the trained model."
)


features = {}

feature_columns = st.columns(4)


for i in range(1, 29):

    feature_name = f"V{i}"

    with feature_columns[(i - 1) % 4]:

        features[feature_name] = st.number_input(
            feature_name,
            value=float(
                sample.get(
                    feature_name,
                    0.0
                )
            ),
            format="%.6f"
        )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔍 Check Transaction",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [[
            time,

            *[
                features[f"V{i}"]
                for i in range(1, 29)
            ],

            amount
        ]],
        columns=[
            "Time",

            *[
                f"V{i}"
                for i in range(1, 29)
            ],

            "Amount"
        ]
    )


    # --------------------------------------------------------
    # Scale Time and Amount
    # --------------------------------------------------------

    input_scaled = input_data.copy()

    input_scaled[
        ["Time", "Amount"]
    ] = scaler.transform(
        input_scaled[
            ["Time", "Amount"]
        ]
    )


    # --------------------------------------------------------
    # Get Fraud Probability
    # --------------------------------------------------------

    fraud_probability = float(
        model.predict_proba(
            input_scaled
        )[0][1]
    )


    probability_percentage = (
        fraud_probability * 100
    )


    # --------------------------------------------------------
    # Apply Threshold
    # --------------------------------------------------------

    prediction = (
        fraud_probability >= float(threshold)
    )


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.subheader("📊 Prediction Result")


    if prediction:

        st.error(
            "🚨 FRAUDULENT TRANSACTION DETECTED"
        )

    else:

        st.success(
            "✅ NORMAL TRANSACTION"
        )


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Fraud Probability",
            f"{probability_percentage:.2f}%"
        )


    with result_col2:

        st.metric(
            "Decision Threshold",
            f"{float(threshold):.2f}"
        )


    # --------------------------------------------------------
    # Probability Bar
    # --------------------------------------------------------

    st.write("Fraud Probability")

    st.progress(
        float(
            min(
                fraud_probability,
                1.0
            )
        )
    )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.subheader("🧠 Model Explanation")

    st.write(
        "SHAP shows which features contributed most "
        "to this prediction."
    )


    try:

        # Create SHAP explainer
        explainer = shap.TreeExplainer(
            model
        )


        # Calculate SHAP values
        shap_values = explainer.shap_values(
            input_scaled
        )


        # ----------------------------------------------------
        # Handle SHAP output
        # ----------------------------------------------------

        if isinstance(
            shap_values,
            list
        ):

            values = shap_values[0][0]

        else:

            values = shap_values[0]


        # Convert values to normal numbers
        values = [
            float(value)
            for value in values
        ]


        # ----------------------------------------------------
        # Create Explanation DataFrame
        # ----------------------------------------------------

        explanation = pd.DataFrame({

            "Feature":
                input_scaled.columns,

            "SHAP Value":
                values

        })


        explanation[
            "Absolute SHAP"
        ] = explanation[
            "SHAP Value"
        ].abs()


        # Sort by importance
        explanation = explanation.sort_values(
            "Absolute SHAP",
            ascending=False
        )


        # Top 10 features
        top_features = explanation.head(10)


        # ----------------------------------------------------
        # Display Table
        # ----------------------------------------------------

        st.write(
            "Top 10 features influencing the prediction:"
        )


        st.dataframe(
            top_features[
                [
                    "Feature",
                    "SHAP Value"
                ]
            ],
            use_container_width=True
        )


        # ----------------------------------------------------
        # SHAP Bar Chart
        # ----------------------------------------------------

        chart_data = top_features.sort_values(
            "SHAP Value"
        )


        fig, ax = plt.subplots()


        ax.barh(
            chart_data["Feature"],
            chart_data["SHAP Value"]
        )


        ax.set_xlabel(
            "SHAP Value"
        )


        ax.set_ylabel(
            "Feature"
        )


        ax.set_title(
            "Top Feature Contributions"
        )


        plt.tight_layout()


        st.pyplot(
            fig
        )


        plt.close(
            fig
        )


    except Exception as e:

        st.warning(
            "SHAP explanation could not be generated."
        )

        st.write(
            f"Error details: {e}"
        )


    # ========================================================
    # VIEW TRANSACTION FEATURES
    # ========================================================

    with st.expander(
        "🔎 View Transaction Features"
    ):

        st.dataframe(
            input_data,
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Credit Card Fraud Detection | "
    "XGBoost + SMOTE + SHAP + MLflow"
)