import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

st.title("Customer Churn Survival Analysis")

# Load models
models = joblib.load("churn_models.pkl")

cox_model = models["cox"]
weibull_model = models["weibull"]
gb_model = models["gb"]
feature_columns = models["features"]

st.write("Predict churn risk using survival analysis models.")

# =========================
# USER INPUT
# =========================

age = st.number_input("Age", min_value=18, max_value=100, value=30)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    value=12
)

usage = st.number_input(
    "Usage Frequency",
    min_value=0,
    value=10
)

support_calls = st.number_input(
    "Support Calls",
    min_value=0,
    value=1
)

payment_delay = st.number_input(
    "Payment Delay",
    min_value=0,
    value=0
)

subscription = st.selectbox(
    "Subscription Type",
    ["Basic", "Standard", "Premium"]
)

contract = st.selectbox(
    "Contract Length",
    ["Monthly", "Quarterly", "Annual"]
)

total_spend = st.number_input(
    "Total Spend",
    min_value=0.0,
    value=500.0
)

last_interaction = st.number_input(
    "Last Interaction (days ago)",
    min_value=0,
    value=10
)

# =========================
# PREDICTION
# =========================

if st.button("Predict"):

    # Raw input dataframe
    data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "Usage Frequency": usage,
        "Support Calls": support_calls,
        "Payment Delay": payment_delay,
        "Subscription Type": subscription,
        "Contract Length": contract,
        "Total Spend": total_spend,
        "Last Interaction": last_interaction
    }])

    # One-hot encode categorical features
    data_encoded = pd.get_dummies(data)

    # Align với feature khi train
    data_aligned = data_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    st.subheader("Prediction Results")

    # =================
    # Cox Model
    # =================
    cox_pred = cox_model.predict_partial_hazard(data_aligned)

    st.write(
        "Cox Model Risk Score:",
        float(cox_pred.values[0])
    )

    # =================
    # Weibull Model
    # =================
    weibull_pred = weibull_model.predict_median(data_aligned)

    st.write(
        "Weibull Predicted Median Survival Time:",
        round(float(weibull_pred.values[0]), 2),
        "months"
    )

    # =================
    # Gradient Boosting
    # =================

    # Drop Tenure vì XGBoost train không dùng Tenure
    gb_input = data_aligned.drop(columns=["Tenure"], errors="ignore")

    dtest = xgb.DMatrix(gb_input)

    gb_pred = gb_model.predict(dtest)

    st.write(
        "Gradient Boosting Risk Score:",
        float(gb_pred[0])
    )

    gb_model.feature_names