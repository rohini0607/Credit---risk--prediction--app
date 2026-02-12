import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="German Credit Risk AI System",
    page_icon="🇩🇪",
    layout="wide"
)

# ---------------- LOAD MODEL (Optimized) ----------------
@st.cache_resource
def load_model():
    return pickle.load(open("german_credit_pipeline.pkl", "rb"))

model = load_model()

# ---------------- TITLE ----------------
st.title("🇩🇪 German Credit Risk Prediction System")
st.markdown("### AI-Powered Credit Risk Evaluation Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📊 Model Information")

st.sidebar.info("""
**Model:** Logistic Regression  
**Dataset:** German Credit Dataset  
**Target:** Good (0) / Bad (1)  
**Evaluation Metric:** ROC-AUC ≈ 0.80  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("👩‍💻 Developed by Rohini")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

# ================= LEFT COLUMN =================
with col1:
    st.subheader("📋 Applicant Financial Details")

    month_duration = st.number_input("Loan Duration (months)", 1, 100, 12)
    credit_amount = st.number_input("Credit Amount", 0, 50000, 5000)
    payment_to_income_ratio = st.number_input("Payment to Income Ratio", 0, 10, 2)
    residence_since = st.number_input("Residence Since (years)", 0, 10, 2)
    age = st.number_input("Age", 18, 100, 30)
    n_credits = st.number_input("Number of Existing Credits", 1, 10, 1)
    n_guarantors = st.number_input("Number of Guarantors", 0, 5, 0)

# ================= RIGHT COLUMN =================
with col2:
    st.subheader("🏠 Applicant Profile Details")

    status_account = st.selectbox("Status Account",
        ["no checking account", "< 0 DM", "0 to < 200 DM", ">= 200 DM"])

    credit_history = st.selectbox("Credit History",
        ["critical account/ other credits existing (not at this bank)",
         "existing credits paid back duly till now",
         "delay in paying off in the past",
         "no credits taken/ all credits paid back duly"])

    purpose = st.selectbox("Purpose",
        ["radio/television", "education", "car (new)", "car (used)",
         "business", "furniture/equipment", "repairs", "vacation/others"])

    status_savings = st.selectbox("Savings",
        ["unknown/ no savings account", "< 100 DM",
         "100 to < 500 DM", "500 to < 1000 DM", ">= 1000 DM"])

    years_employment = st.selectbox("Years Employment",
        ["unemployed", "< 1 year", "1 to < 4 years",
         "4 to < 7 years", ">= 7 years"])

    status_and_sex = st.selectbox("Status and Sex",
        ["male single", "male married/widowed",
         "female divorced/separated/married"])

    secondary_obligor = st.selectbox("Secondary Obligor",
        ["none", "co-applicant", "guarantor"])

    collateral = st.selectbox("Collateral",
        ["none", "car or other", "real estate",
         "savings agreement/life insurance"])

    other_installment_plans = st.selectbox("Other Installment Plans",
        ["none", "bank", "stores"])

    housing = st.selectbox("Housing",
        ["own", "rent", "for free"])

    job = st.selectbox("Job",
        ["unemployed/non-resident", "unskilled resident",
         "skilled employee", "management/self-employed"])

    telephone = st.selectbox("Telephone", ["none", "yes"])
    is_foreign_worker = st.selectbox("Foreign Worker", ["no", "yes"])

# ---------------- PREDICTION SECTION ----------------
st.markdown("---")

if st.button("🚀 Predict Credit Risk"):

    try:
        input_data = pd.DataFrame([{
            "month_duration": month_duration,
            "credit_amount": credit_amount,
            "payment_to_income_ratio": payment_to_income_ratio,
            "residence_since": residence_since,
            "age": age,
            "n_credits": n_credits,
            "n_guarantors": n_guarantors,
            "status_account": status_account,
            "credit_history": credit_history,
            "purpose": purpose,
            "status_savings": status_savings,
            "years_employment": years_employment,
            "status_and_sex": status_and_sex,
            "secondary_obligor": secondary_obligor,
            "collateral": collateral,
            "other_installment_plans": other_installment_plans,
            "housing": housing,
            "job": job,
            "telephone": telephone,
            "is_foreign_worker": is_foreign_worker
        }])

        prediction = model.predict(input_data)[0]
        prob_good, prob_bad = model.predict_proba(input_data)[0]

        st.subheader("📊 Risk Analysis Result")

        colA, colB = st.columns(2)

        with colA:
            st.metric("Probability of High Risk (Bad)", f"{prob_bad:.2%}")

        with colB:
            st.metric("Probability of Low Risk (Good)", f"{prob_good:.2%}")

        # Visual Risk Bar
        st.progress(int(prob_bad * 100))

        # Confidence Level
        confidence = abs(prob_good - prob_bad)

        if confidence > 0.60:
            confidence_level = "Very High"
        elif confidence > 0.40:
            confidence_level = "High"
        elif confidence > 0.20:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"

        st.info(f"🔍 Model Confidence Level: **{confidence_level}**")

        # Final Decision Block
        if prediction == 0:
            st.success("✅ Low Credit Risk (Good Applicant)")
            st.markdown("✔ Recommended for loan approval based on model evaluation.")
        else:
            st.error("⚠ High Credit Risk (Bad Applicant)")
            st.markdown("⚠ Caution advised before loan approval.")

        st.markdown("---")
        st.caption("⚡ AI Decision Support System for Credit Risk Assessment")

    except Exception as e:
        st.error("Prediction Error Occurred")
        st.exception(e)
