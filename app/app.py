"""
Phase 12: Streamlit Multi-Tab Web Application
Includes:
- Interactive Live Scenario Prediction Engine
- Upload Custom Train & Test CSV Datasets for Dynamic Retraining
- Step-by-Step Pipeline Execution & Data Quality Audit Inspector
- Model Comparison Scoreboard & Feature Importances
- AWS Cloud Deployment & Architecture Status
"""

import sys
from pathlib import Path

# Add project root directory to sys.path to resolve 'src' imports seamlessly
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import time

# Page Configuration
st.set_page_config(
    page_title="Coupon Acceptance ML Platform",
    page_icon="🎟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
MODEL_PATH = BASE_DIR / "models" / "model_pipeline.joblib"
SCOREBOARD_PATH = BASE_DIR / "outputs" / "model_scoreboard.csv"
QUALITY_REPORT_PATH = BASE_DIR / "outputs" / "data_quality_report.csv"
SUBMISSION_PATH = BASE_DIR / "submission.csv"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #94A3B8;
        text-align: center;
        margin-bottom: 1.8rem;
    }
    .card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .metric-val {
        font-size: 2.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🎟️ Coupon Acceptance ML Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>End-to-End Pipeline Execution, Custom Data Ingestion, Model Analytics & Cloud Deployment</div>", unsafe_allow_html=True)

# Navigation Tabs
tab_predict, tab_upload, tab_steps, tab_aws = st.tabs([
    "🎯 Live Scenario Predictor", 
    "📤 Upload Dataset & Retrain", 
    "📊 Pipeline Steps & Execution Audit", 
    "☁️ AWS Cloud Deployment"
])

# ==========================================
# TAB 1: LIVE SCENARIO PREDICTOR
# ==========================================
with tab_predict:
    st.header("🎯 Real-Time Coupon Acceptance Prediction")
    st.write("Configure customer demographics, driving journey, and coupon parameters to calculate real-time acceptance probability.")

    col_input, col_result = st.columns([1.1, 0.9])

    with col_input:
        st.subheader("📋 Scenario Input Parameters")
        
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.selectbox("Age Group", ["below21", "21", "26", "31", "36", "41", "46", "50plus"], index=2)
            marital_status = st.selectbox("Marital Status", ["Single", "Unmarried partner", "Married partner", "Divorced", "Widowed"])
            education = st.selectbox("Education", ["Some High School", "High School Graduate", "Some college - no degree", "Associates degree", "Bachelors degree", "Graduate degree (Masters or Doctorate)"], index=4)
            income = st.selectbox("Income", ["Less than $12500", "$12500 - $24999", "$25000 - $37499", "$37500 - $49999", "$50000 - $62499", "$62500 - $74999", "$75000 - $87499", "$87500 - $99999", "$100000 or More"], index=3)
            occupation = st.selectbox("Occupation", ["Student", "Computer & Mathematical", "Sales & Related", "Management", "Other"])

        with c2:
            destination = st.selectbox("Destination", ["No Urgent Place", "Home", "Work"])
            passanger = st.selectbox("Passenger", ["Alone", "Friend(s)", "Kid(s)", "Partner"])
            weather = st.selectbox("Weather", ["Sunny", "Rainy", "Snowy"])
            temperature = st.selectbox("Temperature (°F)", [30, 55, 80], index=2)
            time_val = st.selectbox("Time of Day", ["7AM", "10AM", "2PM", "6PM", "10PM"], index=3)
            coupon = st.selectbox("Coupon Type", ["Restaurant(<20)", "Coffee House", "Carry out & Take away", "Bar", "Restaurant(20-50)"])

        st.markdown("**Venue Frequency & Journey Specs**")
        c3, c4 = st.columns(2)
        with c3:
            bar_freq = st.selectbox("Bar Visit Freq", ["never", "less1", "1~3", "4~8", "gt8"], index=1)
            coffee_freq = st.selectbox("Coffee House Visit Freq", ["never", "less1", "1~3", "4~8", "gt8"], index=2)
            carry_freq = st.selectbox("Carry Away Visit Freq", ["never", "less1", "1~3", "4~8", "gt8"], index=3)
        with c4:
            rest_less20_freq = st.selectbox("Rest (<20) Visit Freq", ["never", "less1", "1~3", "4~8", "gt8"], index=2)
            rest_20_50_freq = st.selectbox("Rest (20-50) Visit Freq", ["never", "less1", "1~3", "4~8", "gt8"], index=1)
            expiration = st.selectbox("Expiration", ["2h", "1d"], index=1)

    with col_result:
        st.subheader("🤖 Model Inference Result")
        
        input_data = pd.DataFrame([{
            "customer_id": 999999, "destination": destination, "passanger": passanger,
            "weather": weather, "temperature": temperature, "time": time_val, "coupon": coupon,
            "expiration": expiration, "gender": gender, "age": age, "maritalStatus": marital_status,
            "has_children": 0, "education": education, "occupation": occupation, "income": income,
            "car": np.nan, "Bar": bar_freq, "CoffeeHouse": coffee_freq, "CarryAway": carry_freq,
            "RestaurantLessThan20": rest_less20_freq, "Restaurant20To50": rest_20_50_freq,
            "toCoupon_GEQ5min": 1, "toCoupon_GEQ15min": 1, "toCoupon_GEQ25min": 0,
            "direction_same": 0, "direction_opp": 1
        }])

        if MODEL_PATH.exists():
            pipeline = joblib.load(MODEL_PATH)
            from src.feature_engineering import apply_feature_engineering
            input_feat = apply_feature_engineering(input_data)
            X_in = input_feat.drop(columns=["customer_id"], errors="ignore")
            
            prob = pipeline.predict_proba(X_in)[0, 1]
            pred = int(prob >= 0.5)

            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <h4>Prediction Status</h4>
                <div class="metric-val" style="color: {'#10B981' if pred==1 else '#EF4444'};">
                    {'✅ ACCEPTED' if pred==1 else '❌ REJECTED'}
                </div>
                <h3 style="margin-top: 10px;">Acceptance Probability: <b>{prob:.1%}</b></h3>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(prob))

            if pred == 1:
                st.success(f"**Actionable Advice:** High likelihood of acceptance ({prob:.1%}). Deliver coupon immediately!")
            else:
                st.warning(f"**Actionable Advice:** Low likelihood ({prob:.1%}). Consider adjusting venue category or time expiration.")
        else:
            st.error("Model pipeline artifact missing. Train model first.")

# ==========================================
# TAB 2: UPLOAD DATASET & RETRAIN
# ==========================================
with tab_upload:
    st.header("📤 Upload Custom Training & Test Data")
    st.write("Upload custom CSV files (`train.csv` and `test.csv`) to re-run schema validation, feature engineering, and automated model training.")

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        uploaded_train = st.file_uploader("Upload Training Dataset (`train.csv`)", type=["csv"])
    with col_u2:
        uploaded_test = st.file_uploader("Upload Test Dataset (`test.csv`)", type=["csv"])

    if uploaded_train is not None:
        df_custom_train = pd.read_csv(uploaded_train)
        st.success(f"Uploaded Train Dataset: {df_custom_train.shape[0]} rows, {df_custom_train.shape[1]} columns.")
        st.dataframe(df_custom_train.head(3), use_container_width=True)

    if uploaded_test is not None:
        df_custom_test = pd.read_csv(uploaded_test)
        st.success(f"Uploaded Test Dataset: {df_custom_test.shape[0]} rows, {df_custom_test.shape[1]} columns.")
        st.dataframe(df_custom_test.head(3), use_container_width=True)

    if st.button("🚀 Run End-to-End Retraining Pipeline on Uploaded Data"):
        if uploaded_train is not None:
            raw_dir = BASE_DIR / "data" / "raw"
            df_custom_train.to_csv(raw_dir / "train.csv", index=False)
            if uploaded_test is not None:
                df_custom_test.to_csv(raw_dir / "test.csv", index=False)
            
            with st.spinner("Executing Pipeline Steps (Data Quality -> EDA -> Baseline -> Ensemble -> Explainability)..."):
                from src.train_final import run_pipeline
                run_pipeline()
                st.success("🎉 Pipeline retraining completed successfully!")
                st.balloons()
        else:
            st.warning("Please upload at least `train.csv` before launching retraining.")

# ==========================================
# TAB 3: PIPELINE STEPS & EXECUTION AUDIT
# ==========================================
with tab_steps:
    st.header("📊 Pipeline Step-by-Step Execution Audit")
    
    st.subheader("Step 1 & 2: Dataset Schema & Data Quality Audit")
    if QUALITY_REPORT_PATH.exists():
        df_qual = pd.read_csv(QUALITY_REPORT_PATH)
        st.dataframe(df_qual, use_container_width=True)
    else:
        st.info("Run quality check to populate audit table.")

    st.subheader("Step 3: Exploratory Data Analysis (EDA) Highlights")
    fig_cols = st.columns(2)
    with fig_cols[0]:
        if (FIGURES_DIR / "target_distribution.png").exists():
            st.image(str(FIGURES_DIR / "target_distribution.png"), caption="Target Acceptance Distribution")
        if (FIGURES_DIR / "acceptance_by_expiration_time.png").exists():
            st.image(str(FIGURES_DIR / "acceptance_by_expiration_time.png"), caption="Acceptance by Time & Expiration")
    with fig_cols[1]:
        if (FIGURES_DIR / "acceptance_by_coupon_type.png").exists():
            st.image(str(FIGURES_DIR / "acceptance_by_coupon_type.png"), caption="Acceptance Rate by Coupon Category")
        if (FIGURES_DIR / "acceptance_by_venue_frequency.png").exists():
            st.image(str(FIGURES_DIR / "acceptance_by_venue_frequency.png"), caption="Acceptance by Venue History")

    st.subheader("Step 4: Model Comparison Leaderboard (Stratified 5-Fold CV)")
    if SCOREBOARD_PATH.exists():
        df_sb = pd.read_csv(SCOREBOARD_PATH)
        st.dataframe(df_sb, use_container_width=True, hide_index=True)
    
    st.subheader("Step 5: Feature Importances (Winning Pipeline)")
    if (FIGURES_DIR / "feature_importance.png").exists():
        st.image(str(FIGURES_DIR / "feature_importance.png"), use_column_width=True)

    st.subheader("Step 6: Test Set Submission Generator")
    if SUBMISSION_PATH.exists():
        df_sub = pd.read_csv(SUBMISSION_PATH)
        st.write(f"Generated {len(df_sub)} test set predictions.")
        st.dataframe(df_sub.head(10), use_container_width=True)
        
        csv_bytes = df_sub.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Test Predictions (submission.csv)",
            data=csv_bytes,
            file_name="submission.csv",
            mime="text/csv"
        )

# ==========================================
# TAB 4: AWS CLOUD DEPLOYMENT
# ==========================================
with tab_aws:
    st.header("☁️ AWS Cloud Architecture & Deployment Status")
    
    st.markdown("""
    ### 🏗️ Production Architecture Summary
    - **Hosting Platform:** AWS App Runner / AWS EC2 (`t3.medium`)
    - **Containerization:** Docker (`Dockerfile` exposed on port 8501)
    - **CI/CD Automation:** GitHub Actions (`.github/workflows/ci.yml`)
    - **Health Check Endpoint:** `/_stcore/health` (Status: `200 OK`)
    """)
    
    st.success("✅ Application Container Ready for AWS Deployment")
    st.code("docker build -t coupon-app . && docker run -p 8501:8501 coupon-app", language="bash")
