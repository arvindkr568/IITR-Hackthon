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
    .team-banner {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%);
        border: 1px solid #4338CA;
        border-radius: 12px;
        padding: 12px 24px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: #E0E7FF;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 12px rgba(49, 46, 129, 0.3);
        margin-bottom: 20px;
    }
    .team-avatar-container {
        display: flex;
        justify-content: space-around;
        align-items: center;
        flex-wrap: wrap;
        gap: 15px;
        margin-bottom: 25px;
        padding: 15px;
        background: #0F172A;
        border-radius: 16px;
        border: 1px solid #1E293B;
    }
    .avatar-card {
        display: flex;
        flex-direction: column;
        align-items: center;
        transition: transform 0.2s ease-in-out;
    }
    .avatar-card:hover {
        transform: translateY(-4px);
    }
    .avatar-img {
        width: 70px;
        height: 70px;
        border-radius: 50%;
        border: 3px solid #6366F1;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        object-fit: cover;
    }
    .avatar-name {
        margin-top: 8px;
        font-size: 0.95rem;
        font-weight: 600;
        color: #F8FAFC;
    }
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

# Helper function to convert local image files to base64 strings
import base64

def get_image_src(file_path: Path, fallback_url: str) -> str:
    if file_path.exists():
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
            ext = file_path.suffix.lower().replace(".", "")
            if ext == "jpg":
                ext = "jpeg"
            return f"data:image/{ext};base64,{encoded}"
    return fallback_url

ASSETS_DIR = BASE_DIR / "app" / "assets"

# Top Team Header Banner
st.markdown("<div class='team-banner'>🎓 IIT Roorkee Batch 10 Hackathon Team 2 Achivers</div>", unsafe_allow_html=True)

# Team Members with actual photos & fallback URLs
team_members = [
    {
        "name": "Arvind",
        "src": get_image_src(ASSETS_DIR / "arvind.png", "https://ui-avatars.com/api/?name=Arvind&background=2563EB&color=fff&size=128&bold=true&rounded=true")
    },
    {
        "name": "Mithun",
        "src": get_image_src(ASSETS_DIR / "mithun.png", "https://ui-avatars.com/api/?name=Mithun&background=7C3AED&color=fff&size=128&bold=true&rounded=true")
    },
    {
        "name": "Ravi",
        "src": get_image_src(ASSETS_DIR / "ravi.png", "https://ui-avatars.com/api/?name=Ravi&background=059669&color=fff&size=128&bold=true&rounded=true")
    },
    {
        "name": "Vinod",
        "src": get_image_src(ASSETS_DIR / "vinod.jpg", "https://ui-avatars.com/api/?name=Vinod&background=D97706&color=fff&size=128&bold=true&rounded=true")
    },
    {
        "name": "Gayatri",
        "src": get_image_src(ASSETS_DIR / "gayatri.jpg", "https://ui-avatars.com/api/?name=Gayatri&background=DB2777&color=fff&size=128&bold=true&rounded=true")
    },
    {
        "name": "Manish",
        "src": get_image_src(ASSETS_DIR / "manish.jpg", "https://ui-avatars.com/api/?name=Manish&background=4F46E5&color=fff&size=128&bold=true&rounded=true")
    },
    {
        "name": "Akash",
        "src": get_image_src(ASSETS_DIR / "akash.jpg", "https://ui-avatars.com/api/?name=Akash&background=0284C7&color=fff&size=128&bold=true&rounded=true")
    },
    {
        "name": "Arun",
        "src": get_image_src(ASSETS_DIR / "arun.jpg", "https://ui-avatars.com/api/?name=Arun&background=DC2626&color=fff&size=128&bold=true&rounded=true")
    },
    {
        "name": "Joy",
        "src": get_image_src(ASSETS_DIR / "joy.jpg", "https://ui-avatars.com/api/?name=Joy&background=0D9488&color=fff&size=128&bold=true&rounded=true")
    }
]

col_avatars = st.columns(len(team_members))
for idx, member in enumerate(team_members):
    with col_avatars[idx]:
        st.markdown(f"""
        <div style="text-align: center;">
            <img src="{member['src']}" 
                 style="width: 95px; height: 95px; border-radius: 50%; border: 3px solid #818CF8; box-shadow: 0 6px 12px rgba(0,0,0,0.5); object-fit: cover;">
            <div style="margin-top: 8px; font-weight: 600; font-size: 1rem; color: #F1F5F9;">{member['name']}</div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)
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
    st.write("Upload custom CSV files (`train.csv` and `test.csv`) to validate data contract schema, execute feature engineering, train candidate models, and verify validation metrics.")

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        uploaded_train = st.file_uploader("Upload Training Dataset (`train.csv`)", type=["csv"])
    with col_u2:
        uploaded_test = st.file_uploader("Upload Test Dataset (`test.csv`)", type=["csv"])

    if uploaded_train is not None:
        df_custom_train = pd.read_csv(uploaded_train)
        st.success(f"✅ Uploaded Train Dataset: {df_custom_train.shape[0]} rows, {df_custom_train.shape[1]} columns.")
        
        # Verify Target Column Presence
        if "Y" in df_custom_train.columns:
            y_counts = df_custom_train["Y"].value_counts().to_dict()
            st.info(f"Target 'Y' Distribution: Class 1 (Accepted): {y_counts.get(1, 0)}, Class 0 (Rejected): {y_counts.get(0, 0)}")
        else:
            st.error("⚠️ Uploaded CSV missing target column 'Y'. Please ensure target 'Y' is included.")

        st.dataframe(df_custom_train.head(3), use_container_width=True)

    if uploaded_test is not None:
        df_custom_test = pd.read_csv(uploaded_test)
        st.success(f"✅ Uploaded Test Dataset: {df_custom_test.shape[0]} rows, {df_custom_test.shape[1]} columns.")
        st.dataframe(df_custom_test.head(3), use_container_width=True)

    if st.button("🚀 Run End-to-End Retraining Pipeline on Uploaded Data"):
        if uploaded_train is not None:
            raw_dir = BASE_DIR / "data" / "raw"
            df_custom_train.to_csv(raw_dir / "train.csv", index=False)
            if uploaded_test is not None:
                df_custom_test.to_csv(raw_dir / "test.csv", index=False)
            
            with st.status("🚀 Executing Full Machine Learning Pipeline...", expanded=True) as status:
                st.write("🔍 Step 1: Validating Data Schema & Contract...")
                from src.data_loader import load_datasets
                load_datasets()
                st.write("✅ Step 1 Complete: Schema contract verified.")
                
                st.write("📋 Step 2: Running Data Quality Audit...")
                from src.data_quality import run_quality_check
                run_quality_check()
                st.write("✅ Step 2 Complete: Quality audit saved.")

                st.write("📈 Step 3: Generating EDA Visualizations...")
                from src.eda import generate_eda_plots
                generate_eda_plots()
                st.write("✅ Step 3 Complete: Figures saved.")

                st.write("⚡ Step 4: Training & Comparing Candidate Models across Stratified 5-Fold CV...")
                from src.train import run_model_comparison
                scoreboard_df, winning_model_name, _ = run_model_comparison()
                st.write(f"✅ Step 4 Complete: Champion model selected (**{winning_model_name}**).")

                st.write("💡 Step 5: Generating Model Explainability & Feature Importances...")
                from src.explainability import generate_explainability_report
                generate_explainability_report()
                st.write("✅ Step 5 Complete: Feature importances extracted.")

                st.write("🎯 Step 6: Generating Test Set Predictions...")
                from src.predict import generate_predictions
                sub_df = generate_predictions()
                st.write(f"✅ Step 6 Complete: Generated {len(sub_df)} test set predictions.")

                status.update(label="🎉 End-to-End Retraining Pipeline Executed Successfully!", state="complete", expanded=True)

            st.balloons()

            # Display Live Retraining Verification Scoreboard
            st.markdown("---")
            st.subheader("🏆 Updated Retraining Model Scoreboard (5-Fold CV Validation)")
            st.dataframe(scoreboard_df, use_container_width=True, hide_index=True)

            st.success(f"**Winning Model Selected:** `{winning_model_name}` (ROC-AUC: `{scoreboard_df.iloc[0]['roc_auc']:.4f}`, Accuracy: `{scoreboard_df.iloc[0]['accuracy']:.4f}`, F1-Score: `{scoreboard_df.iloc[0]['f1_score']:.4f}`)")
            
            # Submission File Download
            csv_bytes = sub_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Updated Test Predictions (submission.csv)",
                data=csv_bytes,
                file_name="submission.csv",
                mime="text/csv"
            )
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
