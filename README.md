# 🎟️ Coupon Acceptance Machine Learning Solution
**IITR ML Hackathon | September 2026**

An end-to-end, production-grade machine learning solution designed to predict whether a customer will accept a recommended restaurant or bar coupon ($Y=1$ vs $Y=0$). Includes modular Python code pipelines, feature engineering, 5-fold cross-validation model comparison, SHAP/feature importance explainability, interactive Streamlit web application, Docker containerization, AWS cloud deployment strategy, and automated GitHub CI/CD testing.

---

## 📁 Repository Structure

```
IITR-Hackthon/
├── data/
│   ├── raw/
│   │   ├── train.csv                      # Training dataset (10,147 rows)
│   │   ├── test.csv                       # Test dataset (2,537 rows)
│   │   └── sample_submission.csv
│   └── processed/
├── notebooks/
│   ├── 01_data_understanding.ipynb        # Phase 2: Schema validation notebook
│   ├── 02_eda.ipynb                       # Phase 4: Exploratory data analysis notebook
│   ├── 03_feature_engineering.ipynb       # Phase 6: Feature engineering notebook
│   ├── 04_baseline_models.ipynb           # Phase 7: Baseline Logistic Regression notebook
│   ├── 05_model_comparison.ipynb          # Phase 8: Model comparison scoreboard notebook
│   └── 06_final_model.ipynb               # Phase 10 & 11: Final pipeline & prediction notebook
├── src/
│   ├── config.py                          # Phase 1: Configuration, paths, constants
│   ├── data_loader.py                     # Phase 2: Ingestion & data contract validation
│   ├── data_quality.py                    # Phase 3: Automated data quality audit
│   ├── eda.py                             # Phase 4: EDA plot generation engine
│   ├── preprocessing.py                   # Phase 5: Sklearn pipeline preprocessor
│   ├── feature_engineering.py             # Phase 6: Leakage-free feature engineering
│   ├── baseline.py                        # Phase 7: Logistic Regression baseline model
│   ├── train.py                           # Phase 8 & 9: 5-Fold CV model comparison & tuning
│   ├── evaluate.py                        # Phase 8: Evaluation metrics framework
│   ├── explainability.py                  # Phase 10: Model explainability & feature importances
│   ├── predict.py                         # Phase 11: Inference engine & submission generator
│   └── train_final.py                     # Master orchestrator running all phases
├── models/
│   └── model_pipeline.joblib              # Winning serialized ML pipeline artifact
├── reports/
│   └── figures/                           # Publication-ready EDA & importance charts
├── app/
│   └── app.py                             # Phase 12: Streamlit web application
├── tests/                                 # Phase 14: Automated Pytest suite
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   └── test_predict.py
├── .github/workflows/
│   └── ci.yml                             # Phase 14: GitHub Actions CI/CD workflow
├── Dockerfile                             # Phase 13: Containerization setup
├── docker-compose.yml                     # Local container orchestration
├── DEPLOYMENT_PLAN.md                     # Phase 13: Detailed AWS cloud deployment plan
├── PRESENTATION_DEMO_GUIDE.md             # Phase 15: Slide deck guide & live demo script
├── requirements.txt                       # Project python dependencies
├── submission.csv                         # Final test predictions (2,537 rows)
└── README.md
```

---

## 🚀 Quick Start & Installation

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/arvindkr568/IITR-Hackthon.git
cd IITR-Hackthon

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run End-to-End Execution Pipeline
To run all phases automatically (Ingestion -> Data Quality -> EDA -> Baseline -> 5-Fold Model Comparison -> Explainability -> Test Prediction Generation):

```bash
PYTHONPATH=. python3 src/train_final.py
```

### 3. Run Automated Tests
```bash
PYTHONPATH=. pytest tests/ -v
```

### 4. Launch Interactive Streamlit Web Application
```bash
streamlit run app/app.py
```
Access the application at `http://localhost:8501`.

---

## 🏆 Model Scoreboard & Validation Results

Evaluated across **Stratified 5-Fold Cross-Validation** using identical train/validation splits:

| Model Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | LogLoss | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Winner)** | **0.7648** | **0.7659** | **0.8441** | **0.8031** | **0.8361** | **0.8587** | **0.4927** | 🏆 **Final Selected Model** |
| **CatBoost** | 0.7642 | 0.7644 | 0.8459 | 0.8031 | 0.8340 | 0.8568 | 0.4969 | Runner-Up |
| **LightGBM** | 0.7600 | 0.7611 | 0.8422 | 0.7996 | 0.8317 | 0.8545 | 0.4983 | Candidate |
| **Random Forest** | 0.7571 | 0.7508 | 0.8571 | 0.8005 | 0.8291 | 0.8517 | 0.5147 | Classical Benchmark |
| **Extra Trees** | 0.7579 | 0.7562 | 0.8473 | 0.7991 | 0.8284 | 0.8499 | 0.5137 | Classical Benchmark |
| **Logistic Regression** | 0.7019 | 0.7169 | 0.7859 | 0.7498 | 0.7589 | 0.7900 | 0.5755 | Linear Baseline |
| **Decision Tree** | 0.7044 | 0.7075 | 0.8185 | 0.7589 | 0.7467 | 0.7565 | 1.4400 | Rule Benchmark |

---

## ☁️ AWS Cloud Deployment Summary

The project is packaged for containerized cloud deployment on AWS:
- **AWS App Runner / ECS:** Automated deployment via `Dockerfile`.
- **AWS EC2:** Nginx reverse proxy configuration documented in [`DEPLOYMENT_PLAN.md`](file:///Users/arvindkumar/AI-Cource/Hackthon/IITR-Hackthon/DEPLOYMENT_PLAN.md).
- **Health Endpoint:** `/_stcore/health`.

---

## 🎤 Presentation & Demo Script

Refer to [`PRESENTATION_DEMO_GUIDE.md`](file:///Users/arvindkumar/AI-Cource/Hackthon/IITR-Hackthon/PRESENTATION_DEMO_GUIDE.md) for the 3-minute live presentation walkthrough, slide content, EDA findings, SHAP insights, and future MLOps scope.
