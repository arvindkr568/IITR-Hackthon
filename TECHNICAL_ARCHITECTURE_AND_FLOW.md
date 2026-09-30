# 📐 Technical Architecture & System Flow Document
**Project:** Coupon Acceptance Machine Learning Platform  
**Event:** IIT Roorkee Batch 10 Hackathon (Team 2)  
**File Reference:** `TECHNICAL_ARCHITECTURE_AND_FLOW.md`

---

## 1. Executive & Technical Overview

The objective of this machine learning system is to predict whether a customer will accept a recommended restaurant or bar coupon ($Y=1$ for accepted, $Y=0$ for rejected) based on customer demographics, driving journey context, time/weather variables, coupon properties, and historical venue visit frequencies.

### Key Data Specifications
- **Training Set (`data/raw/train.csv`):** 10,147 records, 27 features (including target $Y$).
- **Test Set (`data/raw/test.csv`):** 2,537 records, 26 features.
- **Target Variable ($Y$):** Binary classification ($56.84\%$ accepted vs $43.16\%$ rejected).
- **Validation Framework:** Stratified 5-Fold Cross-Validation with zero data leakage.

---

## 2. End-to-End System Architecture & Data Flow

```mermaid
flowchart TD
    A["Raw Datasets<br/>(train.csv, test.csv)"] --> B["src/data_loader.py<br/>Schema Contract Validation"]
    B --> C["src/data_quality.py<br/>Data Quality & Missingness Audit"]
    C --> D["src/feature_engineering.py<br/>Domain Feature Creation"]
    D --> E["src/preprocessing.py<br/>Sklearn ColumnTransformer Pipeline"]
    
    E --> F["src/train.py<br/>Stratified 5-Fold CV Model Comparison"]
    
    subgraph Candidate Models
        M1["Logistic Regression (Baseline)"]
        M2["Decision Tree"]
        M3["Random Forest"]
        M4["Extra Trees"]
        M5["XGBoost (Winner)"]
        M6["LightGBM"]
        M7["CatBoost"]
    end
    
    F --> Candidate Models
    Candidate Models --> G["src/evaluate.py<br/>Metrics Scoreboard Generator"]
    G --> H["Select Champion Model (XGBoost)<br/>ROC-AUC: 0.8361"]
    
    H --> I["Full Dataset Retraining & Serialization<br/>models/model_pipeline.joblib"]
    
    I --> J["src/predict.py<br/>Batch Inference Engine"]
    I --> K["app/app.py<br/>Streamlit Web Application"]
    
    J --> L["submission.csv<br/>(2,537 Test Predictions)"]
    K --> M["AWS Cloud Deployment<br/>(App Runner / EC2 / Docker)"]
```

---

## 3. Phase-by-Phase Technical Module Breakdown

### Module 1: Configuration & Environment Setup (`src/config.py`)
- **Role:** Central repository of paths, seeds (`SEED=42`), schema rules, and ordinal mapping dictionaries.
- **Key Mappings:** `AGE_MAP`, `INCOME_MAP`, `EDU_MAP`, `FREQ_MAP`, `EXPIRATION_HOURS_MAP`, `TIME_HOURS_MAP`.

### Module 2: Data Ingestion & Contract (`src/data_loader.py`)
- **Role:** Loads raw CSV files and enforces data contract rules via `validate_data_schema()`.
- **Contract Enforcement:** Throws `DataContractError` if required feature columns are missing or target values fall outside $\{0, 1\}$. Isolates train and test sets to guarantee zero leakage.

### Module 3: Data Quality Audit (`src/data_quality.py`)
- **Role:** Automated diagnostic audit checking missingness rates, duplicate customer IDs, class balance, and unexpected category variants.
- **Output:** Generates `outputs/data_quality_report.csv`.

### Module 4: Exploratory Data Analysis (`src/eda.py`)
- **Role:** Generates publication-ready visualizations saved to `reports/figures/`:
  - `target_distribution.png`
  - `acceptance_by_coupon_type.png`
  - `acceptance_by_expiration_time.png`
  - `acceptance_by_venue_frequency.png`
  - `acceptance_by_distance_direction.png`

### Module 5: Preprocessing Pipeline (`src/preprocessing.py`)
- **Role:** Builds Scikit-Learn `ColumnTransformer` and `Pipeline` objects.
- **Categorical Processing:** Constant missing value imputation (`"missing"`) $\rightarrow$ `OneHotEncoder(handle_unknown="ignore")`.
- **Numerical Processing:** Median imputation $\rightarrow$ Optional `StandardScaler`.

### Module 6: Leakage-Free Feature Engineering (`src/feature_engineering.py`)
- **Role:** Transforms raw columns into domain features without introducing future-information leakage:
  - `coupon_venue_affinity`: Calculates exact match score between offered coupon type and customer's visit frequency to that specific venue category (single most predictive feature).
  - `distance_score`: Composite indicator derived from distance flags.
  - `is_no_urgent`, `is_alone`, `is_sunny`: Binary contextual flags.
  - `total_venue_activity`: Cumulative activity score across all 5 venue types.

### Module 7: Baseline Model Benchmark (`src/baseline.py`)
- **Role:** Trains Logistic Regression using 5-fold Stratified CV to establish the minimum linear performance benchmark ($ROC\text{-}AUC = 0.7589$).

### Module 8 & 9: Model Training, Comparison & Serialization (`src/train.py`)
- **Role:** Trains 7 candidate algorithms across identical Stratified 5-Fold splits.
- **Model Selection Engine:** Ranks models on validation ROC-AUC and F1-Score. Retrains the winning pipeline on $100\%$ of training data and serializes the complete pipeline object to `models/model_pipeline.joblib`.

### Module 10: Model Explainability (`src/explainability.py`)
- **Role:** Extracts feature importances from the saved pipeline and generates `reports/figures/feature_importance.png`.

### Module 11: Batch Prediction Engine (`src/predict.py`)
- **Role:** Loads `models/model_pipeline.joblib`, runs feature engineering on `test.csv`, and exports formatted predictions to `submission.csv`.

### Module 12: Streamlit Interactive Web Application (`app/app.py`)
- **Role:** Production web app with 4 multi-tab interfaces:
  1. **Live Scenario Predictor:** Real-time probability calculation for custom customer contexts.
  2. **Upload Dataset & Retrain:** Upload custom `train.csv` / `test.csv` and trigger automated end-to-end pipeline execution.
  3. **Pipeline Steps & Audit:** Step-by-step visual breakdown of data quality, EDA charts, scoreboard, and download button for predictions.
  4. **AWS Deployment:** System architecture & container health status.

---

## 4. Model Comparison Scoreboard Results

All models were evaluated across identical **Stratified 5-Fold Cross-Validation** splits:

| Model Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | LogLoss | Role & Selection Rationale |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🏆 **XGBoost (Winner)** | **0.7648** | **0.7659** | **0.8441** | **0.8031** | **0.8361** | **0.8587** | **0.4927** | **Selected Champion:** Best overall discrimination. |
| **CatBoost** | 0.7642 | 0.7644 | 0.8459 | 0.8031 | 0.8340 | 0.8568 | 0.4969 | Advanced categorical boosting runner-up. |
| **LightGBM** | 0.7600 | 0.7611 | 0.8422 | 0.7996 | 0.8317 | 0.8545 | 0.4983 | Leaf-wise gradient boosting candidate. |
| **Random Forest** | 0.7571 | 0.7508 | 0.8571 | 0.8005 | 0.8291 | 0.8517 | 0.5147 | Classical bagged tree ensemble. |
| **Extra Trees** | 0.7579 | 0.7562 | 0.8473 | 0.7991 | 0.8284 | 0.8499 | 0.5137 | Extremely randomized tree ensemble. |
| **Logistic Regression** | 0.7019 | 0.7169 | 0.7859 | 0.7498 | 0.7589 | 0.7900 | 0.5755 | Baseline reference model. |
| **Decision Tree** | 0.7044 | 0.7075 | 0.8185 | 0.7589 | 0.7467 | 0.7565 | 1.4400 | Simple rule-based benchmark. |

---

## 5. Deployment & Containerization Specs

- **Containerization (`Dockerfile`):** Python 3.10-slim image with system dependencies (`libgomp1`) and Streamlit configuration exposed on port `8501`.
- **Health Check Endpoint:** `GET /_stcore/health` returns status `200 OK`.
- **Cloud Hosting Target:** AWS App Runner / AWS EC2 (`t3.medium`) with Nginx reverse proxy and GitHub Actions CI/CD (`.github/workflows/ci.yml`).
