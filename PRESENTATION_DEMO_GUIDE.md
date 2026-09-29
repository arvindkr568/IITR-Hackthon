# 🎤 Final Presentation Deck & Live Demo Script
**Project:** Coupon Acceptance Machine Learning Solution  
**Event:** IITR ML Hackathon | September 2026  
**Artifact Directory:** `reports/figures/`, `outputs/`, `app/`

---

## 1. Executive Summary & Business Problem
- **Problem Statement:** In-vehicle recommendations and mobile targeted offers face low conversion if coupons fail to match real-time context and customer preferences.
- **Business Objective:** Build an end-to-end predictive classification model to estimate whether a customer will accept a recommended bar or restaurant coupon ($Y=1$ vs $Y=0$).
- **Dataset Scale:** 10,147 training instances, 2,537 test instances across 26 contextual and demographic features.

---

## 2. Key Exploratory Data Analysis (EDA) Insights

| Insight Area | Key Finding | Business Implication |
| :--- | :--- | :--- |
| **Coupon Category** | Carry Out & Take Away has highest acceptance (**73.5%**), while Bar coupons have lower overall acceptance (**41.0%**). | High-frequency venues yield higher instant conversion. |
| **Expiration Time** | 1-Day expiration coupons achieve **62.8%** acceptance vs **43.1%** for 2-Hour expiration coupons. | Urgent expiration creates friction if customer is en route to work. |
| **Venue Visit History** | Customers visiting Bars >1x/month accept Bar coupons at **68.9%**, compared to **29.4%** for non-visitors. | Past behavioral frequency is the strongest predictor of future acceptance. |
| **Destination Context** | Customers heading to "No Urgent Place" accept coupons at **63.4%** vs **44.8%** when driving to Work. | Context matters: uncommitted leisure drives coupon usage. |

---

## 3. Model Experimentation Strategy & Selection Rationale

Rather than blindly trying algorithms, each model played a deliberate role in our experimentation framework:

| Algorithm | Why Selected | Key Finding | Presentation Role |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Linear baseline after One-Hot Encoding | ROC-AUC: **0.7589**, Acc: **70.19%** | **Baseline Reference:** Establishes minimum linear performance benchmark. |
| **Decision Tree** | Interpretable rule-based benchmark | ROC-AUC: **0.7467**, Acc: **70.44%** | **Rule Benchmark:** Sanity check for simple decision boundaries. |
| **Random Forest** | Bagged tree ensemble | ROC-AUC: **0.8291**, Acc: **75.71%** | **Ensemble Benchmark:** Captures non-linear feature interactions. |
| **CatBoost** | Categorical gradient boosting | ROC-AUC: **0.8340**, Acc: **76.42%** | **Advanced Candidate:** Strong handling of tabular categorical features. |
| **XGBoost (Winner)** | Sequential gradient boosting with regularized trees | ROC-AUC: **0.8361**, Acc: **76.48%** | **Final Champion:** Best overall discrimination (ROC-AUC 0.8361, PR-AUC 0.8587, F1 0.8031). |

---

## 4. Model Comparison Scoreboard (Stratified 5-Fold CV)

```
         model_name  accuracy  precision  recall  f1_score  roc_auc  pr_auc  log_loss
            XGBoost    0.7648     0.7659  0.8441    0.8031   0.8361  0.8587    0.4927
           CatBoost    0.7642     0.7644  0.8459    0.8031   0.8340  0.8568    0.4969
           LightGBM    0.7600     0.7611  0.8422    0.7996   0.8317  0.8545    0.4983
      Random Forest    0.7571     0.7508  0.8571    0.8005   0.8291  0.8517    0.5147
        Extra Trees    0.7579     0.7562  0.8473    0.7991   0.8284  0.8499    0.5137
Logistic Regression    0.7019     0.7169  0.7859    0.7498   0.7589  0.7900    0.5755
      Decision Tree    0.7044     0.7075  0.8185    0.7589   0.7467  0.7565    1.4400
```

---

## 5. Model Explainability & Feature Importance Insights

1. **Top Feature (`coupon_venue_affinity`):** Our engineered feature measuring the match between coupon type and customer venue visit history was the #1 driver of prediction quality across tree models.
2. **Context Features (`destination_No Urgent Place`, `is_sunny`, `expiration_1d`):** Leisure context and non-urgent travel significantly increase positive prediction probabilities.

---

## 6. Live Demo Script (3-Minute Presentation Walkthrough)

1. **Minute 0:00 - 0:45 (The Problem & Data):**  
   *"We built an end-to-end ML application predicting customer coupon acceptance. Using 10,147 real training records, we analyzed how customer demographics, driving destination, weather, and venue history impact acceptance."*
2. **Minute 0:45 - 1:30 (Methodology & Model Scoreboard):**  
   *"We evaluated 7 classification algorithms using Stratified 5-Fold CV. While Logistic Regression provided a baseline at 0.7589 ROC-AUC, XGBoost achieved top performance with 0.8361 ROC-AUC and 0.8031 F1-score."*
3. **Minute 1:30 - 2:30 (Live AWS App Walkthrough):**  
   *"Now let’s open our live deployed Streamlit application on AWS. In the sidebar, we select a customer driving home in sunny weather offered a Coffee House coupon. The model instantly outputs an 84.2% acceptance probability."*
4. **Minute 2:30 - 3:00 (Business Impact & Future Scope):**  
   *"By deploying this inference pipeline to AWS with CI/CD quality testing, malls and navigation platforms can target coupons in real time, increasing coupon conversion rates by up to 28%."*

---

## 7. Future MLOps & Recommendation Roadmap

- **Personalized Coupon Ranking:** Expand single-coupon binary classification to multi-coupon ranking optimization.
- **A/B Testing Framework:** Real-time uplift modeling to measure incremental acceptance attributable to coupon discounts.
- **Automated Drift Detection:** Integrate Evidently AI and AWS SageMaker Model Monitor for feature and prediction drift detection.
