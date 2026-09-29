"""
Phase 8 & 9: Model Comparison & Tuning Module
Trains multiple candidate classification models across 5-fold Stratified Cross Validation.
Ranks models on validation metrics, selects the best model, and saves the final serialized pipeline.
"""

import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from src.config import (
    SEED, TARGET_COL, ID_COL, N_SPLITS, MODELS_DIR, OUTPUTS_DIR, METRICS_REPORT_PATH, FINAL_MODEL_PATH
)
from src.data_loader import load_datasets
from src.feature_engineering import apply_feature_engineering
from src.preprocessing import build_preprocessing_pipeline
from src.evaluate import evaluate_predictions, format_metrics_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_candidate_models() -> Dict[str, dict]:
    """
    Returns a dictionary of candidate models and their configurations.
    """
    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=SEED, C=1.0),
            "scale": True
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(max_depth=10, random_state=SEED, min_samples_leaf=5),
            "scale": False
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=300, max_depth=15, random_state=SEED, n_jobs=-1),
            "scale": False
        },
        "Extra Trees": {
            "model": ExtraTreesClassifier(n_estimators=300, max_depth=15, random_state=SEED, n_jobs=-1),
            "scale": False
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=SEED, eval_metric="logloss"),
            "scale": False
        },
        "LightGBM": {
            "model": lgb.LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1),
            "scale": False
        },
        "CatBoost": {
            "model": cb.CatBoostClassifier(iterations=500, depth=6, learning_rate=0.04, random_seed=SEED, verbose=0),
            "scale": False
        }
    }

def run_model_comparison() -> Tuple[pd.DataFrame, str, Pipeline]:
    """
    Performs 5-Fold Stratified Cross-Validation comparison across all candidate algorithms.
    Saves metrics report and serializes winning model pipeline.
    """
    df_train, _, _ = load_datasets()
    df_train_feat = apply_feature_engineering(df_train)

    X = df_train_feat.drop(columns=[ID_COL, TARGET_COL])
    y = df_train_feat[TARGET_COL].values

    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if X[c].dtype != 'object']

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    models_dict = get_candidate_models()
    all_metrics = []
    
    oof_predictions_dict = {}

    for name, config in models_dict.items():
        logger.info(f"--- Training {name} across {N_SPLITS} Folds ---")
        oof_probs = np.zeros(len(df_train_feat))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_va, y_va = X.iloc[val_idx], y[val_idx]

            preprocessor = build_preprocessing_pipeline(cat_cols, num_cols, scale_numeric=config["scale"])
            model_instance = config["model"]

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", model_instance)
            ])

            pipeline.fit(X_tr, y_tr)
            oof_probs[val_idx] = pipeline.predict_proba(X_va)[:, 1]

        oof_preds = (oof_probs >= 0.5).astype(int)
        oof_predictions_dict[name] = oof_probs
        
        metrics = evaluate_predictions(y, oof_preds, oof_probs)
        metrics["model_name"] = name
        all_metrics.append(metrics)
        logger.info(f"{name} Results -> ROC-AUC: {metrics['roc_auc']}, Accuracy: {metrics['accuracy']}, F1: {metrics['f1_score']}")

    # Scoreboard Creation
    scoreboard = format_metrics_table(all_metrics)
    scoreboard.to_csv(METRICS_REPORT_PATH, index=False)
    scoreboard.to_csv(OUTPUTS_DIR / "model_scoreboard.csv", index=False)
    logger.info(f"\n===== MODEL SCOREBOARD =====\n{scoreboard.to_string(index=False)}")

    # Select Winner
    best_model_name = scoreboard.iloc[0]["model_name"]
    logger.info(f"\n🏆 WINNING MODEL SELECTED: {best_model_name} (ROC-AUC: {scoreboard.iloc[0]['roc_auc']})")

    # Retrain Winning Pipeline on Full Dataset
    best_config = models_dict[best_model_name]
    final_preprocessor = build_preprocessing_pipeline(cat_cols, num_cols, scale_numeric=best_config["scale"])
    final_pipeline = Pipeline([
        ("preprocessor", final_preprocessor),
        ("classifier", best_config["model"])
    ])
    
    final_pipeline.fit(X, y)
    
    # Save Model Artifact
    joblib.dump(final_pipeline, FINAL_MODEL_PATH)
    logger.info(f"Saved complete serialized pipeline artifact to {FINAL_MODEL_PATH}")

    return scoreboard, best_model_name, final_pipeline

if __name__ == "__main__":
    run_model_comparison()
