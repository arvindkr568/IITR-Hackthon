"""
Phase 7: Baseline Model Module
Trains Logistic Regression baseline using Stratified 5-Fold Cross-Validation.
Establishes baseline performance benchmark and directional coefficient interpretations.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import SEED, TARGET_COL, ID_COL, N_SPLITS
from src.data_loader import load_datasets
from src.feature_engineering import apply_feature_engineering
from src.preprocessing import build_preprocessing_pipeline
from src.evaluate import evaluate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_baseline_model() -> pd.DataFrame:
    """
    Trains and evaluates Logistic Regression baseline with Stratified 5-Fold CV.
    """
    df_train, _, _ = load_datasets()
    df_train_feat = apply_feature_engineering(df_train)

    X = df_train_feat.drop(columns=[ID_COL, TARGET_COL])
    y = df_train_feat[TARGET_COL].values

    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if X[c].dtype != 'object']

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    oof_preds = np.zeros(len(df_train_feat))
    oof_probs = np.zeros(len(df_train_feat))

    logger.info("Training Logistic Regression Baseline model across 5 Stratified Folds...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        preprocessor = build_preprocessing_pipeline(cat_cols, num_cols, scale_numeric=True)
        model = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
        
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        pipe.fit(X_tr, y_tr)

        oof_probs[val_idx] = pipe.predict_proba(X_va)[:, 1]
        oof_preds[val_idx] = (oof_probs[val_idx] >= 0.5).astype(int)

    metrics = evaluate_predictions(y, oof_preds, oof_probs)
    metrics["model_name"] = "Logistic Regression (Baseline)"
    
    logger.info("--- Baseline Logistic Regression Results ---")
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")

    return metrics

if __name__ == "__main__":
    train_baseline_model()
