"""
Phase 11: Prediction & Submission Pipeline Module
Loads serialized model pipeline, transforms test dataset, generates coupon acceptance 
predictions, and outputs valid submission files.
"""

import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    FINAL_MODEL_PATH, SUBMISSION_PATH, OUTPUTS_DIR, ID_COL, TARGET_COL
)
from src.data_loader import load_datasets
from src.feature_engineering import apply_feature_engineering

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_predictions(output_prob: bool = False) -> pd.DataFrame:
    """
    Generates predictions on test dataset using persisted model pipeline.
    
    Args:
        output_prob: If True, outputs predicted probabilities instead of binary class labels.
        
    Returns:
        DataFrame containing customer_id and predictions.
    """
    logger.info("Loading test dataset for inference...")
    _, df_test, sample_sub = load_datasets()

    logger.info(f"Loading final trained pipeline from {FINAL_MODEL_PATH}...")
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model pipeline artifact missing at {FINAL_MODEL_PATH}. Run training first.")

    pipeline = joblib.load(FINAL_MODEL_PATH)

    # Feature engineering on test set
    df_test_feat = apply_feature_engineering(df_test)
    X_test = df_test_feat.drop(columns=[ID_COL], errors="ignore")

    # Predict
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    submission = pd.DataFrame({
        ID_COL: df_test[ID_COL],
        TARGET_COL: probs if output_prob else preds
    })

    # Validation of submission contract
    if len(submission) != 2537:
        logger.warning(f"Submission row count discrepancy: expected 2537, got {len(submission)}")

    if submission[TARGET_COL].isnull().sum() > 0:
        raise ValueError("Submission file contains NaN prediction values!")

    # Save submission file to root and outputs folder
    submission.to_csv(SUBMISSION_PATH, index=False)
    submission.to_csv(OUTPUTS_DIR / "submission.csv", index=False)
    
    logger.info(f"Successfully generated test predictions ({len(submission)} rows).")
    logger.info(f"Saved submission file to {SUBMISSION_PATH}")
    logger.info(f"Class predictions breakdown:\n{pd.Series(preds).value_counts().to_string()}")

    return submission

if __name__ == "__main__":
    generate_predictions()
