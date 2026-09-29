"""
Phase 3: Data Quality Assessment Module
Performs automated diagnostics on raw dataset: missingness patterns, duplicates,
anomalies, cardinalities, and target imbalance analysis.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import TARGET_COL, ID_COL, OUTPUTS_DIR
from src.data_loader import load_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def assess_data_quality(df: pd.DataFrame, dataset_name: str = "Train") -> pd.DataFrame:
    """
    Performs comprehensive data quality audit and returns a summary DataFrame.
    """
    logger.info(f"--- Data Quality Audit: {dataset_name} ---")
    
    total_rows = len(df)
    report_data = []

    for col in df.columns:
        n_missing = df[col].isnull().sum()
        pct_missing = (n_missing / total_rows) * 100
        n_unique = df[col].nunique(dropna=True)
        dtype = df[col].dtype
        sample_vals = df[col].dropna().unique()[:3].tolist()

        report_data.append({
            "dataset": dataset_name,
            "feature": col,
            "dtype": str(dtype),
            "missing_count": n_missing,
            "missing_pct": round(pct_missing, 2),
            "unique_count": n_unique,
            "sample_values": str(sample_vals)
        })

    report_df = pd.DataFrame(report_data)
    
    # Audit duplicates
    dup_ids = df[ID_COL].duplicated().sum() if ID_COL in df.columns else 0
    dup_rows = df.duplicated().sum()
    logger.info(f"{dataset_name} - Duplicate Customer IDs: {dup_ids}, Duplicate Full Rows: {dup_rows}")

    # Audit target class imbalance
    if TARGET_COL in df.columns:
        counts = df[TARGET_COL].value_counts()
        pcts = df[TARGET_COL].value_counts(normalize=True) * 100
        logger.info(f"Target Distribution:\n0 (Rejected): {counts.get(0, 0)} ({pcts.get(0, 0):.2f}%)\n1 (Accepted): {counts.get(1, 0)} ({pcts.get(1, 0):.2f}%)")

    return report_df

def run_quality_check() -> pd.DataFrame:
    df_train, df_test, _ = load_datasets()
    train_report = assess_data_quality(df_train, "Train")
    test_report = assess_data_quality(df_test, "Test")
    
    combined_report = pd.concat([train_report, test_report], ignore_index=True)
    report_path = OUTPUTS_DIR / "data_quality_report.csv"
    combined_report.to_csv(report_path, index=False)
    logger.info(f"Saved complete data quality report to {report_path}")
    return combined_report

if __name__ == "__main__":
    run_quality_check()
