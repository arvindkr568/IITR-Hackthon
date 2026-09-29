"""
Phase 2: Dataset Ingestion & Data Contract Module
Handles data loading, validation of data schema against contract definitions,
and isolation of train/test sets to prevent leakage.
"""

import logging
import pandas as pd
from typing import Tuple, Dict, Any
from src.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, SAMPLE_SUBMISSION_PATH, 
    TARGET_COL, ID_COL, RAW_CATEGORICAL_COLS, RAW_NUMERICAL_COLS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataContractError(ValueError):
    """Exception raised when raw dataset violates the expected data contract."""
    pass

def validate_data_schema(df: pd.DataFrame, is_train: bool = True) -> bool:
    """
    Validates dataset schema against expected columns and types.
    """
    expected_cols = [ID_COL] + RAW_CATEGORICAL_COLS + RAW_NUMERICAL_COLS
    if is_train:
        expected_cols.append(TARGET_COL)

    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        raise DataContractError(f"Dataset missing required columns: {missing_cols}")

    if is_train:
        unique_targets = set(df[TARGET_COL].dropna().unique())
        if not unique_targets.issubset({0, 1}):
            raise DataContractError(f"Target column '{TARGET_COL}' contains invalid values: {unique_targets}")

    logger.info(f"Data Schema Validation Passed ({'Train' if is_train else 'Test'}). Shape: {df.shape}")
    return True

def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads training, test, and sample submission datasets from raw data directory.
    
    Returns:
        Tuple of (df_train, df_test, sample_submission)
    """
    logger.info(f"Loading datasets from {TRAIN_DATA_PATH.parent}...")
    
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Train data file not found at {TRAIN_DATA_PATH}")
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test data file not found at {TEST_DATA_PATH}")
        
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)
    
    sample_sub = None
    if SAMPLE_SUBMISSION_PATH.exists():
        sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    validate_data_schema(df_train, is_train=True)
    validate_data_schema(df_test, is_train=False)

    return df_train, df_test, sample_sub

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a structured summary dictionary for data inspection.
    """
    return {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": df.columns.tolist(),
        "missing_counts": df.isnull().sum().to_dict(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }

if __name__ == "__main__":
    train, test, sub = load_datasets()
    print("Train dataset summary:", get_data_summary(train)["num_rows"], "rows")
    print("Test dataset summary:", get_data_summary(test)["num_rows"], "rows")
