"""
Phase 14: Unit Test Suite for Model Inference & Prediction Output
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pytest
import pandas as pd
from src.config import FINAL_MODEL_PATH
from src.predict import generate_predictions

def test_model_artifact_exists():
    assert FINAL_MODEL_PATH.exists(), "Model artifact pipeline file should exist."

def test_generate_predictions_schema():
    sub = generate_predictions()
    assert isinstance(sub, pd.DataFrame)
    assert len(sub) == 2537
    assert list(sub.columns) == ["customer_id", "Y"]
    assert set(sub["Y"].unique()).issubset({0, 1})
    assert sub["Y"].isnull().sum() == 0
