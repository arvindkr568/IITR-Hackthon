"""
Phase 14: Unit Test Suite for Data Loader & Contract Validation
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pytest
import pandas as pd
from src.data_loader import load_datasets, validate_data_schema, DataContractError

def test_load_datasets_exist():
    df_train, df_test, sample_sub = load_datasets()
    assert isinstance(df_train, pd.DataFrame)
    assert isinstance(df_test, pd.DataFrame)
    assert len(df_train) == 10147
    assert len(df_test) == 2537

def test_validate_data_schema_valid():
    df_train, df_test, _ = load_datasets()
    assert validate_data_schema(df_train, is_train=True) is True
    assert validate_data_schema(df_test, is_train=False) is True

def test_validate_data_schema_invalid():
    invalid_df = pd.DataFrame({"dummy_column": [1, 2, 3]})
    with pytest.raises(DataContractError):
        validate_data_schema(invalid_df, is_train=True)
