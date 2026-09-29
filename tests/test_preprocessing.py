"""
Phase 14: Unit Test Suite for Feature Engineering & Preprocessing
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pytest
import pandas as pd
import numpy as np
from src.data_loader import load_datasets
from src.feature_engineering import apply_feature_engineering
from src.preprocessing import build_preprocessing_pipeline, FeatureEngineeringTransformer

def test_feature_engineering_outputs():
    df_train, _, _ = load_datasets()
    df_feat = apply_feature_engineering(df_train)
    
    assert "coupon_venue_affinity" in df_feat.columns
    assert "distance_score" in df_feat.columns
    assert "age_num" in df_feat.columns
    assert df_feat["coupon_venue_affinity"].isnull().sum() == 0

def test_preprocessing_pipeline_transform():
    df_train, _, _ = load_datasets()
    df_feat = apply_feature_engineering(df_train)
    
    cat_cols = [c for c in df_feat.columns if df_feat[c].dtype == 'object' and c not in ["customer_id", "Y"]]
    num_cols = [c for c in df_feat.columns if df_feat[c].dtype != 'object' and c not in ["customer_id", "Y"]]
    
    preprocessor = build_preprocessing_pipeline(cat_cols, num_cols)
    X_trans = preprocessor.fit_transform(df_feat)
    
    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == len(df_train)
    assert not np.isnan(X_trans).any()
