"""
Phase 5: Preprocessing Pipeline Module
Constructs reusable Scikit-Learn pipelines for handling missing values, encoding categorical 
variables, and scaling numeric features. Prevents train/test leakage.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from src.config import (
    RAW_CATEGORICAL_COLS, RAW_NUMERICAL_COLS, ID_COL, TARGET_COL
)
from src.feature_engineering import apply_feature_engineering

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for feature engineering step."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return apply_feature_engineering(X)

def build_preprocessing_pipeline(
    categorical_cols: list = None,
    numerical_cols: list = None,
    scale_numeric: bool = False
) -> ColumnTransformer:
    """
    Creates a ColumnTransformer pipeline for imputing and encoding tabular data.
    
    Args:
        categorical_cols: List of categorical feature names.
        numerical_cols: List of numerical feature names.
        scale_numeric: If True, applies StandardScaler to numerical columns.
        
    Returns:
        ColumnTransformer instance.
    """
    if categorical_cols is None:
        categorical_cols = RAW_CATEGORICAL_COLS
    if numerical_cols is None:
        numerical_cols = RAW_NUMERICAL_COLS

    # Categorical Pipeline: Impute missing with 'missing' -> OneHotEncoder
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Numerical Pipeline: Impute missing with median -> (optional) StandardScaler
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipeline = Pipeline(num_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, categorical_cols),
            ("num", num_pipeline, numerical_cols)
        ],
        remainder="drop"
    )
    
    return preprocessor

def get_preprocessed_feature_names(preprocessor: ColumnTransformer) -> list:
    """Extracts output feature names from fitted ColumnTransformer."""
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "cat":
            ohe = trans.named_steps["onehot"]
            feature_names.extend(ohe.get_feature_names_out(cols))
        elif name == "num":
            feature_names.extend(cols)
    return feature_names

if __name__ == "__main__":
    from src.data_loader import load_datasets
    train, test, _ = load_datasets()
    train_feat = apply_feature_engineering(train)
    
    cat_cols = [c for c in train_feat.columns if train_feat[c].dtype == 'object' and c not in [ID_COL, TARGET_COL]]
    num_cols = [c for c in train_feat.columns if train_feat[c].dtype != 'object' and c not in [ID_COL, TARGET_COL]]
    
    pipeline = build_preprocessing_pipeline(cat_cols, num_cols)
    X_trans = pipeline.fit_transform(train_feat)
    print("Preprocessing completed cleanly! Transformed X shape:", X_trans.shape)
