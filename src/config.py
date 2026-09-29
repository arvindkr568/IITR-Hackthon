"""
Phase 1: Configuration Module
Coupon Acceptance ML Project Configuration
Centralized management of paths, random seeds, schema definitions, and feature configurations.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Create directories if they do not exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, OUTPUTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# File Paths
TRAIN_DATA_PATH = RAW_DATA_DIR / "train.csv"
TEST_DATA_PATH = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "sample_submission.csv"
FINAL_MODEL_PATH = MODELS_DIR / "model_pipeline.joblib"
METRICS_REPORT_PATH = OUTPUTS_DIR / "model_metrics.csv"
SUBMISSION_PATH = BASE_DIR / "submission.csv"

# Global Constants
SEED = 42
TARGET_COL = "Y"
ID_COL = "customer_id"
N_SPLITS = 5

# Categorical & Numerical Schema Definition
RAW_NUMERICAL_COLS = [
    "temperature", "has_children", "toCoupon_GEQ5min", 
    "toCoupon_GEQ15min", "toCoupon_GEQ25min", "direction_same", "direction_opp"
]

RAW_CATEGORICAL_COLS = [
    "destination", "passanger", "weather", "time", "coupon", "expiration",
    "gender", "age", "maritalStatus", "education", "occupation", "income",
    "car", "Bar", "CoffeeHouse", "CarryAway", "RestaurantLessThan20", "Restaurant20To50"
]

# Ordinal Mappings for Feature Engineering
AGE_MAP = {
    "below21": 18, "21": 21, "26": 26, "31": 31, "36": 36, "41": 41, "46": 46, "50plus": 55
}

INCOME_MAP = {
    "Less than $12500": 6250, "$12500 - $24999": 18750, "$25000 - $37499": 31250,
    "$37500 - $49999": 43750, "$50000 - $62499": 56250, "$62500 - $74999": 68750,
    "$75000 - $87499": 81250, "$87500 - $99999": 93750, "$100000 or More": 112500
}

EDU_MAP = {
    "Some High School": 0, "High School Graduate": 1, "Some college - no degree": 2,
    "Associates degree": 3, "Bachelors degree": 4, "Graduate degree (Masters or Doctorate)": 5
}

FREQ_MAP = {
    "never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4
}

FREQ_COLS = ["Bar", "CoffeeHouse", "CarryAway", "RestaurantLessThan20", "Restaurant20To50"]

EXPIRATION_HOURS_MAP = {
    "2h": 2, "1d": 24
}

TIME_HOURS_MAP = {
    "7AM": 7, "10AM": 10, "2PM": 14, "6PM": 18, "10PM": 22
}
