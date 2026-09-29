"""
Phase 6: Feature Engineering Module
Transforms raw customer, journey, and coupon attributes into predictive domain features
without introducing data leakage.
"""

import logging
import pandas as pd
import numpy as np
from src.config import (
    AGE_MAP, INCOME_MAP, EDU_MAP, FREQ_MAP, FREQ_COLS, 
    EXPIRATION_HOURS_MAP, TIME_HOURS_MAP
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies deterministic domain feature engineering to raw input DataFrame.
    Guarantees no future-information leakage.
    """
    df_feat = df.copy()

    # 1. Numerical Ordinal Mappings
    df_feat["age_num"] = df_feat["age"].map(AGE_MAP).fillna(26).astype(float)
    df_feat["income_num"] = df_feat["income"].map(INCOME_MAP).fillna(43750).astype(float)
    df_feat["edu_num"] = df_feat["education"].map(EDU_MAP).fillna(2).astype(float)

    # 2. Ordinal Venue Frequencies
    for col in FREQ_COLS:
        df_feat[f"{col}_num"] = df_feat[col].map(FREQ_MAP).fillna(1).astype(float)

    # 3. Contextual Time & Expiration Features
    df_feat["expiration_hours"] = df_feat["expiration"].map(EXPIRATION_HOURS_MAP).fillna(24).astype(float)
    df_feat["time_hour"] = df_feat["time"].map(TIME_HOURS_MAP).fillna(14).astype(float)

    # 4. Coupon Category - Customer Behavioral Affinity
    def get_coupon_affinity(row):
        coupon = str(row["coupon"])
        if "Bar" in coupon:
            return row["Bar_num"]
        elif "Coffee" in coupon:
            return row["CoffeeHouse_num"]
        elif "Carry" in coupon:
            return row["CarryAway_num"]
        elif "<20" in coupon:
            return row["RestaurantLessThan20_num"]
        elif "20-50" in coupon:
            return row["Restaurant20To50_num"]
        return 1.0

    df_feat["coupon_venue_affinity"] = df_feat.apply(get_coupon_affinity, axis=1)

    # 5. Composite Distance & Context Indicators
    df_feat["distance_score"] = (
        df_feat["toCoupon_GEQ5min"].fillna(1) + 
        df_feat["toCoupon_GEQ15min"].fillna(0) + 
        df_feat["toCoupon_GEQ25min"].fillna(0)
    )
    
    df_feat["is_no_urgent"] = (df_feat["destination"] == "No Urgent Place").astype(float)
    df_feat["is_alone"] = (df_feat["passanger"] == "Alone").astype(float)
    df_feat["is_sunny"] = (df_feat["weather"] == "Sunny").astype(float)
    
    # 6. Overall Activity Score
    df_feat["total_venue_activity"] = (
        df_feat["Bar_num"] + df_feat["CoffeeHouse_num"] + 
        df_feat["CarryAway_num"] + df_feat["RestaurantLessThan20_num"] + 
        df_feat["Restaurant20To50_num"]
    )

    logger.info(f"Engineered {len(df_feat.columns) - len(df.columns)} new features. Total columns: {len(df_feat.columns)}")
    return df_feat

if __name__ == "__main__":
    from src.data_loader import load_datasets
    train, test, _ = load_datasets()
    df_train_feat = apply_feature_engineering(train)
    print("Feature engineering successful! Sample columns:", df_train_feat.columns.tolist()[-8:])
