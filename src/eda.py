"""
Phase 4: Exploratory Data Analysis (EDA) Module
Generates statistical summaries and presentation-ready charts for coupon acceptance patterns.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.config import TARGET_COL, FIGURES_DIR
from src.data_loader import load_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Styling configuration for presentation-ready figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 18
})

def generate_eda_plots():
    df_train, _, _ = load_datasets()
    logger.info("Generating EDA charts...")

    # 1. Target Class Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df_train, x=TARGET_COL, palette=["#e74c3c", "#2ecc71"], ax=ax)
    ax.set_title("Coupon Acceptance Target Distribution (Y)")
    ax.set_xticklabels(["Rejected (0)", "Accepted (1)"])
    ax.set_ylabel("Customer Count")
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height} ({height/len(df_train):.1%})",
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha="center", va="center", color="white", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=300)
    plt.close()

    # 2. Acceptance Rate by Coupon Category
    fig, ax = plt.subplots(figsize=(10, 6))
    coupon_acc = df_train.groupby("coupon")[TARGET_COL].mean().reset_index().sort_values(by=TARGET_COL, ascending=False)
    sns.barplot(data=coupon_acc, x="coupon", y=TARGET_COL, palette="viridis", ax=ax)
    ax.set_title("Acceptance Rate by Coupon Category")
    ax.set_ylabel("Acceptance Rate (Mean Y)")
    ax.set_ylim(0, 1)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.02),
                    ha="center", va="bottom", fontweight="bold")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "acceptance_by_coupon_type.png", dpi=300)
    plt.close()

    # 3. Acceptance by Expiration and Time
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_train, x="time", y=TARGET_COL, hue="expiration", palette="magma", ax=ax)
    ax.set_title("Coupon Acceptance Rate by Time of Day & Expiration")
    ax.set_ylabel("Acceptance Rate")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "acceptance_by_expiration_time.png", dpi=300)
    plt.close()

    # 4. Acceptance by Venue Visit Frequency (Bar vs CoffeeHouse vs Restaurants)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    venues = ["Bar", "CoffeeHouse", "CarryAway", "RestaurantLessThan20"]
    order = ["never", "less1", "1~3", "4~8", "gt8"]

    for idx, venue in enumerate(venues):
        r, c = idx // 2, idx % 2
        venue_df = df_train.groupby(venue)[TARGET_COL].mean().reindex(order).reset_index()
        sns.barplot(data=venue_df, x=venue, y=TARGET_COL, palette="Blues_d", ax=axes[r, c])
        axes[r, c].set_title(f"Acceptance Rate by {venue} Frequency")
        axes[r, c].set_ylim(0, 1)
        axes[r, c].set_ylabel("Acceptance Rate")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "acceptance_by_venue_frequency.png", dpi=300)
    plt.close()

    # 5. Acceptance by Distance / Context
    fig, ax = plt.subplots(figsize=(9, 5))
    dist_df = pd.DataFrame({
        "GEQ15min": df_train.groupby("toCoupon_GEQ15min")[TARGET_COL].mean(),
        "GEQ25min": df_train.groupby("toCoupon_GEQ25min")[TARGET_COL].mean(),
        "Same_Direction": df_train.groupby("direction_same")[TARGET_COL].mean()
    }).T
    dist_df.plot(kind="bar", stacked=False, color=["#95a5a6", "#2980b9"], figsize=(9, 5), ax=ax)
    ax.set_title("Acceptance Rate by Distance & Driving Direction Context")
    ax.set_ylabel("Acceptance Rate")
    ax.set_xticklabels(["Distance >= 15 min", "Distance >= 25 min", "Driving in Same Direction"], rotation=0)
    ax.legend(["Indicator = 0", "Indicator = 1"])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "acceptance_by_distance_direction.png", dpi=300)
    plt.close()

    logger.info(f"Successfully generated all EDA charts under {FIGURES_DIR}")

if __name__ == "__main__":
    generate_eda_plots()
