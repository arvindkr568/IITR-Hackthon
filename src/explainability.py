"""
Phase 10: Model Explainability Module
Extracts global feature importances from the trained winning model pipeline
and generates presentation-ready feature importance visualizations.
"""

import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FINAL_MODEL_PATH, FIGURES_DIR, TARGET_COL, ID_COL
from src.data_loader import load_datasets
from src.feature_engineering import apply_feature_engineering
from src.preprocessing import get_preprocessed_feature_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_explainability_report():
    """
    Loads saved model pipeline, computes feature importance, and saves visualization.
    """
    logger.info("Loading final model pipeline for explainability analysis...")
    pipeline = joblib.load(FINAL_MODEL_PATH)

    df_train, _, _ = load_datasets()
    df_train_feat = apply_feature_engineering(df_train)

    X = df_train_feat.drop(columns=[ID_COL, TARGET_COL])
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if X[c].dtype != 'object']

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Extract feature names after one-hot encoding
    feature_names = get_preprocessed_feature_names(preprocessor)

    # Extract feature importances
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        logger.warning("Classifier does not expose feature_importances_ or coef_")
        return

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    df_imp.to_csv(FIGURES_DIR.parent / "feature_importance.csv", index=False)
    logger.info(f"Top 10 Most Influential Features:\n{df_imp.head(10).to_string(index=False)}")

    # Plot top 20 features
    top_n = df_imp.head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=top_n, x="importance", y="feature", palette="crest", ax=ax)
    ax.set_title("Top 20 Feature Importances (Winning XGBoost Model)")
    ax.set_xlabel("Relative Feature Importance Score")
    ax.set_ylabel("Feature Name")
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / "feature_importance.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved feature importance chart to {fig_path}")

    return df_imp

if __name__ == "__main__":
    generate_explainability_report()
