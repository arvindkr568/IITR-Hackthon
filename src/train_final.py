"""
Master End-to-End Execution Pipeline
Runs all implementation phases in sequence:
Phase 2: Ingestion -> Phase 3: Data Quality -> Phase 4: EDA -> Phase 5/6: Feature Engineering ->
Phase 7: Baseline -> Phase 8/9: Model Comparison & Tuning -> Phase 10: Explainability -> Phase 11: Predictions.
"""

import logging
import time
from src.data_loader import load_datasets
from src.data_quality import run_quality_check
from src.eda import generate_eda_plots
from src.baseline import train_baseline_model
from src.train import run_model_comparison
from src.explainability import generate_explainability_report
from src.predict import generate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_pipeline():
    start_time = time.time()
    logger.info("==========================================================================")
    logger.info("🚀 STARTING END-TO-END COUPON ACCEPTANCE ML PIPELINE EXECUTION")
    logger.info("==========================================================================")

    # Phase 2 & 3: Ingestion and Quality Audit
    logger.info("\n--- STEP 1: Data Ingestion & Quality Audit ---")
    load_datasets()
    run_quality_check()

    # Phase 4: Exploratory Data Analysis
    logger.info("\n--- STEP 2: Exploratory Data Analysis ---")
    generate_eda_plots()

    # Phase 7: Baseline Model
    logger.info("\n--- STEP 3: Baseline Logistic Regression Benchmark ---")
    train_baseline_model()

    # Phase 8 & 9: Model Comparison & Tuning
    logger.info("\n--- STEP 4: Candidate Model Comparison & Pipeline Serialization ---")
    scoreboard, winning_model_name, _ = run_model_comparison()

    # Phase 10: Model Explainability
    logger.info("\n--- STEP 5: Model Explainability & SHAP/Feature Importances ---")
    generate_explainability_report()

    # Phase 11: Test Predictions
    logger.info("\n--- STEP 6: Generating Final Test Set Predictions ---")
    generate_predictions()

    elapsed = time.time() - start_time
    logger.info("==========================================================================")
    logger.info(f"🎉 PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    logger.info(f"🏆 Winning Model: {winning_model_name}")
    logger.info("==========================================================================")

if __name__ == "__main__":
    run_pipeline()
