"""
Phase 8 & 9: Evaluation Framework Module
Calculates comprehensive classification metrics: Accuracy, Precision, Recall, 
F1-Score, ROC-AUC, PR-AUC, LogLoss, and Confusion Matrix.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_predictions(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: np.ndarray = None
) -> Dict[str, float]:
    """
    Computes all standard classification metrics for model evaluation.
    
    Args:
        y_true: True binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted probabilities of class 1.
        
    Returns:
        Dict containing all evaluation metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4)
    }
    
    if y_prob is not None:
        # Clip probabilities to avoid log loss infinity error
        y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
        metrics["pr_auc"] = round(average_precision_score(y_true, y_prob), 4)
        metrics["log_loss"] = round(log_loss(y_true, y_prob_clipped), 4)
        
    cm = confusion_matrix(y_true, y_pred)
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])

    return metrics

def format_metrics_table(metrics_list: list) -> pd.DataFrame:
    """
    Converts a list of model metric dictionaries into a clean formatted comparison DataFrame.
    """
    df_metrics = pd.DataFrame(metrics_list)
    cols_order = ["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "log_loss"]
    existing_cols = [c for c in cols_order if c in df_metrics.columns]
    return df_metrics[existing_cols].sort_values(by="roc_auc", ascending=False)
