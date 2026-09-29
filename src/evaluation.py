"""
STOCKSENSE Evaluation & Split Module
Implements chronological train/val/test splits, regression/classification metric benchmarks, 
sMAPE, under-forecast rate, residual analysis, probability calibration, and metric comparison table generators.
"""

from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
import logging

logger = logging.getLogger("stocksense.evaluation")

def chronological_split(
    df: pd.DataFrame, 
    train_ratio: float = 0.70, 
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs strict chronological train / validation / test split based on dates.
    Zero data leakage: data is strictly ordered by date.
    """
    df = df.sort_values(by="date").reset_index(drop=True)
    unique_dates = df["date"].sort_values().unique()
    n_dates = len(unique_dates)
    
    n_train = int(n_dates * train_ratio)
    n_val = int(n_dates * val_ratio)
    
    train_dates = unique_dates[:n_train]
    val_dates = unique_dates[n_train:n_train + n_val]
    test_dates = unique_dates[n_train + n_val:]
    
    train_df = df[df["date"].isin(train_dates)].copy()
    val_df = df[df["date"].isin(val_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    
    logger.info(
        f"Chronological Split: Train ({train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')}, {len(train_df)} rows), "
        f"Val ({val_df['date'].min().strftime('%Y-%m-%d')} to {val_df['date'].max().strftime('%Y-%m-%d')}, {len(val_df)} rows), "
        f"Test ({test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')}, {len(test_df)} rows)"
    )
    return train_df, val_df, test_df

def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) safe against 0 demand.
    sMAPE = 100% * (2 / n) * sum(|y_true - y_pred| / (|y_true| + |y_pred| + 1e-5))
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-5
    return float(np.mean(np.abs(y_pred - y_true) / denominator) * 100.0)

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
    """Calculates comprehensive regression metrics: MAE, RMSE, sMAPE, R2, Bias, Under-forecast Rate."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smape = calculate_smape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Bias: mean(y_pred - y_true). Negative means under-forecasting on average.
    bias = float(np.mean(y_pred - y_true))
    
    # Under-forecast rate: percentage of times actual demand > predicted demand (lost sales risk)
    under_forecast_rate = float(np.mean(y_true > y_pred) * 100.0)
    
    return {
        "Model": model_name,
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "sMAPE (%)": round(float(smape), 2),
        "R2": round(float(r2), 4),
        "Bias": round(float(bias), 3),
        "Under-Forecast Rate (%)": round(float(under_forecast_rate), 2)
    }

def evaluate_classification(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.50, model_name: str = "Model") -> Dict[str, Any]:
    """Calculates classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred_prob = np.asarray(y_pred_prob, dtype=float)
    y_pred_bin = (y_pred_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred_bin)
    prec = precision_score(y_true, y_pred_bin, zero_division=0)
    rec = recall_score(y_true, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred_prob)
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = average_precision_score(y_true, y_pred_prob)
    except Exception:
        pr_auc = 0.5
        
    cm = confusion_matrix(y_true, y_pred_bin).tolist()
    
    return {
        "Model": model_name,
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1 Score": round(float(f1), 4),
        "ROC-AUC": round(float(roc_auc), 4),
        "PR-AUC": round(float(pr_auc), 4),
        "Confusion Matrix": cm
    }
