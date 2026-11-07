"""Duration metrics shared by training and checkpoint evaluation."""

from typing import Dict
import numpy as np

def compute_metrics_s(y_true_min: np.ndarray, y_pred_min: np.ndarray) -> Dict[str, float]:
    y_true_s = np.asarray(y_true_min, dtype='float64') * 60.0
    y_pred_s = np.asarray(y_pred_min, dtype='float64') * 60.0
    err = y_true_s - y_pred_s
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    medae = float(np.median(np.abs(err)))
    wape = float(np.sum(np.abs(err)) / np.maximum(np.sum(y_true_s), 1e-8))
    ss_tot = np.sum((y_true_s - np.mean(y_true_s)) ** 2)
    ss_res = np.sum(err ** 2)
    r2 = float(1.0 - ss_res / np.maximum(ss_tot, 1e-8))
    log_true = np.log1p(np.maximum(np.asarray(y_true_min, dtype='float64'), 0.0))
    log_pred = np.log1p(np.maximum(np.asarray(y_pred_min, dtype='float64'), 0.0))
    rmsle = float(np.sqrt(np.mean((log_true - log_pred) ** 2)))
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MedAE': medae,
        'WAPE': wape,
        'RMSLE': rmsle,
        'R2': r2
    }
