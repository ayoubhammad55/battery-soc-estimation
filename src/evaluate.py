"""
evaluate.py — Shared evaluation utilities for SOC estimation models.

Functions
---------
evaluate_predictions : compute RMSE, MAE, R², Max Error on inverse-scaled SOC
print_metrics_table  : pretty-print a DataFrame of metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(
    name: str,
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    scaler_y,
) -> dict:
    """
    Inverse-scale predictions and compute regression metrics.

    Parameters
    ----------
    name          : Friendly model name for the report row.
    y_true_scaled : Ground-truth SOC in [0, 1] (scaler output).
    y_pred_scaled : Model prediction in [0, 1] (scaler output).
    scaler_y      : Fitted sklearn MinMaxScaler for the SOC target.

    Returns
    -------
    dict with keys: Model, RMSE, MAE, R2, Max Error, y_true, y_pred
        y_true / y_pred are in original SOC % (0–100).
    """
    y_true = scaler_y.inverse_transform(
        np.asarray(y_true_scaled).reshape(-1, 1)
    ).ravel()
    y_pred = scaler_y.inverse_transform(
        np.asarray(y_pred_scaled).reshape(-1, 1)
    ).ravel()

    return {
        "Model":     name,
        "RMSE":      float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":       float(mean_absolute_error(y_true, y_pred)),
        "R2":        float(r2_score(y_true, y_pred)),
        "Max Error": float(np.max(np.abs(y_true - y_pred))),
        "y_true":    y_true,
        "y_pred":    y_pred,
    }


def print_metrics_table(metrics_df: pd.DataFrame) -> None:
    """Pretty-print a metrics DataFrame to stdout."""
    print("\n" + "=" * 65)
    print(" SOC Estimation — Model Comparison")
    print("=" * 65)
    print(metrics_df.round(4).to_string())
    print("=" * 65)
