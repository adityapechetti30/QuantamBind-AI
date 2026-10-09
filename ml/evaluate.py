"""Model evaluation metrics for QuantumBind AI."""

from __future__ import annotations
import math
from typing import Dict, List, Sequence, Union


def calculate_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> Dict[str, float]:
    """Calculate regression metrics: MAE, RMSE, and R² directly from actual predictions.

    Calculated via exact closed-form formulations:
      - MAE  = (1/n) * sum(|y_true - y_pred|)
      - MSE  = (1/n) * sum((y_true - y_pred)^2)
      - RMSE = sqrt(MSE)
      - R²   = 1 - (SS_res / SS_tot)
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, y_pred has {len(y_pred)}."
        )
    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics on empty predictions.")

    n = len(y_true)
    abs_errors = [abs(float(yt) - float(yp)) for yt, yp in zip(y_true, y_pred)]
    sq_errors = [(float(yt) - float(yp)) ** 2 for yt, yp in zip(y_true, y_pred)]

    mae = sum(abs_errors) / n
    mse = sum(sq_errors) / n
    rmse = math.sqrt(mse)

    mean_y = sum(float(yt) for yt in y_true) / n
    ss_tot = sum((float(yt) - mean_y) ** 2 for yt in y_true)
    ss_res = sum(sq_errors)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }


def print_evaluation_summary(
    metrics: Dict[str, float],
    dataset_name: str = "Test Set",
    sample_count: int = 0,
) -> None:
    """Print formatted evaluation report."""
    print("----------------------------------------")
    print(f" Evaluation Report: {dataset_name}")
    if sample_count:
        print(f" Sample Count: {sample_count}")
    print("----------------------------------------")
    print(f"  MAE  (Mean Absolute Error): {metrics['mae']:.4f}")
    print(f"  RMSE (Root Mean Sq. Error): {metrics['rmse']:.4f}")
    print(f"  R²   (Coeff. Determination): {metrics['r2']:.4f}")
    print("----------------------------------------")
