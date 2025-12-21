from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(predictions: pd.DataFrame) -> dict:
    """
    predictions doit contenir: y_true, y_pred
    """
    if predictions.empty:
        raise ValueError("predictions est vide")

    y_true = predictions["y_true"].to_numpy()
    y_pred = predictions["y_pred"].to_numpy()

    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    # R2 (déf. simple)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"rmse": rmse, "mae": mae, "r2": r2, "n": int(len(y_true))}