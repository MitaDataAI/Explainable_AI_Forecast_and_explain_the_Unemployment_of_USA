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


def metrics_by_windows(
    predictions: pd.DataFrame,
    *,
    val_start: str = "1983-01-01",
    val_end: str = "1989-12-01",
    test_start: str = "1990-01-01",
    test_end: str | None = None,
    y_pred_col: str = "y_pred",
) -> dict:
    """
    Calcule des métriques sur 2 fenêtres:
    - validation: [val_start, val_end]
    - test: [test_start, test_end]
    predictions: index datetime (t_end) et colonnes: y_true, y_pred (ou autre y_pred_col)
    """
    if predictions.empty:
        raise ValueError("predictions est vide")

    if "y_true" not in predictions.columns or y_pred_col not in predictions.columns:
        raise KeyError(f"predictions doit contenir 'y_true' et '{y_pred_col}'")

    preds = predictions.copy()
    preds = preds.rename(columns={y_pred_col: "y_pred"})  # regression_metrics attend y_pred

    val = preds.loc[val_start:val_end]
    test = preds.loc[test_start:] if test_end is None else preds.loc[test_start:test_end]

    out = {
        "validation": regression_metrics(val) if not val.empty else None,
        "test": regression_metrics(test) if not test.empty else None,
        "windows": {
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
            "y_pred_col": y_pred_col,
        },
    }
    return out


def format_window_report(window_metrics: dict) -> str:
    """
    Formate un affichage type:
    📊 Validation 83–89 — n=84 | MAE=... | RMSE=... | R²=...
    """
    m_val = window_metrics.get("validation")
    m_test = window_metrics.get("test")

    lines = []

    if m_val is not None:
        lines.append(
            f"📊 Validation 83–89 — n={m_val['n']} | MAE={m_val['mae']:.3f} | RMSE={m_val['rmse']:.3f} | R²={m_val['r2']:.3f}"
        )
    else:
        lines.append("📊 Validation 83–89 — (aucune prédiction dans cette fenêtre)")

    if m_test is not None:
        lines.append(
            f"📊 Test 90–… — n={m_test['n']} | MAE={m_test['mae']:.3f} | RMSE={m_test['rmse']:.3f} | R²={m_test['r2']:.3f}"
        )
    else:
        lines.append("📊 Test 90–… — (aucune prédiction dans cette fenêtre)")

    return "\n".join(lines)
