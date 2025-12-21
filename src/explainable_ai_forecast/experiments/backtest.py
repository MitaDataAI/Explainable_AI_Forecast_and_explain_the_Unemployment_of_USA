from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    predictions: pd.DataFrame  # index = t_end (date de décision), colonnes: y_true, y_pred
    train_ends: list[pd.Timestamp]


def _drop_nan_train(X_tr: pd.DataFrame, y_tr: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    mask = X_tr.notna().all(axis=1) & y_tr.notna()
    return X_tr.loc[mask], y_tr.loc[mask]


def pseudo_oos_expanding(
    X: pd.DataFrame,
    y_future: pd.Series,
    *,
    model,
    min_train_n: int = 36,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> BacktestResult:
    """
    Backtest pseudo out-of-sample en expanding window.

    - À chaque t_end (date index), train sur <= t_end, puis prédire y_future[t_end]
      en utilisant X[t_end].

    Hypothèses:
    - X et y_future sont alignés sur le même index (date t)
    - y_future(t) représente la valeur à t+h (donc NaN sur la fin)
    """
    if min_train_n <= 1:
        raise ValueError("min_train_n doit être > 1")

    idx = X.index
    if not idx.equals(y_future.index):
        raise ValueError("X et y_future doivent avoir le même index")

    # Filtrage période backtest (sur t_end)
    t0 = pd.Timestamp(start) if start else idx.min()
    t1 = pd.Timestamp(end) if end else idx.max()
    valid_dates = idx[(idx >= t0) & (idx <= t1)]

    rows = []
    train_ends = []

    for t_end in valid_dates:
        X_tr = X.loc[:t_end]
        y_tr = y_future.loc[:t_end]

        X_tr, y_tr = _drop_nan_train(X_tr, y_tr)
        if len(X_tr) < min_train_n:
            continue

        x_fore = X.loc[[t_end]]
        y_true = y_future.loc[t_end]

        # si on ne peut pas scorer à cette date
        if x_fore.isna().any(axis=1).iloc[0] or pd.isna(y_true):
            continue

        # fit modèle (clone léger : on suppose que "model" est un estimateur sklearn prêt à être refit)
        m = model.__class__(**getattr(model, "get_params", lambda: {})()) if hasattr(model, "get_params") else model
        if hasattr(m, "fit"):
            m.fit(X_tr.to_numpy(), y_tr.to_numpy())
        else:
            raise TypeError("model doit avoir une méthode fit()")

        y_pred = float(m.predict(x_fore.to_numpy())[0])

        rows.append({"date": t_end, "y_true": float(y_true), "y_pred": y_pred})
        train_ends.append(pd.Timestamp(t_end))

    pred_df = pd.DataFrame(rows).set_index("date").sort_index()
    return BacktestResult(predictions=pred_df, train_ends=train_ends)