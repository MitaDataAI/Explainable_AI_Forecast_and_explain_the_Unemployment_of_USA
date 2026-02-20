# experiment_utils.py
# ============================================================
# Utilities: build "MAE (p)" pivot tables (DM test) by segments
# + robust wide builder from OOS artifacts (AR1/ARp/LR) + RIDGE bkt
# Key fix: normalize all dates to Month-Start (MS) to avoid NaN per segments.
# FIX (2026-02-19): _ensure_datetime_index now supports BOTH 'date' and 'ds'
# so MLflow-exported CSVs with a 'date' column align correctly.
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from math import sqrt, erf, isfinite
from typing import Iterable, Optional, Dict, List, Tuple


# =========================
# DM test p-value (approx)
# =========================
def _phi(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def dm_pvalue(loss_diff: np.ndarray, lags: int = 0) -> float:
    """
    Diebold-Mariano p-value (approx normal) for loss differences, with optional
    Newey-West correction (Bartlett weights).

    Parameters
    ----------
    loss_diff : array-like
        loss_model - loss_ref (e.g., abs_error_model - abs_error_ref).
    lags : int
        Newey-West max lag (often h-1 for horizon h).

    Returns
    -------
    float
        Two-sided p-value in [0,1] or np.nan.
    """
    x = np.asarray(loss_diff, dtype=float)
    x = x[np.isfinite(x)]
    T = x.size
    if T < 3:
        return np.nan

    dbar = x.mean()
    xc = x - dbar

    gamma0 = float(np.dot(xc, xc) / T)
    var = gamma0

    if lags > 0:
        max_k = min(lags, T - 1)
        for k in range(1, max_k + 1):
            w = 1.0 - k / (lags + 1.0)  # Bartlett weights
            cov = float(np.dot(xc[k:], xc[:-k]) / T)
            var += 2.0 * w * cov

    if var <= 0 or not np.isfinite(var):
        return np.nan

    stat = dbar / sqrt(var / T)
    p = 2.0 * (1.0 - _phi(abs(stat)))
    return float(max(0.0, min(1.0, p)))


# =========================
# Pivot builder: MAE (p)
# =========================
def make_mae_dm_pivot(
    wide: pd.DataFrame,
    segments: List[Tuple[str, Optional[str], str]],
    *,
    methods: Optional[Iterable[str]] = None,
    include_overall: bool = True,
    overall_label: str = "Ensemble",
    min_obs: int = 20,
    round_digits: int = 4,
    add_dm: bool = True,
    dm_lags: int = 11,
) -> pd.DataFrame:
    """
    Build a pivot table with cells like:
      "MAE" or "MAE (pvalue)"
    where pvalue is Diebold-Mariano vs best model in the window.

    Expected wide format:
      - DatetimeIndex OR a 'date' column
      - 'true' column
      - one column per method (e.g., AR1, ARp, LR, RIDGE)
    """
    df = wide.copy()

    # --- date handling
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("wide doit avoir un DatetimeIndex ou une colonne 'date'.")

    # ✅ IMPORTANT: tz-naive + normalisation Month-Start
    idx = pd.to_datetime(df.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)

    df.index = idx.to_period("M").to_timestamp(how="start")
    df = df.sort_index()

    # ✅ avoid duplicate dates after normalization
    df = df[~df.index.duplicated(keep="last")]

    if "true" not in df.columns:
        raise ValueError("wide doit contenir la colonne 'true'.")

    if methods is None:
        meths = [c for c in df.columns if c != "true"]
    else:
        meths = [m for m in methods if m in df.columns and m != "true"]

    if len(meths) == 0:
        return pd.DataFrame()

    full_start, full_end = df.index.min(), df.index.max()

    windows: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
    if include_overall:
        windows.append((full_start, full_end, overall_label))

    # ✅ Normalize segment bounds to Month-Start too
    for start, end, label in segments:
        s = pd.to_datetime(start, errors="coerce")
        e = pd.to_datetime(end, errors="coerce") if end is not None else full_end
        if pd.isna(s) or pd.isna(e):
            raise ValueError(f"Segment invalide: {(start, end, label)}")

        s = s.to_period("M").to_timestamp(how="start")
        e = e.to_period("M").to_timestamp(how="start")

        windows.append((s, e, label))

    rows = []

    for start, end, label in windows:
        sub = df.loc[start:end, ["true"] + meths].copy().dropna(subset=["true"])

        maes: Dict[str, float] = {}
        err_abs: Dict[str, pd.Series] = {}

        for m in meths:
            diffs = (sub["true"] - sub[m]).abs().dropna()
            err_abs[m] = diffs
            maes[m] = float(diffs.mean()) if diffs.shape[0] >= min_obs else np.nan

        finite_models = [m for m in meths if isfinite(maes.get(m, np.nan))]
        best_m = min(finite_models, key=lambda k: maes[k]) if finite_models else None

        for m in meths:
            mae_val = maes.get(m, np.nan)
            if not isfinite(mae_val):
                rows.append((m, label, np.nan))
                continue

            cell = f"{mae_val:.{round_digits}f}"

            if add_dm and best_m is not None and m != best_m:
                v1 = err_abs[m]
                v2 = err_abs[best_m]
                common = v1.index.intersection(v2.index)
                diff = (v1.loc[common] - v2.loc[common]).to_numpy()

                if diff.size >= min_obs:
                    pval = dm_pvalue(diff, lags=dm_lags)
                    if isfinite(pval):
                        cell = f"{cell} ({pval:.3f})"

            rows.append((m, label, cell))

    out = pd.DataFrame(rows, columns=["model", "period", "value"])
    pivot = out.pivot(index="model", columns="period", values="value")

    desired_cols = ([overall_label] if include_overall else []) + [lbl for _, _, lbl in segments]
    pivot = pivot.reindex(columns=desired_cols)
    pivot = pivot.reindex(index=meths)

    pivot.columns.name = "period"
    pivot.index.name = "model"
    return pivot


# =========================
# Build "wide" from OOS dfs
# =========================
def _normalize_month_start(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Force any datetime index onto month-start timestamps."""
    idx = pd.to_datetime(idx, errors="coerce")
    idx = idx[~pd.isna(idx)]
    return idx.to_period("M").to_timestamp(how="start")


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a DatetimeIndex, accepting:
    - a 'date' column (MLflow-exported CSVs often use this)
    - a 'ds' column (Nixtla / forecasting conventions)
    - already DatetimeIndex
    Then normalizes to Month-Start and removes duplicates.
    """
    d = df.copy()

    # ✅ FIX: accept BOTH 'date' and 'ds'
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"]).set_index("date")
    elif "ds" in d.columns:
        d["ds"] = pd.to_datetime(d["ds"], errors="coerce")
        d = d.dropna(subset=["ds"]).set_index("ds")

    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors="coerce")

    d = d[~d.index.isna()].sort_index()

    # normalize month start
    d.index = _normalize_month_start(d.index)

    # remove duplicate timestamps after normalization
    d = d[~d.index.duplicated(keep="last")]

    return d


def _pick_existing(d: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in d.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def to_wide_from_oos(
    *,
    df_ar1: Optional[pd.DataFrame] = None,
    df_arp: Optional[pd.DataFrame] = None,
    df_lr: Optional[pd.DataFrame] = None,
    bkt_ridge_final: Optional[pd.DataFrame] = None,
    # names in wide
    col_ar1: str = "AR1",
    col_arp: str = "ARp",
    col_lr: str = "LR",
    col_ridge: str = "RIDGE",
    # source columns (auto if None)
    y_true_candidates: Optional[List[str]] = None,
    yhat_candidates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a "wide" DataFrame with columns:
      date, true, AR1, ARp, LR, RIDGE (depending on what is provided)

    Notes
    -----
    - All dates are normalized to Month-Start to align models.
    - 'true' uses RIDGE's 'y' first (if provided), else AR1/ARp/LR y_true.
    """
    if y_true_candidates is None:
        y_true_candidates = ["y_true", "y", "y_obs", "y_actual"]
    if yhat_candidates is None:
        yhat_candidates = ["y_hat", "y_pred", "pred"]

    pieces: Dict[str, pd.DataFrame] = {}

    def _extract(df: pd.DataFrame, model_col_name: str, pred_key: str) -> pd.DataFrame:
        d = _ensure_datetime_index(df)

        ytrue_col = _pick_existing(d, y_true_candidates)
        yhat_col = _pick_existing(d, yhat_candidates + [pred_key])

        if yhat_col is None:
            raise ValueError(
                f"Impossible de trouver la prediction pour {model_col_name}. Colonnes={list(d.columns)}"
            )

        out = pd.DataFrame(index=d.index)
        if ytrue_col is not None:
            out["true"] = pd.to_numeric(d[ytrue_col], errors="coerce")
        out[model_col_name] = pd.to_numeric(d[yhat_col], errors="coerce")
        return out

    # OOS models
    if df_ar1 is not None:
        pieces[col_ar1] = _extract(df_ar1, col_ar1, col_ar1)
    if df_arp is not None:
        pieces[col_arp] = _extract(df_arp, col_arp, col_arp)
    if df_lr is not None:
        pieces[col_lr] = _extract(df_lr, col_lr, col_lr)

    # RIDGE bkt format
    if bkt_ridge_final is not None:
        d = bkt_ridge_final.copy()
        if "ds" not in d.columns:
            raise ValueError("bkt_ridge_final doit contenir une colonne 'ds'.")
        d["ds"] = pd.to_datetime(d["ds"], errors="coerce")
        d = d.dropna(subset=["ds"]).set_index("ds").sort_index()
        d.index = _normalize_month_start(d.index)

        if "y" not in d.columns:
            raise ValueError("bkt_ridge_final doit contenir la colonne 'y' (observé).")

        ridge_col = col_ridge if col_ridge in d.columns else _pick_existing(d, [col_ridge, "RIDGE"])
        if ridge_col is None:
            raise ValueError(f"bkt_ridge_final: prediction RIDGE introuvable. Colonnes={list(d.columns)}")

        out = pd.DataFrame(index=d.index)
        out["true"] = pd.to_numeric(d["y"], errors="coerce")
        out[col_ridge] = pd.to_numeric(d[ridge_col], errors="coerce")
        pieces[col_ridge] = out

    if not pieces:
        raise ValueError("Aucune source fournie (df_ar1/df_arp/df_lr/bkt_ridge_final).")

    # union index
    all_index = None
    for part in pieces.values():
        all_index = part.index if all_index is None else all_index.union(part.index)

    wide = pd.DataFrame(index=all_index)
    wide.index.name = "date"

    # true: priority RIDGE, then AR1 -> ARp -> LR
    wide["true"] = np.nan
    if col_ridge in pieces and "true" in pieces[col_ridge].columns:
        wide["true"] = pieces[col_ridge]["true"].reindex(wide.index)
    for k in [col_ar1, col_arp, col_lr]:
        if k in pieces and "true" in pieces[k].columns:
            wide["true"] = wide["true"].fillna(pieces[k]["true"].reindex(wide.index))

    # preds
    for k, part in pieces.items():
        if k in part.columns:
            wide[k] = part[k].reindex(wide.index)

    # reset index to column
    wide = wide.reset_index()

    # ensure date is datetime and normalized
    wide["date"] = pd.to_datetime(wide["date"], errors="coerce")
    wide = wide.dropna(subset=["date"]).sort_values("date")

    return wide