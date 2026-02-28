from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
from feast import FeatureStore

# =========================================================
# 1) Utils: trouver la racine projet + repo Feast
# =========================================================
def find_project_root(start: Path, marker: str = "2_data_processing") -> Path:
    """Walk up parents until we find a folder named `marker`."""
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(
        f"Cannot find project root: marker '{marker}' not found from {start}"
    )

def get_feast_repo_path(project_root: Path) -> Path:
    """Return the expected Feast repo path."""
    feast_repo = (
        project_root
        / "2_data_processing"
        / "feature_store"
        / "feast_repo"
        / "feature_repo"
    )
    if not feast_repo.exists():
        raise FileNotFoundError(f"Feast repo path not found: {feast_repo}")
    return feast_repo

# =========================================================
# 2) Utils: build entity_df
# =========================================================
def build_entity_df(
    series_ids: list[str],
    start: str = "1960-01-01",
    end: str = "2025-08-01",
    freq: str = "MS",
) -> pd.DataFrame:
    """Create the entity dataframe (series_id, date) for Feast."""
    dates = pd.date_range(start=start, end=end, freq=freq)
    entity_df = pd.MultiIndex.from_product(
        [series_ids, dates],
        names=["series_id", "date"],
    ).to_frame(index=False)
    return entity_df

# =========================================================
# 3) Utils: load features from Feast and pivot to wide
# =========================================================
def load_wide_from_feast(
    feature_ref: str,
    series_ids: list[str],
    start: str = "1960-01-01",
    end: str = "2025-08-01",
    freq: str = "MS",
    project_marker: str = "2_data_processing",
    expected_value_col: str | None = None,
) -> pd.DataFrame:
    """
    Load a single feature (e.g., "raw_value:value" or "stationary_value:value")
    from Feast, then pivot to wide dataframe (index=date, columns=series_id).

    Parameters
    ----------
    feature_ref : str
        Feast feature reference, e.g. "raw_value:value" or "stationary_value:value"
    expected_value_col : str | None
        If you know the exact returned column name (e.g. "raw_value__value"),
        you can pass it. Otherwise it auto-detects the returned value column.
    """
    # Locate Feast repo
    project_root = find_project_root(Path.cwd(), marker=project_marker)
    feast_repo_path = get_feast_repo_path(project_root)

    # Init Feast store
    store = FeatureStore(repo_path=str(feast_repo_path))

    # Build entity df
    entity_df = build_entity_df(series_ids, start=start, end=end, freq=freq)

    # Retrieve features
    df_long = store.get_historical_features(
        entity_df=entity_df,
        features=[feature_ref],
    ).to_df()

    # Identify the value column produced by Feast
    if expected_value_col is not None:
        value_col = expected_value_col
        if value_col not in df_long.columns:
            raise KeyError(
                f"expected_value_col='{value_col}' not found. "
                f"Available columns: {list(df_long.columns)}"
            )
    else:
        # ✅ Case A: Feast returns directly "value"
        if "value" in df_long.columns and "series_id" in df_long.columns and "date" in df_long.columns:
            value_col = "value"
        else:
            # ✅ Case B (standard): "<feature_view>__<feature_name>"
            candidates = [c for c in df_long.columns if "__" in c and c not in ("series_id", "date")]
            if len(candidates) == 0:
                raise KeyError(
                    "No Feast value column found. "
                    f"Available columns: {list(df_long.columns)}"
                )

            if len(candidates) > 1:
                view, feat = feature_ref.split(":")
                preferred = f"{view}__{feat}"
                value_col = preferred if preferred in candidates else candidates[0]
            else:
                value_col = candidates[0]

    # LONG -> WIDE
    df_wide = (
        df_long
        .rename(columns={value_col: "value"})
        .pivot(index="date", columns="series_id", values="value")
        .sort_index()
    )

    # Ensure datetime index
    df_wide.index = pd.to_datetime(df_wide.index)
    return df_wide


# =========================================================
# 4) Utils: Build dataset (Target UNRATE + Exogenous variables)
# =========================================================
def build_unrate_exog_dataset(
    df_stationary: pd.DataFrame,
    target_id: str = "UNRATE",
    value_col: str = "value",
    series_col: str = "series_id",
    date_col: str = "date",
    unique_id: str = "UNRATE",
    dropna: bool = True,
    align_to_month_start: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Target variable: stationary UNRATE
    Exogenous variables: other macro series at the same timestamp

    Input df_stationary must be LONG format with columns:
      - series_id, date, value

    Returns:
      - df_model: wide dataset [date, y, <exog...>]
      - ts_lr: MLForecast long format [unique_id, ds, y, <exog...>]
      - exog_cols: list of exogenous column names
    """
    required = {series_col, date_col, value_col}
    missing = required - set(df_stationary.columns)
    if missing:
        raise ValueError(f"df_stationary is missing columns: {sorted(missing)}")

    df = df_stationary[[series_col, date_col, value_col]].copy()

    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Optional: enforce month-start timestamps (MS)
    if align_to_month_start:
        df[date_col] = (
            df[date_col]
            .dt.to_period("M")
            .dt.to_timestamp(how="start")
            .dt.normalize()
        )

    # -------------------------
    # Target y
    # -------------------------
    df_y = (
        df[df[series_col] == target_id]
        .sort_values(date_col)
        .rename(columns={value_col: "y"})
        .reset_index(drop=True)
    )

    # -------------------------
    # Exogenous X (all other series)
    # -------------------------
    df_x_long = df[df[series_col] != target_id].copy()

    # Keep only dates present in y
    df_x_long = df_x_long[df_x_long[date_col].isin(df_y[date_col])]

    df_x = (
        df_x_long
        .pivot_table(index=date_col, columns=series_col, values=value_col, aggfunc="last")
        .reset_index()
    )

    # Merge y + X
    df_model = (
        df_y[[date_col, "y"]]
        .merge(df_x, on=date_col, how="left")
    )

    if dropna:
        df_model = df_model.dropna()

    # -------------------------
    # MLForecast format
    # -------------------------
    ts_lr = df_model.rename(columns={date_col: "ds"}).copy()
    ts_lr["unique_id"] = unique_id

    exog_cols = [c for c in ts_lr.columns if c not in ["unique_id", "ds", "y"]]
    ts_lr = ts_lr[["unique_id", "ds", "y"] + exog_cols].copy()

    # Ensure ds is month-start normalized
    if align_to_month_start:
        ts_lr["ds"] = (
            pd.to_datetime(ts_lr["ds"])
            .dt.to_period("M")
            .dt.to_timestamp(how="start")
            .dt.normalize()
        )

    return df_model, ts_lr, exog_cols

# ============================================================
# utils_ar_p.py — AR(p) : CV rolling MAE + fit/predict + Conformal (pos)
# (Bagging OFF dans ce fichier)
# ============================================================

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.ar_model import AutoReg


# ============================================================
# ---------- Paramètres (defaults) ----------
# ============================================================

DEFAULT_CFG = dict(
    h=12,
    min_train_n=36,          # ≥ 3 ans
    trend="c",               # "c" (constante) ou "n" (sans constante)
    p_grid=range(1, 13),     # p ∈ {1,…,12}
    cv_update_every_months=36,
    cv_anchor=pd.Timestamp("1983-01-01"),
    # Bagging OFF
    use_bagging=False,
    # Conformal
    use_conformal=True,
    alpha=0.05,
    step_size=12,
    pi_windows=3,            # si tu veux + stable, monte à 24
)


# ============================================================
# ---------- Utils ----------
# ============================================================

def months_since(anchor: pd.Timestamp, t: pd.Timestamp) -> int:
    return (t.year - anchor.year) * 12 + (t.month - anchor.month)


def rolling_mae_for_p(
    y_series: pd.Series,
    p: int,
    h: int,
    min_train: int,
    trend: str,
) -> float:
    rows = []
    last_t_end = y_series.index.max() - relativedelta(months=h)

    for t_end in y_series.index:
        if t_end > last_t_end:
            break

        y_tr = y_series.loc[:t_end]
        if len(y_tr) < max(min_train, p + 1):
            continue

        model = AutoReg(y_tr, lags=int(p), old_names=False, trend=trend).fit()
        fc = model.predict(start=len(y_tr), end=len(y_tr) + h - 1)
        yhat_h = float(fc.iloc[-1])

        t_fore = t_end + relativedelta(months=h)
        if t_fore in y_series.index:
            rows.append((t_fore, yhat_h, float(y_series.loc[t_fore])))

    if not rows:
        return float("inf")

    tmp = pd.DataFrame(rows, columns=["date", "y_hat", "y_true"]).set_index("date")
    return float(mean_absolute_error(tmp["y_true"], tmp["y_hat"]))


def select_p_by_cv(
    y_tr: pd.Series,
    p_grid,
    h: int,
    min_train: int,
    trend: str,
) -> int:
    best_p, best_score = None, float("inf")
    for p in p_grid:
        score = rolling_mae_for_p(y_tr, int(p), h, min_train, trend)
        if score < best_score:
            best_score, best_p = score, int(p)
    return int(best_p if best_p is not None else 1)


def fit_predict_ar_p(y_tr: pd.Series, h: int, *, trend: str = "c", p: int = 1) -> float:
    m = AutoReg(y_tr, lags=int(p), old_names=False, trend=trend).fit()
    fc = m.predict(start=len(y_tr), end=len(y_tr) + h - 1)
    return float(fc.iloc[-1])


def conformal_q_from_past_windows_pos_ARp(
    y: pd.Series,
    i_end: int,
    *,
    h: int = 12,
    step_size: int = 12,
    pi_windows: int = 24,
    trend: str = "c",
    p: int = 12,
    alpha: float = 0.05,
    min_train_n: int = 36,
) -> float:
    errs = []

    for k in range(1, int(pi_windows) + 1):
        i_cal = i_end - k * int(step_size)
        i_cal_fore = i_cal + int(h)

        if i_cal < 0 or i_cal_fore >= len(y):
            continue

        y_tr_cal = y.iloc[: i_cal + 1]
        if len(y_tr_cal) < max(int(min_train_n), int(p) + 2):
            continue

        try:
            yhat_cal = fit_predict_ar_p(y_tr_cal, h=int(h), trend=trend, p=int(p))
        except Exception:
            continue

        err = abs(float(y.iloc[i_cal_fore]) - float(yhat_cal))
        if np.isfinite(err):
            errs.append(err)

    if len(errs) == 0:
        return np.nan

    n = len(errs)
    q_level = np.ceil((n + 1) * (1 - float(alpha))) / n
    q_level = min(max(q_level, 0.0), 1.0)
    return float(np.quantile(errs, q_level))


# ============================================================
# ---------- Pseudo-OOS runner ----------
# ============================================================

def run_pseudo_oos_ar_p_no_bagging(
    y: pd.Series,
    *,
    h: int = DEFAULT_CFG["h"],
    min_train_n: int = DEFAULT_CFG["min_train_n"],
    trend: str = DEFAULT_CFG["trend"],
    p_grid=DEFAULT_CFG["p_grid"],
    cv_update_every_months: int = DEFAULT_CFG["cv_update_every_months"],
    cv_anchor: pd.Timestamp = DEFAULT_CFG["cv_anchor"],
    use_conformal: bool = DEFAULT_CFG["use_conformal"],
    alpha: float = DEFAULT_CFG["alpha"],
    step_size: int = DEFAULT_CFG["step_size"],
    pi_windows: int = DEFAULT_CFG["pi_windows"],
    verbose_cv: bool = True,
):
    """
    Retourne:
      - df_oos: index=date_fore (t_end + h), colonnes:
          y_hat, y_true, p_selected, lo_95, hi_95, y_hat_base
      - meta: dict avec last_model, last_fit_end
    """
    rows = []
    last_model = None
    last_fit_end = None
    current_p = None

    last_t_end = y.index.max() - relativedelta(months=h)

    for i_end, t_end in enumerate(y.index):
        if t_end > last_t_end:
            break

        y_tr = y.loc[:t_end]
        if len(y_tr) < int(min_train_n):
            continue

        # Re-CV à partir de cv_anchor tous les cv_update_every_months
        if t_end >= cv_anchor:
            m = months_since(cv_anchor, t_end)
            need_cv = (m % int(cv_update_every_months) == 0)
        else:
            need_cv = False

        if current_p is None and not need_cv:
            current_p = 1

        if need_cv:
            current_p = select_p_by_cv(y_tr, p_grid, int(h), int(min_train_n), trend)
            if verbose_cv:
                print(f"[CV] {t_end.date()} → p* = {current_p}")

        # Fit + forecast h (BASE)
        arp = AutoReg(y_tr, lags=int(current_p), old_names=False, trend=trend).fit()
        last_model = arp
        last_fit_end = t_end

        fc = arp.predict(start=len(y_tr), end=len(y_tr) + int(h) - 1)
        yhat_h = float(fc.iloc[-1])
        yhat_base = yhat_h

        t_fore = t_end + relativedelta(months=h)
        if t_fore not in y.index:
            continue
        y_true = float(y.loc[t_fore])

        lo_95 = np.nan
        hi_95 = np.nan

        if use_conformal:
            q = conformal_q_from_past_windows_pos_ARp(
                y=y, i_end=i_end,
                h=int(h), step_size=int(step_size), pi_windows=int(pi_windows),
                trend=trend, p=int(current_p), alpha=float(alpha),
                min_train_n=int(min_train_n),
            )
            if np.isfinite(q):
                lo_95 = float(yhat_h - q)
                hi_95 = float(yhat_h + q)

        rows.append((t_fore, yhat_h, y_true, int(current_p), lo_95, hi_95, yhat_base))

    if rows:
        df_oos = (
            pd.DataFrame(
                rows,
                columns=["date", "y_hat", "y_true", "p_selected", "lo_95", "hi_95", "y_hat_base"],
            )
            .set_index("date")
            .sort_index()
        )
    else:
        df_oos = pd.DataFrame(
            columns=["y_hat", "y_true", "p_selected", "lo_95", "hi_95", "y_hat_base"]
        )
        df_oos.index = pd.to_datetime(pd.Index([]))

    meta = {"last_model": last_model, "last_fit_end": last_fit_end}
    return df_oos, meta