from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
from feast import FeatureStore
from dataclasses import dataclass
from mlforecast.utils import PredictionIntervals
from sklearn.model_selection import ParameterGrid, ParameterSampler

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


# -------------------------------------------------------------------
# 1) LOAD : Feast -> WIDE (inchangé dans l'esprit, un peu durci)
# -------------------------------------------------------------------
def load_wide_from_feast(
    feature_ref: str,
    series_ids: list[str],
    start: str = "1960-01-01",
    end: str = "2025-08-01",
    freq: str = "MS",
    project_marker: str = "2_data_processing",
    expected_value_col: str | None = None,
    *,
    ensure_ms: bool = True,
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
    ensure_ms : bool
        If True, force index to Month Start ("MS") timestamps.
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

    # Basic sanity checks
    required = {"series_id", "date"}
    missing = required - set(df_long.columns)
    if missing:
        raise KeyError(f"Missing columns from Feast result: {missing}. Got: {list(df_long.columns)}")

    # Identify value column produced by Feast
    if expected_value_col is not None:
        value_col = expected_value_col
        if value_col not in df_long.columns:
            raise KeyError(
                f"expected_value_col='{value_col}' not found. "
                f"Available columns: {list(df_long.columns)}"
            )
    else:
        # Case A: Feast returns directly "value"
        if "value" in df_long.columns:
            value_col = "value"
        else:
            # Case B: "<feature_view>__<feature_name>"
            candidates = [c for c in df_long.columns if "__" in c and c not in ("series_id", "date")]
            if not candidates:
                raise KeyError(
                    "No Feast value column found. "
                    f"Available columns: {list(df_long.columns)}"
                )

            if len(candidates) > 1 and ":" in feature_ref:
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

    # Ensure datetime index + (option) MS alignment
    idx = pd.to_datetime(df_wide.index)
    if ensure_ms:
        idx = idx.to_period("M").to_timestamp(how="start").normalize()
    df_wide.index = idx
    df_wide.index.name = "date"

    return df_wide


# -------------------------------------------------------------------
# 2) FEATURE ENGINEERING : appliquer des lags sur exog (LAG variable)
# -------------------------------------------------------------------
def apply_exog_lags(
    ts: pd.DataFrame,
    exog_cols: list[str],
    *,
    lags: int | list[int] = 12,
    group_col: str = "unique_id",
    time_col: str = "ds",
    drop_original_exog: bool = True,
    dropna: bool = True,
    target_col: str = "y",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create lagged versions of exogenous columns.

    Returns
    -------
    ts_out : pd.DataFrame
        DataFrame with lagged exog columns added (and optionally original exog removed).
    exog_cols_out : list[str]
        New exogenous column names (lagged only).
    """
    if isinstance(lags, int):
        lags_list = [lags]
    else:
        lags_list = list(lags)

    if not exog_cols:
        return ts.copy(), []

    ts_out = ts.sort_values([group_col, time_col]).copy()

    new_exogs: list[str] = []
    for L in lags_list:
        if L <= 0:
            raise ValueError(f"lags must be positive. Got {L}")
        for c in exog_cols:
            new_c = f"{c}_lag{L}"
            ts_out[new_c] = ts_out.groupby(group_col, sort=False)[c].shift(L)
            new_exogs.append(new_c)

    if drop_original_exog:
        ts_out = ts_out.drop(columns=exog_cols)

    if dropna:
        needed = [target_col] + new_exogs
        ts_out = ts_out.dropna(subset=needed).reset_index(drop=True)

    return ts_out, new_exogs


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


# -----------------------------
# Helpers temps
# -----------------------------
def _ensure_ms(x):
    x = pd.Timestamp(x)
    return x.to_period("M").to_timestamp(how="start").normalize()

def _n_windows_monthly(ds_start, ds_end):
    return (ds_end.year - ds_start.year) * 12 + (ds_end.month - ds_start.month) + 1

def _slice_cv_block(ts, cutoff_start, n_windows, h):
    cutoff_end = cutoff_start + relativedelta(months=n_windows - 1)
    ds_end = cutoff_end + relativedelta(months=h)
    return ts[ts["ds"] <= ds_end].copy(), cutoff_end, ds_end


# -----------------------------
# Spec modèle (tu ajoutes juste ici)
# -----------------------------
@dataclass
class ModelSpec:
    name: str                         # "LR", "RIDGE", "LGBM", etc.
    build_mlf: Callable[[str, Dict[str, Any]], MLForecast]  # (freq, params)->MLForecast
    pred_col: str                     # colonne de forecast dans cv (ex "LR")
    tunable: bool = False
    param_space: Optional[Dict[str, Iterable[Any]]] = None
    search: str = "random"            # "random"|"grid"
    n_iter: int = 50                  # si random
    tune_cv_windows: int = 6
    tune_every_months: int = 36
    use_conformal_in_tune: bool = False
    fixed_params: Optional[Dict[str, Any]] = None

# -----------------------------
# Tuner générique (pour tous les modèles)
# -----------------------------
def _tune_on_train(
    ts_train: pd.DataFrame,
    *,
    spec: ModelSpec,
    freq: str,
    h: int,
    levels: List[int],
    seed: int,
    pi_windows_cap: int,
    min_train_n: Optional[int] = None,
) -> tuple[Optional[Dict[str, Any]], float]:

    if min_train_n is not None and len(ts_train) < int(min_train_n):
        return None, float("nan")

    if not spec.tunable:
        return (spec.fixed_params or {}), float("nan")

    if not spec.param_space:
        raise ValueError(f"{spec.name}: tunable=True mais param_space=None")

    # conformal au tuning (souvent OFF)
    pi_tune = (
        PredictionIntervals(h=h, n_windows=min(int(spec.tune_cv_windows), int(pi_windows_cap)),
                            method="conformal_distribution")
        if spec.use_conformal_in_tune else None
    )

    # générateur d'essais
    if spec.search == "grid":
        sampler = ParameterGrid(spec.param_space)
    else:
        sampler = ParameterSampler(spec.param_space, n_iter=int(spec.n_iter), random_state=int(seed))

    best_params = None
    best_score = np.inf

    for params in sampler:
        params = dict(params)

        mlf = spec.build_mlf(freq, params)
        cv = mlf.cross_validation(
            df=ts_train,
            h=h,
            step_size=1,
            n_windows=int(spec.tune_cv_windows),
            prediction_intervals=pi_tune,
            level=list(levels) if pi_tune is not None else None,
            fitted=False,
            static_features=[],
            dropna=True,
        )
        score = mean_absolute_error(cv["y"], cv[spec.pred_col])

        if score < best_score:
            best_score = float(score)
            best_params = params

    return best_params, float(best_score)

# -----------------------------
# Runner multi-modèles : tu ajoutes juste un spec dans la liste
# -----------------------------
def run_backtesting_generic(
    ts: pd.DataFrame,
    *,
    model_specs: List[ModelSpec],
    freq: str,
    h: int,
    exp_start,
    exp_end,
    step_size: int,
    pi_windows: int,
    levels: List[int],
    seed: int = 0,
    min_train_n: Optional[int] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:

    # run modèle par modèle puis merge
    bkts = []
    metas = {}
    bundles = {}   # ✅ NOUVEAU

    for spec in model_specs:
        bkt_m, meta_m = backtest_one_model_tune_blocks(
            ts,
            spec=spec,
            freq=freq,
            h=h,
            exp_start=exp_start,
            exp_end=exp_end,
            step_size=step_size,
            pi_windows=pi_windows,
            levels=levels,
            seed=seed,
            min_train_n=min_train_n,
        )
        metas[spec.name] = meta_m

        # ✅ NOUVEAU : bundle explicabilité (si dispo)
        fitted_models = meta_m.get("fitted_models", None)
        train_dates   = meta_m.get("train_fit_dates", None) or meta_m.get("train_periods", None)
        features      = meta_m.get("features", None)
        preprocs      = meta_m.get("preprocs", None)

        if isinstance(fitted_models, list) and len(fitted_models) > 0 and train_dates is not None and features is not None:
            bundles[spec.name] = {
                "models": fitted_models,
                "train_fit_dates": list(pd.to_datetime(pd.Index(train_dates))),
                "features": list(features),
                "preprocs": preprocs,
                "params": meta_m,   # pratique: tu gardes toute la meta
            }

        if len(bkt_m):
            bkts.append(bkt_m)

    if not bkts:
        return pd.DataFrame(), {"error": "aucun modèle n’a produit de backtest", "metas": metas, "bundles": bundles}

    # merge wide sur clés
    keys = ["unique_id", "ds", "cutoff", "y"]
    bkt_all = bkts[0].copy()

    for b in bkts[1:]:
        keep_cols = [c for c in b.columns if c not in bkt_all.columns or c in keys]
        bkt_all = bkt_all.merge(b[keep_cols], on=keys, how="outer")

    bkt_all = bkt_all.sort_values(["unique_id", "ds", "cutoff"]).reset_index(drop=True)

    # ✅ 1 ligne par ds : dernier cutoff
    bkt_final = (
        bkt_all.sort_values(["unique_id", "ds", "cutoff"])
               .groupby(["unique_id", "ds"], as_index=False)
               .tail(1)
               .reset_index(drop=True)
    )

    meta = {"metas": metas, "bundles": bundles} 
    return bkt_final, meta


def backtest_one_model_tune_blocks(
    ts: pd.DataFrame,
    *,
    spec: ModelSpec,
    freq: str,
    h: int,
    exp_start,
    exp_end,
    step_size: int,
    pi_windows: int,
    levels: List[int],
    seed: int = 0,
    min_train_n: Optional[int] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:

    ts = ts.copy()
    ts["ds"] = (
        pd.to_datetime(ts["ds"], errors="coerce")
          .dt.to_period("M")
          .dt.to_timestamp(how="start")
          .dt.normalize()
    )
    if ts["ds"].isna().any():
        bad = ts[ts["ds"].isna()].head()
        raise ValueError(f"Dates 'ds' invalides après parsing. Exemples:\n{bad}")

    exp_start = _ensure_ms(exp_start)
    exp_end   = _ensure_ms(exp_end)

    cutoff_start_all = exp_start - relativedelta(months=h)
    cutoff_end_all   = exp_end   - relativedelta(months=h)
    total_partitions = _n_windows_monthly(cutoff_start_all, cutoff_end_all)

    # anti-fuite
    ts = ts[ts["ds"] <= exp_end].copy()

    # conformal au backtest (ici ON)
    pi = PredictionIntervals(h=h, n_windows=int(pi_windows), method="conformal_distribution")

    # blocs de retrain/tuning
    blocks = []
    remaining = int(total_partitions)
    cur = cutoff_start_all
    while remaining > 0:
        n_win = min(int(spec.tune_every_months), remaining)
        blocks.append((cur, n_win))
        cur = cur + relativedelta(months=n_win)
        remaining -= n_win

    all_bkts = []
    params_history = []
    tune_history = []

    # ✅ explicabilité (sans refit) : on veut des modèles récupérables
    fitted_models: list = []
    train_fit_dates: list = []

    # colonnes candidates selon versions
    candidate_model_cols = ["fitted", "fitted_model", "model", "models", "forecaster", "estimator"]

    for block_idx, (cutoff_start_blk, n_windows_blk) in enumerate(blocks, start=1):
        ts_blk, _, _ = _slice_cv_block(ts, cutoff_start_blk, int(n_windows_blk), h)

        ts_train_for_tune = ts[ts["ds"] <= cutoff_start_blk].copy()
        if min_train_n is not None and len(ts_train_for_tune) < int(min_train_n):
            continue

        best_params, tune_mae = _tune_on_train(
            ts_train_for_tune,
            spec=spec,
            freq=freq,
            h=h,
            levels=levels,
            seed=seed,
            pi_windows_cap=int(pi_windows),
            min_train_n=min_train_n,
        )
        if best_params is None:
            continue

        params_history.append({
            "model": spec.name,
            "block": block_idx,
            "cutoff_start": cutoff_start_blk,
            "n_windows": int(n_windows_blk),
            **best_params,
        })
        tune_history.append({
            "model": spec.name,
            "block": block_idx,
            "cutoff_start": cutoff_start_blk,
            "tune_mae": float(tune_mae),
        })

        # --------- CV predictions (avec PI)
        mlf_blk = spec.build_mlf(freq, best_params)

        bkt_blk = mlf_blk.cross_validation(
            df=ts_blk,
            h=h,
            step_size=int(step_size),
            n_windows=int(n_windows_blk),
            prediction_intervals=pi,
            level=list(levels),
            fitted=True,
            static_features=[],
            dropna=True,
        )

        bkt_blk[f"{spec.name}_tune_block"] = block_idx
        bkt_blk[f"{spec.name}_tune_mae"] = float(tune_mae)

        # --------- DEBUG (tu peux laisser, ou enlever après)
        # print(f"[DEBUG {spec.name}] cols: {list(bkt_blk.columns)}")

        # --------- 1) Essai: récupérer modèles directement dans le DF (si colonne existe)
        col_found = next((c for c in candidate_model_cols if c in bkt_blk.columns), None)

        if col_found is not None:
            tmp_fit = (
                bkt_blk[["cutoff", col_found]]
                .dropna()
                .drop_duplicates("cutoff")
                .sort_values("cutoff")
            )
            fitted_models.extend(tmp_fit[col_found].tolist())
            train_fit_dates.extend(pd.to_datetime(tmp_fit["cutoff"]).tolist())

        else:
            # --------- 2) Fallback garanti: 1 modèle par bloc (fit sur ts_train_for_tune)
            # Utile si ta version MLForecast ne renvoie pas l'objet fitted dans le DataFrame
            mlf_fitted = spec.build_mlf(freq, best_params)
            mlf_fitted.fit(ts_train_for_tune, static_features=[])
            fitted_models.append(mlf_fitted)
            train_fit_dates.append(pd.to_datetime(cutoff_start_blk))

        all_bkts.append(bkt_blk)

    if not all_bkts:
        return pd.DataFrame(), {"error": f"{spec.name}: aucun bloc produit"}

    bkt = pd.concat(all_bkts, ignore_index=True)

    # filtre exp exact
    bkt = bkt[(bkt["ds"] >= exp_start) & (bkt["ds"] <= exp_end)].copy()
    bkt = bkt.sort_values(["unique_id", "ds", "cutoff"]).reset_index(drop=True)

    # features pour explicabilité
    _features = None
    if "exog_cols" in globals():
        try:
            _features = list(globals()["exog_cols"])
        except Exception:
            _features = None

    meta = dict(
        model=spec.name,
        h=int(h),
        step_size=int(step_size),
        exp_start=exp_start,
        exp_end=exp_end,
        cutoff_start=cutoff_start_all,
        cutoff_end=cutoff_end_all,
        partitions=int(total_partitions),
        pi_windows=int(pi_windows),
        tune_every_months=int(spec.tune_every_months),
        tune_cv_windows=int(spec.tune_cv_windows),
        tunable=bool(spec.tunable),
        search=spec.search,
        n_iter=int(spec.n_iter),
        use_conformal_in_tune=bool(spec.use_conformal_in_tune),
        param_space=spec.param_space,
        params_history=params_history,
        tune_history=tune_history,

        # ✅ bundle explicabilité
        fitted_models=fitted_models,
        train_fit_dates=train_fit_dates,
        features=_features,
        preprocs=None,
    )
    return bkt, meta


# Transform Backtesting

def prepare_backtest_partitions(
    bkt_df: pd.DataFrame,
    *,
    exp_start,
    exp_end,
    date_col: str = "ds",
    bins=None,
    labels=None,
) -> pd.DataFrame:
    """
    Clean backtest dataframe and create time partitions.

    Parameters
    ----------
    bkt_df : DataFrame
        Backtest dataframe (ex: bkt_models_final)
    exp_start : datetime-like
        Start of evaluation window
    exp_end : datetime-like
        End of evaluation window
    date_col : str
        Date column (default: "ds")
    bins : list-like
        Partition boundaries
    labels : list-like
        Partition labels

    Returns
    -------
    DataFrame
        Backtest dataframe with partition column
    """

    df = bkt_df.copy()

    # ------------------------------------------------
    # Date cleaning
    # ------------------------------------------------
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if pd.api.types.is_datetime64tz_dtype(df[date_col]):
        df[date_col] = df[date_col].dt.tz_convert(None)

    df = df.dropna(subset=[date_col])

    # ------------------------------------------------
    # Filter evaluation window
    # ------------------------------------------------
    exp_start = pd.Timestamp(exp_start)
    exp_end = pd.Timestamp(exp_end)

    df = df[(df[date_col] >= exp_start) & (df[date_col] <= exp_end)].reset_index(drop=True)

    # ------------------------------------------------
    # Default partitions
    # ------------------------------------------------
    if bins is None:
        bins = pd.to_datetime([
            "1990-01-01",
            "2000-01-01",
            "2009-01-01",
            "2020-01-01",
            "2025-09-01",
        ])

    if labels is None:
        labels = [
            "1990-1999",
            "2000-2008",
            "2009-2019",
            "2020-end",
        ]

    # ------------------------------------------------
    # Create partitions
    # ------------------------------------------------
    df["partition"] = pd.cut(
        df[date_col],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    df = df.dropna(subset=["partition"]).reset_index(drop=True)

    return df


