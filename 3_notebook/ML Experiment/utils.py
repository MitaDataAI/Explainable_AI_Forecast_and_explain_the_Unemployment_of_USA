from pathlib import Path
import pandas as pd
from feast import FeatureStore
import numpy as np

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid, ParameterSampler

from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals


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


def make_ts_from_wide(
    df_wide: pd.DataFrame,
    target_col: str = "UNRATE",
    lags: int | list[int] = 12,
    *,
    unique_id_value: str | None = None,
    date_name: str = "ds",
    target_name: str = "y",
    drop_original_exog: bool = True,
    dropna: bool = True,
    include_target_lags: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a MLForecast-style dataframe from a wide dataframe:
    [unique_id, ds, y, exog...] then apply exogenous lags.
    """
    if target_col not in df_wide.columns:
        raise KeyError(
            f"target_col='{target_col}' not found in columns: {list(df_wide.columns)}"
        )

    if isinstance(lags, int):
        lags = [lags]

    df = df_wide.sort_index().copy()

    if unique_id_value is None:
        unique_id_value = target_col

    # -----------------------------
    # Include or exclude target lags
    # -----------------------------
    if include_target_lags:
        exog_cols = list(df.columns)
    else:
        exog_cols = [c for c in df.columns if c != target_col]

    ts_df = df.reset_index().rename(columns={"date": date_name})
    ts_df[target_name] = ts_df[target_col]
    ts_df["unique_id"] = unique_id_value

    ts_df = ts_df[["unique_id", date_name, target_name] + exog_cols].copy()
    ts_df = ts_df.sort_values(["unique_id", date_name]).copy()

    exog_cols_lagged = []
    for c in exog_cols:
        for lag in lags:
            new_c = f"{c}_lag{lag}"
            ts_df[new_c] = ts_df.groupby("unique_id")[c].shift(lag)
            exog_cols_lagged.append(new_c)

    if drop_original_exog:
        ts_df = ts_df.drop(columns=exog_cols, errors="ignore")

    if dropna:
        ts_df = ts_df.dropna(
            subset=[target_name] + exog_cols_lagged
        ).reset_index(drop=True)

    return ts_df, exog_cols_lagged

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
# Spec modèle 
# -----------------------------
@dataclass
class ModelSpec:
    name: str                         #
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

    meta = {"metas": metas, "bundles": bundles}  # ✅ NOUVEAU
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


import pandas as pd


def prepare_backtest_partitions(
    df: pd.DataFrame,
    date_col: str = "ds",
    start_date=None,
    end_date=None,
    bins=None,
    labels=None,
    partition_col: str = "partition",
    cutoff_col: str = "cutoff",
    drop_na_date: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Prépare un DataFrame de backtest en :
    1. copiant le DataFrame
    2. convertissant la colonne date en datetime
    3. supprimant éventuellement la timezone
    4. filtrant sur [start_date, end_date]
    5. créant une colonne de partition temporelle avec pd.cut

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame source.
    date_col : str, default="ds"
        Nom de la colonne de date à utiliser pour les partitions.
    start_date : str or pd.Timestamp, optional
        Date minimale incluse.
    end_date : str or pd.Timestamp, optional
        Date maximale incluse.
    bins : list-like, optional
        Bornes temporelles pour pd.cut.
        Exemple :
        ["1990-01-01","2000-01-01","2009-01-01","2020-01-01","2025-09-01"]
    labels : list-like, optional
        Labels associés aux bins.
        Exemple :
        ["1990-1999", "2000-2008", "2009-2019", "2020-end"]
    partition_col : str, default="partition"
        Nom de la colonne de partition créée.
    cutoff_col : str, default="cutoff"
        Nom éventuel de la colonne cutoff pour affichage debug.
    drop_na_date : bool, default=True
        Si True, supprime les lignes où la date est invalide.
    verbose : bool, default=True
        Si True, affiche un aperçu et les effectifs par partition.

    Retour
    ------
    pd.DataFrame
        DataFrame préparé avec colonne de partition.
    """

    out = df.copy()

    # Conversion date
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    # Enlever timezone si besoin
    if pd.api.types.is_datetime64tz_dtype(out[date_col]):
        out[date_col] = out[date_col].dt.tz_convert(None)

    # Drop NA dates
    if drop_na_date:
        out = out.dropna(subset=[date_col])

    # Filtre période
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        out = out[out[date_col] >= start_date]

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        out = out[out[date_col] <= end_date]

    out = out.reset_index(drop=True)

    # Partitionnement
    if bins is not None and labels is not None:
        bins = pd.to_datetime(bins)
        out[partition_col] = pd.cut(
            out[date_col],
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=True,
        )
        out = out.dropna(subset=[partition_col]).reset_index(drop=True)

    # Affichage
    if verbose:
        cols_to_show = [date_col, partition_col]
        if cutoff_col in out.columns:
            cols_to_show.insert(1, cutoff_col)

        print(out[cols_to_show].head(10))

        if partition_col in out.columns:
            print(out[partition_col].value_counts().sort_index())

    return out


# =========================================================
# ✅ EXPLICABILITÉ ROLLING GÉNÉRALE
# Permutation MAE + Permutation Deviance + SHAP share
# =========================================================

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


# =========================================================
# (0) Métrique deviance = MSE
# =========================================================
def mse_deviance(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean((y_true[m] - y_pred[m]) ** 2)) if np.any(m) else np.nan


# =========================================================
# (1) Helpers
# =========================================================
def _unwrap_estimator_from_mlf(maybe_mlf, preferred_key=None):
    if hasattr(maybe_mlf, "models_") and isinstance(getattr(maybe_mlf, "models_"), dict):
        d = maybe_mlf.models_
        if preferred_key is not None and preferred_key in d:
            return d[preferred_key]
        return next(iter(d.values()))
    return maybe_mlf


def _predict_any(model_obj, X_feat: pd.DataFrame, *, model_key=None) -> np.ndarray:
    est = _unwrap_estimator_from_mlf(model_obj, preferred_key=model_key)

    drop_cols = [c for c in ["ds", "unique_id", "y"] if c in X_feat.columns]
    X = X_feat.drop(columns=drop_cols, errors="ignore")

    fn = getattr(est, "feature_names_in_", None)
    if fn is not None:
        fn = [c for c in fn if c in X.columns]
        if len(fn) > 0:
            X = X[fn]

    return np.asarray(est.predict(X), float)


# =========================================================
# (2) Permutation importance ROLLING
# =========================================================
def perm_ratio_pseudo_oos(
    *,
    exp_results: dict,
    df_all: pd.DataFrame,
    target_col: str,
    h: int,
    metric_fn,
    restrict_eval_window=None,
    n_repeats: int = 10,
    random_state: int = 42,
    model_key: str | None = None,
):
    models = list(exp_results["models"])
    feats = list(exp_results["features"])
    periods = pd.to_datetime(pd.Index(exp_results["train_periods"]))

    df = df_all.copy().sort_values("ds").reset_index(drop=True)
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"]).reset_index(drop=True)

    if restrict_eval_window is not None:
        s, e = pd.to_datetime(restrict_eval_window[0]), pd.to_datetime(restrict_eval_window[1])
        df = df[(df["ds"] >= s) & (df["ds"] <= e)].copy()

    if len(df) <= h + 5:
        return pd.DataFrame(columns=["feature", "ratio_mean", "ratio_std", "n_windows"])

    # Align horizon
    X_full = df[["ds"] + feats].copy()
    y_full = df[target_col].shift(-h)

    X_full = X_full.iloc[:-h].reset_index(drop=True)
    y_full = y_full.iloc[:-h].reset_index(drop=True)

    ok = ~X_full[feats].isna().any(axis=1) & y_full.notna()
    X_full = X_full.loc[ok].reset_index(drop=True)
    y_full = y_full.loc[ok].reset_index(drop=True)

    if len(X_full) < 30:
        return pd.DataFrame(columns=["feature", "ratio_mean", "ratio_std", "n_windows"])

    rng = np.random.default_rng(random_state)
    ratios = {f: [] for f in feats}

    n = min(len(models), len(periods))

    for i in range(n):
        model = models[i]
        end_time = periods[i]

        idx = X_full["ds"] <= end_time
        Xw = X_full.loc[idx].copy()
        yw = y_full.loc[idx].copy()

        if len(Xw) < 25:
            continue

        yhat = _predict_any(model, Xw, model_key=model_key)
        base = metric_fn(yw.to_numpy(), yhat)

        if not np.isfinite(base) or base == 0:
            continue

        for f in feats:
            v = Xw[f].to_numpy()
            perm_scores = []

            for _ in range(n_repeats):
                Xp = Xw.copy()
                Xp[f] = v[rng.permutation(len(v))]
                ypp = _predict_any(model, Xp, model_key=model_key)
                perm_scores.append(metric_fn(yw.to_numpy(), ypp))

            ratios[f].append(float(np.mean(perm_scores) / base))

    rows = []
    for f, arr in ratios.items():
        if len(arr) > 0:
            rows.append({
                "feature": f,
                "ratio_mean": float(np.mean(arr)),
                "ratio_std": float(np.std(arr)),
                "n_windows": int(len(arr)),
            })

    if len(rows) == 0:
        return pd.DataFrame(columns=["feature", "ratio_mean", "ratio_std", "n_windows"])

    return (
        pd.DataFrame(rows)
        .sort_values("ratio_mean", ascending=False)
        .reset_index(drop=True)
    )


# =========================================================
# (3) SHAP SHARE ROLLING
# =========================================================
def shap_share_pseudo_oos(
    *,
    exp_results: dict,
    model_key: str | None = None,
):
    feats = list(exp_results["features"])
    models = list(exp_results["models"])

    weights = {f: [] for f in feats}

    for model in models:
        est = _unwrap_estimator_from_mlf(model, preferred_key=model_key)

        if hasattr(est, "coef_"):
            w = np.abs(np.asarray(est.coef_, float)).reshape(-1)
        elif hasattr(est, "feature_importances_"):
            w = np.abs(np.asarray(est.feature_importances_, float)).reshape(-1)
        else:
            continue

        w = w[:len(feats)]
        total = w.sum()

        if total == 0 or not np.isfinite(total):
            continue

        share = w / total

        for f, val in zip(feats[:len(share)], share):
            weights[f].append(float(val))

    rows = []
    for f, arr in weights.items():
        if len(arr) > 0:
            rows.append({
                "feature": f,
                "shap_share_mean": float(np.mean(arr)),
                "shap_share_std": float(np.std(arr)),
                "n_windows": int(len(arr)),
            })

    if len(rows) == 0:
        return pd.DataFrame(columns=["feature", "shap_share_mean", "shap_share_std", "n_windows"])

    return (
        pd.DataFrame(rows)
        .sort_values("shap_share_mean", ascending=False)
        .reset_index(drop=True)
    )


# =========================================================
# (4) Fonction générale par partition
# =========================================================
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def compute_explainability_by_partition(
    *,
    bkt_score: pd.DataFrame,
    ts_df: pd.DataFrame,
    meta_models: dict,
    target_col: str = "y",
    date_col: str = "ds",
    partition_col: str = "partition",
    horizon: int = 12,
    normalize_month_start: bool = True,
    verbose: bool = True,
):
    """
    Calcule l'explicabilité rolling par partition :
    - Permutation MAE
    - Permutation Deviance
    - SHAP share

    Paramètres
    ----------
    bkt_score : pd.DataFrame
        DataFrame de backtest contenant au moins [date_col, partition_col].
    ts_df : pd.DataFrame
        Série temporelle complète contenant au moins [date_col, target_col] + features.
    meta_models : dict
        Dictionnaire contenant obligatoirement meta_models["bundles"] avec structure :
        {
            "bundles": {
                model_name: {
                    "models": [...],
                    "train_fit_dates": [...],
                    "features": [...]
                }
            }
        }
    target_col : str
        Nom de la variable cible.
    date_col : str
        Nom de la colonne date.
    partition_col : str
        Nom de la colonne partition.
    horizon : int
        Horizon de forecast h.
    normalize_month_start : bool
        Si True, convertit les dates en début de mois.
    verbose : bool
        Si True, affiche la progression.

    Retour
    ------
    tuple
        (
            results_perm_mae_by_part,
            results_perm_deviance_by_part,
            results_shap_share_by_part
        )
    """

    results_perm_mae_by_part = {}
    results_perm_deviance_by_part = {}
    results_shap_share_by_part = {}

    if partition_col not in bkt_score.columns:
        raise ValueError(f"Colonne absente dans bkt_score: {partition_col}")

    if date_col not in bkt_score.columns:
        raise ValueError(f"Colonne absente dans bkt_score: {date_col}")

    if date_col not in ts_df.columns:
        raise ValueError(f"Colonne absente dans ts_df: {date_col}")

    if target_col not in ts_df.columns:
        raise ValueError(f"Colonne absente dans ts_df: {target_col}")

    if "bundles" not in meta_models or not meta_models["bundles"]:
        raise ValueError("Aucun bundle disponible dans meta_models['bundles'].")

    bkt_local = bkt_score.copy()
    bkt_local[date_col] = pd.to_datetime(bkt_local[date_col], errors="coerce")

    if normalize_month_start:
        bkt_local[date_col] = (
            bkt_local[date_col]
            .dt.to_period("M")
            .dt.to_timestamp(how="start")
            .dt.normalize()
        )

    bkt_local = bkt_local.dropna(subset=[date_col]).reset_index(drop=True)

    partitions = sorted(bkt_local[partition_col].astype(str).unique())

    if verbose:
        print("Partitions détectées:", partitions)

    for partition in partitions:

        if verbose:
            print(f"\n==============================")
            print(f"PARTITION: {partition}")
            print(f"==============================")

        # -----------------------------------------------------
        # 1) Filtrage temporel des données
        # -----------------------------------------------------
        ts_part = ts_df.copy()
        ts_part[date_col] = pd.to_datetime(ts_part[date_col], errors="coerce")

        if normalize_month_start:
            ts_part[date_col] = (
                ts_part[date_col]
                .dt.to_period("M")
                .dt.to_timestamp(how="start")
                .dt.normalize()
            )

        ts_part = ts_part.dropna(subset=[date_col]).copy()

        dates_part = bkt_local.loc[
            bkt_local[partition_col].astype(str) == partition,
            date_col
        ]

        if len(dates_part) == 0:
            continue

        start_p = dates_part.min()
        end_p = dates_part.max()

        ts_part = ts_part[
            (ts_part[date_col] >= start_p) &
            (ts_part[date_col] <= end_p)
        ].copy()

        if verbose:
            print("Rows used:", len(ts_part))

        if ts_part.empty:
            results_perm_mae_by_part.setdefault(partition, {})
            results_perm_deviance_by_part.setdefault(partition, {})
            results_shap_share_by_part.setdefault(partition, {})
            continue

        # -----------------------------------------------------
        # 2) Calcul par modèle
        # -----------------------------------------------------
        for model_name, bundle in meta_models["bundles"].items():

            if verbose:
                print(f"→ Computing {model_name} for {partition}")

            if not all(k in bundle for k in ["models", "train_fit_dates", "features"]):
                if verbose:
                    print(f"   ⚠️ Bundle incomplet pour {model_name}.")
                continue

            models_all = bundle["models"]
            dates_all = pd.to_datetime(bundle["train_fit_dates"], errors="coerce")
            feats_all = list(bundle["features"])

            valid_mask = ~pd.isna(dates_all)
            models_all = [m for m, keep in zip(models_all, valid_mask) if keep]
            dates_all = [d for d, keep in zip(dates_all, valid_mask) if keep]

            # filtrage rolling jusqu'à la fin de partition
            mask = [d <= end_p for d in dates_all]

            models_filtered = [m for m, keep in zip(models_all, mask) if keep]
            dates_filtered = [d for d, keep in zip(dates_all, mask) if keep]

            if len(models_filtered) == 0:
                if verbose:
                    print("   ⚠️ Aucun modèle valide pour cette partition.")
                continue

            exp = {
                "models": models_filtered,
                "features": feats_all,
                "train_periods": dates_filtered,
            }

            # ---------------------------
            # Permutation MAE
            # ---------------------------
            try:
                df_perm_mae = perm_ratio_pseudo_oos(
                    exp_results=exp,
                    df_all=ts_part,
                    target_col=target_col,
                    h=horizon,
                    metric_fn=mean_absolute_error,
                    restrict_eval_window=(str(start_p.date()), str(end_p.date())),
                    model_key=model_name,
                )
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ perm_mae failed: {e}")
                df_perm_mae = pd.DataFrame(
                    columns=["feature", "ratio_mean", "ratio_std", "n_windows"]
                )

            # ---------------------------
            # Permutation Deviance
            # ---------------------------
            try:
                df_perm_dev = perm_ratio_pseudo_oos(
                    exp_results=exp,
                    df_all=ts_part,
                    target_col=target_col,
                    h=horizon,
                    metric_fn=mse_deviance,
                    restrict_eval_window=(str(start_p.date()), str(end_p.date())),
                    model_key=model_name,
                )
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ perm_dev failed: {e}")
                df_perm_dev = pd.DataFrame(
                    columns=["feature", "ratio_mean", "ratio_std", "n_windows"]
                )

            # ---------------------------
            # SHAP share
            # ---------------------------
            try:
                df_shap = shap_share_pseudo_oos(
                    exp_results=exp,
                    model_key=model_name,
                )
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ shap_share failed: {e}")
                df_shap = pd.DataFrame(
                    columns=["feature", "shap_share_mean", "shap_share_std", "n_windows"]
                )

            # ---------------------------
            # Stockage
            # ---------------------------
            results_perm_mae_by_part.setdefault(partition, {})[model_name] = df_perm_mae
            results_perm_deviance_by_part.setdefault(partition, {})[model_name] = df_perm_dev
            results_shap_share_by_part.setdefault(partition, {})[model_name] = df_shap

    if verbose:
        print("\n✅ Explainability ready by partition")
        print("Partitions calculées:", list(results_perm_mae_by_part.keys()))

    return (
        results_perm_mae_by_part,
        results_perm_deviance_by_part,
        results_shap_share_by_part,
    )


import numpy as np
import pandas as pd


def build_score_and_explainability_tables(
    *,
    bkt_score: pd.DataFrame,
    results_perm_mae_by_part: dict,
    results_perm_deviance_by_part: dict,
    results_shap_share_by_part: dict,
    models=None,
    unique_id_col: str = "unique_id",
    date_col: str = "ds",
    cutoff_col: str = "cutoff",
    target_col: str = "y",
    partition_col: str = "partition",
    top_k: int = 2,
    verbose: bool = True,
):
    """
    Construit :
    - score_df      : métriques de performance agrégées
    - score_df_exp  : score enrichi avec explicabilité
    - leaderboard_exp : top_k modèles par partition

    Paramètres
    ----------
    bkt_score : pd.DataFrame
        DataFrame de backtest contenant y, partitions, forecasts et intervalles.
    results_perm_mae_by_part : dict
        Dictionnaire {partition: {model: df_perm_mae}}
    results_perm_deviance_by_part : dict
        Dictionnaire {partition: {model: df_perm_dev}}
    results_shap_share_by_part : dict
        Dictionnaire {partition: {model: df_shap}}
    models : list[str] | None
        Liste des modèles à considérer. Ex: ["LR", "RIDGE", "LGBM"]
    unique_id_col, date_col, cutoff_col, target_col, partition_col : str
        Noms de colonnes.
    top_k : int
        Nombre de modèles à garder par partition dans le leaderboard.
    verbose : bool
        Affichage des résumés.

    Retour
    ------
    tuple
        long_sc2, score_df, score_df_exp, leaderboard_exp
    """

    if models is None:
        models = ["LR", "RIDGE", "LGBM"]

    tmp = bkt_score.copy()

    required_cols = [unique_id_col, date_col, target_col, partition_col]
    missing = [c for c in required_cols if c not in tmp.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans bkt_score: {missing}")

    # =========================================================
    # 1) Sécurité bornes (lower <= upper)
    # =========================================================
    for m in models:
        lo = f"{m}-lo-95"
        hi = f"{m}-hi-95"
        if lo in tmp.columns and hi in tmp.columns:
            tmp[[lo, hi]] = np.sort(tmp[[lo, hi]].to_numpy(), axis=1)

    # =========================================================
    # 2) Wide -> Long + features de scoring
    # =========================================================
    rows = []

    base_cols = [c for c in [unique_id_col, date_col, cutoff_col, target_col, partition_col] if c in tmp.columns]

    for m in models:
        lo = f"{m}-lo-95"
        hi = f"{m}-hi-95"

        if m not in tmp.columns:
            continue

        s = tmp[base_cols].copy()
        s["model_label"] = m
        s["model_name"] = m

        s["forecast"] = tmp[m]
        s["lower"] = tmp[lo] if lo in tmp.columns else np.nan
        s["upper"] = tmp[hi] if hi in tmp.columns else np.nan

        s["abs_err"] = (s[target_col] - s["forecast"]).abs()
        s["covered"] = ((s[target_col] >= s["lower"]) & (s[target_col] <= s["upper"])).astype(int)
        s["int_width"] = (s["upper"] - s["lower"]).abs()

        rows.append(s)

    if not rows:
        raise ValueError("Aucune ligne produite à partir des modèles demandés.")

    long_sc = pd.concat(rows, ignore_index=True)
    long_sc[partition_col] = long_sc[partition_col].astype(str)

    # =========================================================
    # 3) Ajouter partition ALL
    # =========================================================
    long_all = long_sc.copy()
    long_all[partition_col] = "ALL"
    long_sc2 = pd.concat([long_sc, long_all], ignore_index=True)

    # =========================================================
    # 4) Score agrégé
    # =========================================================
    score_df = (
        long_sc2
        .groupby([unique_id_col, "model_label", "model_name", partition_col], observed=True)
        .agg(
            mae=("abs_err", "mean"),
            coverage=("covered", "mean"),
            width=("int_width", "mean"),
            n=(target_col, "size"),
        )
        .reset_index()
    )

    # =========================================================
    # 5) Helpers explicabilité
    # =========================================================
    def extract_top1_perm_by_part(res_dict_by_part):
        rows_out = []

        for partition, models_dict in res_dict_by_part.items():
            if not isinstance(models_dict, dict):
                continue

            for model, df in models_dict.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                if "feature" not in df.columns or "ratio_mean" not in df.columns:
                    continue

                top = df.sort_values("ratio_mean", ascending=False).iloc[0]
                rows_out.append({
                    partition_col: str(partition),
                    "model_label": str(model),
                    "perm_top1_feature": str(top["feature"]),
                    "perm_top1_value": float(top["ratio_mean"]),
                })

        return pd.DataFrame(rows_out)

    def extract_top1_shap_by_part(res_dict_by_part):
        rows_out = []

        for partition, models_dict in res_dict_by_part.items():
            if not isinstance(models_dict, dict):
                continue

            for model, df in models_dict.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                if "feature" not in df.columns or "shap_share_mean" not in df.columns:
                    continue

                top = df.sort_values("shap_share_mean", ascending=False).iloc[0]
                rows_out.append({
                    partition_col: str(partition),
                    "model_label": str(model),
                    "shap_top1_feature": str(top["feature"]),
                    "shap_top1_value": float(top["shap_share_mean"]),
                })

        return pd.DataFrame(rows_out)

    perm_mae_top1 = extract_top1_perm_by_part(results_perm_mae_by_part).rename(columns={
        "perm_top1_feature": "perm_mae_top1_feature",
        "perm_top1_value": "perm_mae_top1",
    })

    perm_dev_top1 = extract_top1_perm_by_part(results_perm_deviance_by_part).rename(columns={
        "perm_top1_feature": "perm_dev_top1_feature",
        "perm_top1_value": "perm_dev_top1",
    })

    shap_top1 = extract_top1_shap_by_part(results_shap_share_by_part).rename(columns={
        "shap_top1_feature": "shap_share_top1_feature",
        "shap_top1_value": "shap_share_top1",
    })

    # =========================================================
    # 6) Ajouter ALL pondéré par n
    # =========================================================
    weights_df = (
        score_df[score_df[partition_col] != "ALL"][
            [partition_col, "model_label", "n"]
        ]
        .drop_duplicates()
    )

    def add_all_partition_weighted(df_exp, value_cols):
        if df_exp.empty:
            return df_exp

        dfw = df_exp.merge(weights_df, on=[partition_col, "model_label"], how="left")
        dfw["n"] = dfw["n"].fillna(0).astype(float)

        feature_cols = [c for c in dfw.columns if "feature" in c]

        out = []
        for model, g in dfw[dfw[partition_col] != "ALL"].groupby("model_label"):
            row = {"model_label": model, partition_col: "ALL"}

            if feature_cols:
                g2 = g.sort_values("n", ascending=False)
                for fc in feature_cols:
                    row[fc] = g2.iloc[0][fc]

            for vc in value_cols:
                num = (g[vc].astype(float) * g["n"]).sum()
                den = g["n"].sum()
                row[vc] = float(num / den) if den > 0 else np.nan

            out.append(row)

        df_all = pd.DataFrame(out)
        return pd.concat([df_exp, df_all], ignore_index=True)

    perm_mae_top1 = add_all_partition_weighted(perm_mae_top1, value_cols=["perm_mae_top1"])
    perm_dev_top1 = add_all_partition_weighted(perm_dev_top1, value_cols=["perm_dev_top1"])
    shap_top1 = add_all_partition_weighted(shap_top1, value_cols=["shap_share_top1"])

    # =========================================================
    # 7) Merge explicabilité
    # =========================================================
    score_df_exp = (
        score_df
        .merge(perm_mae_top1, on=[partition_col, "model_label"], how="left")
        .merge(perm_dev_top1, on=[partition_col, "model_label"], how="left")
        .merge(shap_top1, on=[partition_col, "model_label"], how="left")
    )

    # =========================================================
    # 8) Leaderboard enrichi
    # =========================================================
    leaderboard_exp = (
        score_df_exp
        .sort_values(
            by=[partition_col, "mae", "coverage", "width"],
            ascending=[True, True, False, True],
        )
        .groupby(partition_col, as_index=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    if verbose:
        print("\n=== SCORE COMPLET (Performance + Explicabilité) ===")
        print(score_df_exp.sort_values([partition_col, "mae"]).head(50))

        print(f"\n=== TOP {top_k} par partition (incluant ALL) ===")
        print(leaderboard_exp)

    return long_sc2, score_df, score_df_exp, leaderboard_exp

import os
import io
import json
import pickle
import joblib
import shutil
import logging
import warnings
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
import mlflow
import mlflow.data
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)

# Optional flavors
try:
    import mlflow.sklearn
    _HAS_MLFLOW_SKLEARN = True
except Exception:
    _HAS_MLFLOW_SKLEARN = False

try:
    import mlflow.lightgbm
    _HAS_MLFLOW_LIGHTGBM = True
except Exception:
    _HAS_MLFLOW_LIGHTGBM = False

try:
    import mlflow.xgboost
    _HAS_MLFLOW_XGBOOST = True
except Exception:
    _HAS_MLFLOW_XGBOOST = False

try:
    import mlflow.statsmodels
    _HAS_MLFLOW_STATSMODELS = True
except Exception:
    _HAS_MLFLOW_STATSMODELS = False

try:
    import mlflow.pyfunc
    _HAS_MLFLOW_PYFUNC = True
except Exception:
    _HAS_MLFLOW_PYFUNC = False


# =========================================================
# Helpers généraux
# =========================================================
def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Period,)):
        return str(obj)
    return str(obj)


def _safe_write_json(obj, path):
    _ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _safe_write_pickle(obj, path):
    _ensure_dir(Path(path).parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _normalize_ds_in_df(df, ds_col="ds"):
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    out = df.copy()
    if ds_col in out.columns:
        out[ds_col] = pd.to_datetime(out[ds_col], errors="coerce")
        out[ds_col] = out[ds_col].dt.to_period("M").dt.to_timestamp(how="start")
    return out


def _subset_partition(df, partition):
    if df is None or not isinstance(df, pd.DataFrame):
        return None
    if "partition" not in df.columns:
        return None
    out = df[df["partition"].astype(str) == str(partition)].copy()
    return out if len(out) else None


def _find_top1_feature(df_part, score_col):
    if df_part is None or len(df_part) == 0 or score_col not in df_part.columns:
        return None
    tmp = df_part.copy().sort_values(score_col, ascending=False)
    row = tmp.iloc[0]
    feat_col = "feature" if "feature" in tmp.columns else None
    if feat_col is None:
        return None
    return str(row[feat_col])


def _list_artifacts_recursive(client, run_id, path=""):
    out = []
    items = client.list_artifacts(run_id, path)
    for item in items:
        if item.is_dir:
            out.extend(_list_artifacts_recursive(client, run_id, item.path))
        else:
            out.append(item.path)
    return out


def _download_artifact(client, run_id, artifact_path, dst_dir):
    _ensure_dir(dst_dir)
    return client.download_artifacts(run_id, artifact_path, dst_dir)


def _unwrap_estimator_from_mlf(maybe_mlf, preferred_key=None):
    base = maybe_mlf

    if hasattr(base, "_base"):
        try:
            base = base._base
        except Exception:
            pass

    if hasattr(base, "models_") and isinstance(getattr(base, "models_"), dict):
        d = base.models_
        if preferred_key is not None and preferred_key in d:
            return _unwrap_estimator_from_mlf(d[preferred_key], preferred_key=None)
        if len(d):
            return _unwrap_estimator_from_mlf(next(iter(d.values())), preferred_key=None)

    for attr in ["model", "estimator", "_model", "_estimator"]:
        if hasattr(base, attr):
            try:
                inner = getattr(base, attr)
                if inner is not None and inner is not base:
                    return _unwrap_estimator_from_mlf(inner, preferred_key=None)
            except Exception:
                pass

    return base


def _serialize_model_candidate(model_obj, base_dir, stem):
    _ensure_dir(base_dir)

    joblib_path = os.path.join(base_dir, f"{stem}.joblib")
    pkl_path = os.path.join(base_dir, f"{stem}.pkl")

    try:
        joblib.dump(model_obj, joblib_path)
        return joblib_path
    except Exception:
        pass

    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(model_obj, f)
        return pkl_path
    except Exception:
        pass

    return None


def _coerce_explainability_obj(obj, model_label=None):
    if obj is None:
        return None

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()

    elif isinstance(obj, dict):
        if model_label is not None and model_label in obj and isinstance(obj[model_label], pd.DataFrame):
            df = obj[model_label].copy()
        elif len(obj) > 0 and all(isinstance(v, pd.DataFrame) for v in obj.values()):
            parts = []
            for k, v in obj.items():
                tmp = v.copy()
                if "model_label" not in tmp.columns:
                    tmp["model_label"] = str(k)
                parts.append(tmp)
            df = pd.concat(parts, axis=0, ignore_index=True)
        else:
            df = pd.DataFrame([obj])

    elif isinstance(obj, (list, tuple)):
        parts = []
        for x in obj:
            if isinstance(x, pd.DataFrame):
                parts.append(x.copy())
            elif isinstance(x, dict):
                parts.append(pd.DataFrame([x]))
        df = pd.concat(parts, axis=0, ignore_index=True) if len(parts) else None
    else:
        return None

    if df is None or len(df) == 0:
        return None

    if model_label is not None and "model_label" in df.columns:
        df = df[df["model_label"].astype(str) == str(model_label)].copy()

    return df if len(df) else None


# =========================================================
# Helpers MLflow dataset / model
# =========================================================
def _infer_signature_safe(model_obj, X_example):
    try:
        from mlflow.models import infer_signature
        if X_example is None or not isinstance(X_example, pd.DataFrame) or len(X_example) == 0:
            return None
        X_small = X_example.head(min(50, len(X_example))).copy()
        y_pred = model_obj.predict(X_small)
        return infer_signature(X_small, y_pred)
    except Exception:
        return None


def _input_example_safe(X_example):
    try:
        if X_example is None or not isinstance(X_example, pd.DataFrame) or len(X_example) == 0:
            return None
        return X_example.head(min(5, len(X_example))).copy()
    except Exception:
        return None


def _looks_like_lightgbm(model_obj):
    name = type(model_obj).__name__.lower()
    mod = getattr(type(model_obj), "__module__", "").lower()
    return ("lightgbm" in mod) or ("lgbm" in name)


def _looks_like_xgboost(model_obj):
    name = type(model_obj).__name__.lower()
    mod = getattr(type(model_obj), "__module__", "").lower()
    return ("xgboost" in mod) or ("xgb" in name)


def _looks_like_statsmodels(model_obj):
    mod = getattr(type(model_obj), "__module__", "").lower()
    return "statsmodels" in mod


def _looks_like_sklearn(model_obj):
    mod = getattr(type(model_obj), "__module__", "").lower()
    return ("sklearn" in mod) or ("scikit_learn" in mod)


class _GenericPyfuncWrapper(mlflow.pyfunc.PythonModel if _HAS_MLFLOW_PYFUNC else object):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            return self.model.predict(model_input)
        return self.model.predict(pd.DataFrame(model_input))


def _build_feast_dataset_name(feast_feature_name, model_label, partition):
    return f"feast_{feast_feature_name}_{model_label}_{partition}"


def _save_dataset_snapshot_for_mlflow(df_dataset, snapshot_dir, snapshot_name):
    """
    Sauvegarde une copie locale parquet du dataset pour fournir à MLflow
    une source concrète et stable, plus robuste pour l'affichage UI.
    """
    _ensure_dir(snapshot_dir)

    df_to_save = df_dataset.copy()
    if "ds" in df_to_save.columns:
        df_to_save["ds"] = pd.to_datetime(df_to_save["ds"], errors="coerce")

    snapshot_path = os.path.join(snapshot_dir, f"{snapshot_name}.parquet")
    df_to_save.to_parquet(snapshot_path, index=False)

    # URI absolu locale
    snapshot_uri = Path(snapshot_path).resolve().as_uri()
    return snapshot_path, snapshot_uri


def _log_feast_dataset_entity_to_mlflow(
    df_dataset,
    *,
    feast_feature_name,
    model_label,
    partition,
    tmp_dir,
    context="training",
    source_dataset_name=None,
):
    """
    Version robuste pour FEAST:
    - snapshot parquet local
    - source URI concrète
    - name explicite
    - log_input(dataset)
    - log_artifact(snapshot)
    """
    try:
        if df_dataset is None or not isinstance(df_dataset, pd.DataFrame) or len(df_dataset) == 0:
            return False, "Empty or invalid source_dataset_df", None

        ds_df = df_dataset.copy()

        # Normalisation légère
        if "ds" in ds_df.columns:
            ds_df["ds"] = pd.to_datetime(ds_df["ds"], errors="coerce")

        dataset_name = (
            str(source_dataset_name)
            if source_dataset_name is not None and str(source_dataset_name).strip() != ""
            else _build_feast_dataset_name(feast_feature_name, model_label, partition)
        )

        snapshot_dir = os.path.join(tmp_dir, "dataset_snapshot")
        snapshot_path, snapshot_uri = _save_dataset_snapshot_for_mlflow(
            ds_df,
            snapshot_dir=snapshot_dir,
            snapshot_name=dataset_name,
        )

        # Construction dataset MLflow
        dataset = mlflow.data.from_pandas(
            ds_df,
            source=snapshot_uri,
            name=dataset_name,
        )

        # Important pour l'UI Dataset
        mlflow.log_input(dataset, context=context)

        # On log aussi le snapshot comme artifact visible
        mlflow.log_artifact(snapshot_path, artifact_path="dataset_snapshot")

        # Petit résumé lisible
        dataset_meta = {
            "dataset_name": dataset_name,
            "feast_feature_name": feast_feature_name,
            "partition": partition,
            "model_label": model_label,
            "n_rows": int(len(ds_df)),
            "n_cols": int(ds_df.shape[1]),
            "columns": list(map(str, ds_df.columns)),
            "snapshot_uri": snapshot_uri,
            "context": context,
        }
        meta_path = os.path.join(snapshot_dir, f"{dataset_name}_meta.json")
        _safe_write_json(dataset_meta, meta_path)
        mlflow.log_artifact(meta_path, artifact_path="dataset_snapshot")

        # Tags utiles
        mlflow.set_tag("dataset_name", dataset_name)
        mlflow.set_tag("dataset_source_kind", "feast_snapshot")
        mlflow.set_tag("feast_feature_name", str(feast_feature_name))
        mlflow.set_tag("dataset_snapshot_uri", snapshot_uri)

        return True, None, dataset_name

    except Exception as e:
        return False, str(e), None


def _log_model_entity_to_mlflow(model_obj, *, model_name, X_example=None):
    signature = _infer_signature_safe(model_obj, X_example)
    input_example = _input_example_safe(X_example)

    if _HAS_MLFLOW_LIGHTGBM and _looks_like_lightgbm(model_obj):
        try:
            mlflow.lightgbm.log_model(
                lgb_model=model_obj,
                name=model_name,
                signature=signature,
                input_example=input_example,
            )
            return "lightgbm"
        except Exception:
            pass

    if _HAS_MLFLOW_XGBOOST and _looks_like_xgboost(model_obj):
        try:
            mlflow.xgboost.log_model(
                xgb_model=model_obj,
                name=model_name,
                signature=signature,
                input_example=input_example,
            )
            return "xgboost"
        except Exception:
            pass

    if _HAS_MLFLOW_STATSMODELS and _looks_like_statsmodels(model_obj):
        try:
            mlflow.statsmodels.log_model(
                statsmodels_model=model_obj,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example,
            )
            return "statsmodels"
        except Exception:
            pass

    if _HAS_MLFLOW_SKLEARN and _looks_like_sklearn(model_obj):
        try:
            mlflow.sklearn.log_model(
                sk_model=model_obj,
                name=model_name,
                signature=signature,
                input_example=input_example,
            )
            return "sklearn"
        except Exception:
            pass

    if _HAS_MLFLOW_SKLEARN:
        try:
            mlflow.sklearn.log_model(
                sk_model=model_obj,
                name=model_name,
                signature=signature,
                input_example=input_example,
            )
            return "sklearn-fallback"
        except Exception:
            pass

    if _HAS_MLFLOW_PYFUNC:
        try:
            mlflow.pyfunc.log_model(
                artifact_path=model_name,
                python_model=_GenericPyfuncWrapper(model_obj),
                signature=signature,
                input_example=input_example,
            )
            return "pyfunc"
        except Exception:
            return None

    return None


def _load_mlflow_logged_model(run_id, artifact_path):
    uri = f"runs:/{run_id}/{artifact_path}"

    if _HAS_MLFLOW_LIGHTGBM:
        try:
            return mlflow.lightgbm.load_model(uri)
        except Exception:
            pass

    if _HAS_MLFLOW_XGBOOST:
        try:
            return mlflow.xgboost.load_model(uri)
        except Exception:
            pass

    if _HAS_MLFLOW_STATSMODELS:
        try:
            return mlflow.statsmodels.load_model(uri)
        except Exception:
            pass

    if _HAS_MLFLOW_SKLEARN:
        try:
            return mlflow.sklearn.load_model(uri)
        except Exception:
            pass

    if _HAS_MLFLOW_PYFUNC:
        try:
            return mlflow.pyfunc.load_model(uri)
        except Exception:
            pass

    return None


# =========================================================
# Figure logging
# =========================================================
def log_matplotlib_figure_to_mlflow(fig, artifact_file, dpi=200, close_after=False):
    mlflow.log_figure(fig, artifact_file)
    if close_after:
        import matplotlib.pyplot as plt
        plt.close(fig)


def save_and_log_matplotlib_figure(fig, artifact_dir, filename, dpi=200, close_after=False):
    _ensure_dir(artifact_dir)
    png_path = os.path.join(artifact_dir, filename)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    mlflow.log_artifact(png_path, artifact_path="plots")

    if close_after:
        import matplotlib.pyplot as plt
        plt.close(fig)

    return png_path


# =========================================================
# Export principal vers MLflow
# =========================================================
def log_experiment_runs_to_mlflow(
    *,
    score_df_exp,
    leaderboard_exp,
    bkt_score,
    meta_models,
    tracking_uri,
    experiment_name,
    feast_feature_name,
    ts_features,
    results_perm_mae_by_part=None,
    results_perm_deviance_by_part=None,
    results_shap_share_by_part=None,
    fitted_models=None,
    train_fit_dates=None,
    X_by_partition=None,
    features_by_partition=None,
    extra_figures=None,
    run_tags=None,
    run_name_fn=None,
    tmp_root="mlflow_tmp",
    log_mlflow_dataset=True,
    log_mlflow_model=True,
    source_dataset_df=None,
    source_dataset_name=None,     # gardé pour compatibilité
    source_dataset_source=None,   # gardé pour compatibilité
    target_col=None,
    n_lags=None,
    exog_cols=None,
):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    df_log = score_df_exp.copy()

    required_cols = {"model_label", "partition", "mae"}
    missing = required_cols - set(df_log.columns)
    if missing:
        raise ValueError(f"score_df_exp missing required columns: {missing}")

    if "model_name" not in df_log.columns:
        df_log["model_name"] = df_log["model_label"].astype(str)

    df_log["model_label"] = df_log["model_label"].astype(str)
    df_log["partition"] = df_log["partition"].astype(str)
    df_log["model_name"] = df_log["model_name"].astype(str)

    for _, row in df_log.iterrows():
        model_label = str(row["model_label"])
        partition = str(row["partition"])
        model_name = str(row["model_name"])

        run_name = str(run_name_fn(row)) if callable(run_name_fn) else f"{model_label} | {partition}"

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("model_label", model_label)
                mlflow.set_tag("partition", partition)
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("run_name_custom", run_name)
                mlflow.set_tag("feast_feature_name", str(feast_feature_name))

                if run_tags:
                    for k, v in run_tags.items():
                        mlflow.set_tag(str(k), str(v))

                mlflow.log_param("model_label", model_label)
                mlflow.log_param("partition", partition)
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("run_name", run_name)

                if ts_features is not None:
                    if isinstance(ts_features, (list, tuple)):
                        mlflow.log_param("n_ts_features", len(ts_features))
                        mlflow.log_param("ts_features", ", ".join(map(str, ts_features)))
                    else:
                        mlflow.log_param("ts_features", str(ts_features))

                if target_col is not None:
                    mlflow.log_param("target_col", str(target_col))

                if n_lags is not None:
                    try:
                        mlflow.log_param("n_lags", int(n_lags))
                    except Exception:
                        mlflow.log_param("n_lags", str(n_lags))

                if exog_cols is not None:
                    try:
                        mlflow.log_param("n_exog_cols", len(exog_cols))
                    except Exception:
                        pass
                    try:
                        mlflow.log_param("exog_cols", ", ".join(map(str, exog_cols)))
                    except Exception:
                        mlflow.log_param("exog_cols", str(exog_cols))

                for col, val in row.items():
                    if col in {"model_label", "partition", "model_name"}:
                        continue
                    if isinstance(val, (int, float, np.integer, np.floating)) and pd.notna(val):
                        try:
                            mlflow.log_metric(col, float(val))
                        except Exception:
                            pass

                tmp_dir = Path(tmp_root) / model_label / partition
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                tmp_dir.mkdir(parents=True, exist_ok=True)

                row_df = pd.DataFrame([row])
                row_csv = tmp_dir / f"score_row_{model_label}_{partition}.csv"
                row_df.to_csv(row_csv, index=False)
                mlflow.log_artifact(str(row_csv), artifact_path="score")

                lb_part = _subset_partition(leaderboard_exp, partition)
                if lb_part is not None:
                    p = tmp_dir / f"leaderboard_{partition}.csv"
                    lb_part.to_csv(p, index=False)
                    mlflow.log_artifact(str(p), artifact_path="leaderboard")

                if results_perm_mae_by_part is not None and partition in results_perm_mae_by_part:
                    dfp = _coerce_explainability_obj(results_perm_mae_by_part[partition], model_label=model_label)
                    if dfp is not None and len(dfp):
                        p = tmp_dir / f"results_perm_mae_{model_label}_{partition}.csv"
                        dfp.to_csv(p, index=False)
                        mlflow.log_artifact(str(p), artifact_path="explainability")
                        score_col = "perm_mae_ratio" if "perm_mae_ratio" in dfp.columns else dfp.columns[-1]
                        top1 = _find_top1_feature(dfp, score_col)
                        if top1 is not None:
                            mlflow.set_tag("perm_mae_top1", top1)

                if results_perm_deviance_by_part is not None and partition in results_perm_deviance_by_part:
                    dfp = _coerce_explainability_obj(results_perm_deviance_by_part[partition], model_label=model_label)
                    if dfp is not None and len(dfp):
                        p = tmp_dir / f"results_perm_dev_{model_label}_{partition}.csv"
                        dfp.to_csv(p, index=False)
                        mlflow.log_artifact(str(p), artifact_path="explainability")
                        score_col = "perm_deviance_ratio" if "perm_deviance_ratio" in dfp.columns else dfp.columns[-1]
                        top1 = _find_top1_feature(dfp, score_col)
                        if top1 is not None:
                            mlflow.set_tag("perm_dev_top1", top1)

                if results_shap_share_by_part is not None and partition in results_shap_share_by_part:
                    dfp = _coerce_explainability_obj(results_shap_share_by_part[partition], model_label=model_label)
                    if dfp is not None and len(dfp):
                        p = tmp_dir / f"results_shap_share_{model_label}_{partition}.csv"
                        dfp.to_csv(p, index=False)
                        mlflow.log_artifact(str(p), artifact_path="explainability")
                        score_col = "shap_share" if "shap_share" in dfp.columns else dfp.columns[-1]
                        top1 = _find_top1_feature(dfp, score_col)
                        if top1 is not None:
                            mlflow.set_tag("shap_top1", top1)

                bkt_part = None
                if isinstance(bkt_score, dict):
                    bkt_part = bkt_score.get((model_label, partition), None)
                elif isinstance(bkt_score, pd.DataFrame):
                    tmp = bkt_score.copy()
                    if "model_label" in tmp.columns and "partition" in tmp.columns:
                        m = (
                            tmp["model_label"].astype(str).eq(model_label)
                            & tmp["partition"].astype(str).eq(partition)
                        )
                        bkt_part = tmp.loc[m].copy()
                    elif "model_label" in tmp.columns:
                        m = tmp["model_label"].astype(str).eq(model_label)
                        bkt_part = tmp.loc[m].copy()

                if isinstance(bkt_part, pd.DataFrame) and len(bkt_part):
                    bkt_part = _normalize_ds_in_df(bkt_part, ds_col="ds")
                    p = tmp_dir / f"bkt_{model_label}_{partition}.parquet"
                    bkt_part.to_parquet(p, index=False)
                    mlflow.log_artifact(str(p), artifact_path="backtest")

                meta_obj = None
                if isinstance(meta_models, dict):
                    meta_obj = meta_models.get((model_label, partition), meta_models.get(model_label, None))
                    if meta_obj is None:
                        meta_obj = meta_models.get("metas", {}).get(model_label, None)

                if meta_obj is not None:
                    p_json = tmp_dir / f"meta_{model_label}_{partition}.json"
                    _safe_write_json(meta_obj, p_json)
                    mlflow.log_artifact(str(p_json), artifact_path="meta")

                    p_pkl = tmp_dir / f"meta_{model_label}_{partition}.pkl"
                    _safe_write_pickle(meta_obj, p_pkl)
                    mlflow.log_artifact(str(p_pkl), artifact_path="meta")

                # =====================================================
                # DATASET FEAST visible dans MLflow UI
                # =====================================================
                dataset_ok = False
                dataset_err = None
                dataset_name_logged = None

                if log_mlflow_dataset and isinstance(source_dataset_df, pd.DataFrame):
                    dataset_ok, dataset_err, dataset_name_logged = _log_feast_dataset_entity_to_mlflow(
                        source_dataset_df,
                        feast_feature_name=feast_feature_name,
                        model_label=model_label,
                        partition=partition,
                        tmp_dir=str(tmp_dir),
                        context="training",
                        source_dataset_name=source_dataset_name,
                    )

                mlflow.set_tag("dataset_logged", str(bool(dataset_ok)).lower())
                if dataset_name_logged is not None:
                    mlflow.set_tag("dataset_name_logged", dataset_name_logged)

                if dataset_err is not None:
                    mlflow.set_tag("dataset_log_error", str(dataset_err)[:500])
                    p = tmp_dir / "dataset_log_error.txt"
                    with open(p, "w", encoding="utf-8") as f:
                        f.write(str(dataset_err))
                    mlflow.log_artifact(str(p), artifact_path="logs")

                Xp2 = None
                if isinstance(X_by_partition, dict):
                    Xp = X_by_partition.get((model_label, partition), None)
                    if Xp is None:
                        Xp = X_by_partition.get((model_label, "ALL"), None)
                    if isinstance(Xp, pd.DataFrame):
                        Xp2 = Xp.copy()
                        if "ds" in Xp2.columns:
                            Xp2["ds"] = pd.to_datetime(Xp2["ds"], errors="coerce")
                        p = tmp_dir / f"X_{model_label}_{partition}.parquet"
                        Xp2.to_parquet(p, index=False)
                        mlflow.log_artifact(str(p), artifact_path="functional")

                if isinstance(features_by_partition, dict):
                    feats = features_by_partition.get((model_label, partition), None)
                    if feats is None:
                        feats = features_by_partition.get((model_label, "ALL"), None)
                    if feats is not None:
                        p = tmp_dir / f"features_{model_label}_{partition}.json"
                        _safe_write_json(list(map(str, feats)), p)
                        mlflow.log_artifact(str(p), artifact_path="functional")

                model_obj = None
                if isinstance(fitted_models, dict):
                    model_obj = fitted_models.get((model_label, partition), None)
                    if model_obj is None:
                        model_obj = fitted_models.get((model_label, "ALL"), None)
                    if model_obj is None:
                        model_obj = fitted_models.get(model_label, None)

                if model_obj is not None:
                    saved_path = _serialize_model_candidate(
                        model_obj=model_obj,
                        base_dir=tmp_dir / "models",
                        stem=f"model_{model_label}_{partition}",
                    )
                    if saved_path is not None:
                        mlflow.log_artifact(str(saved_path), artifact_path="models")

                    try:
                        est = _unwrap_estimator_from_mlf(model_obj, preferred_key=model_label)
                        saved_est = _serialize_model_candidate(
                            model_obj=est,
                            base_dir=tmp_dir / "models",
                            stem=f"estimator_{model_label}_{partition}",
                        )
                        if saved_est is not None:
                            mlflow.log_artifact(str(saved_est), artifact_path="models")

                        if log_mlflow_model:
                            X_example = Xp2 if isinstance(Xp2, pd.DataFrame) else source_dataset_df
                            flavor_used = _log_model_entity_to_mlflow(
                                est,
                                model_name=f"model_entity_{model_label}_{partition}",
                                X_example=X_example,
                            )
                            if flavor_used is not None:
                                mlflow.set_tag("mlflow_model_flavor", flavor_used)
                    except Exception as e:
                        p = tmp_dir / "model_log_error.txt"
                        with open(p, "w", encoding="utf-8") as f:
                            f.write(str(e))
                        mlflow.log_artifact(str(p), artifact_path="logs")

                if isinstance(train_fit_dates, dict):
                    tfd = train_fit_dates.get((model_label, partition), None)
                    if tfd is None:
                        tfd = train_fit_dates.get(model_label, None)
                    if tfd is not None:
                        p = tmp_dir / f"train_fit_dates_{model_label}_{partition}.json"
                        _safe_write_json(tfd, p)
                        mlflow.log_artifact(str(p), artifact_path="meta")

                if isinstance(extra_figures, dict):
                    figs = extra_figures.get((model_label, partition), None)
                    if isinstance(figs, dict):
                        for fname, fig in figs.items():
                            try:
                                mlflow.log_figure(fig, f"plots/{fname}")
                            except Exception:
                                local_fig = tmp_dir / fname
                                fig.savefig(local_fig, dpi=200, bbox_inches="tight")
                                mlflow.log_artifact(str(local_fig), artifact_path="plots")

                out_txt = stdout_buf.getvalue().strip()
                err_txt = stderr_buf.getvalue().strip()

                if out_txt:
                    p = tmp_dir / "stdout.txt"
                    with open(p, "w", encoding="utf-8") as f:
                        f.write(out_txt)
                    mlflow.log_artifact(str(p), artifact_path="logs")

                if err_txt:
                    p = tmp_dir / "stderr.txt"
                    with open(p, "w", encoding="utf-8") as f:
                        f.write(err_txt)
                    mlflow.log_artifact(str(p), artifact_path="logs")


# =========================================================
# Import MLflow complet
# =========================================================
def import_mlflow_experiment(
    *,
    tracking_uri,
    experiment_name,
    dl_dir="mlflow_import",
    prefer_latest_only=True,
    try_load_logged_mlflow_models=True,
):
    _ensure_dir(dl_dir)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment introuvable: {experiment_name}")

    runs = mlflow.search_runs([exp.experiment_id], output_format="pandas")
    if runs.empty:
        raise ValueError("Aucun run trouvé.")

    runs = runs.sort_values("start_time", ascending=False).reset_index(drop=True)

    def _get_col(df, cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    col_model = _get_col(runs, ["params.model_label", "tags.model_label"])
    col_part = _get_col(runs, ["params.partition", "tags.partition"])
    col_name = _get_col(runs, ["params.model_name", "tags.model_name"])

    if col_model is None or col_part is None:
        raise ValueError("Impossible de trouver model_label / partition dans les runs MLflow.")

    runs["model_label"] = runs[col_model].astype(str)
    runs["partition"] = runs[col_part].astype(str)
    runs["model_name"] = runs[col_name].astype(str) if col_name is not None else runs["model_label"].astype(str)

    if prefer_latest_only:
        runs = (
            runs.sort_values("start_time", ascending=False)
            .drop_duplicates(subset=["model_label", "partition"], keep="first")
            .reset_index(drop=True)
        )

    score_rows = []
    leaderboard_rows = []

    results_perm_mae_by_part = {}
    results_perm_deviance_by_part = {}
    results_shap_share_by_part = {}

    bkt_by_run = {}
    X_by_run = {}
    features_by_run = {}
    models_mlflow = {}
    meta_by_run = {}
    train_fit_dates_by_run = {}

    def _safe_read_csv(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    def _safe_read_parquet(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def _safe_read_json(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _safe_load_model(path):
        try:
            if str(path).lower().endswith(".joblib"):
                return joblib.load(path)
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _append_expl_store(store, partition, df, model_label):
        if df is None or len(df) == 0:
            return
        tmp = df.copy()
        if "model_label" not in tmp.columns:
            tmp["model_label"] = str(model_label)
        store.setdefault(partition, [])
        store[partition].append(tmp)

    for _, rr in runs.iterrows():
        run_id = rr["run_id"]
        model_label = str(rr["model_label"])
        partition = str(rr["partition"])
        key = (model_label, partition)

        run_dir = os.path.join(dl_dir, model_label, partition)
        _ensure_dir(run_dir)

        try:
            artifacts = _list_artifacts_recursive(client, run_id, path="")
        except Exception:
            artifacts = []

        for ap in artifacts:
            apl = ap.lower()

            if "score/" in apl and apl.endswith(".csv"):
                local = _download_artifact(client, run_id, ap, run_dir)
                df = _safe_read_csv(local)
                if df is not None and len(df):
                    if "model_label" not in df.columns:
                        df["model_label"] = model_label
                    if "partition" not in df.columns:
                        df["partition"] = partition
                    score_rows.append(df)

            elif "leaderboard/" in apl and apl.endswith(".csv"):
                local = _download_artifact(client, run_id, ap, run_dir)
                df = _safe_read_csv(local)
                if df is not None and len(df):
                    if "partition" not in df.columns:
                        df["partition"] = partition
                    leaderboard_rows.append(df)

            elif "explainability/" in apl and apl.endswith(".csv"):
                local = _download_artifact(client, run_id, ap, run_dir)
                df = _safe_read_csv(local)
                if df is None or len(df) == 0:
                    continue

                if "perm_mae" in apl:
                    _append_expl_store(results_perm_mae_by_part, partition, df, model_label)
                elif "perm_dev" in apl or "perm_deviance" in apl:
                    _append_expl_store(results_perm_deviance_by_part, partition, df, model_label)
                elif "shap" in apl:
                    _append_expl_store(results_shap_share_by_part, partition, df, model_label)

            elif "backtest/" in apl and apl.endswith(".parquet"):
                local = _download_artifact(client, run_id, ap, run_dir)
                df = _safe_read_parquet(local)
                if df is not None:
                    bkt_by_run[key] = df

            elif ("functional/" in apl or "/x_" in apl or apl.startswith("x_")) and apl.endswith(".parquet"):
                local = _download_artifact(client, run_id, ap, run_dir)
                df = _safe_read_parquet(local)
                if df is not None:
                    X_by_run[key] = df

            elif ("functional/" in apl or "features_" in apl) and apl.endswith(".json"):
                local = _download_artifact(client, run_id, ap, run_dir)
                obj = _safe_read_json(local)
                if obj is not None:
                    features_by_run[key] = obj

            elif "meta/" in apl and "train_fit_dates_" in apl and apl.endswith(".json"):
                local = _download_artifact(client, run_id, ap, run_dir)
                obj = _safe_read_json(local)
                if obj is not None:
                    train_fit_dates_by_run[key] = obj

            elif "meta/" in apl and "meta_" in apl and apl.endswith(".json"):
                local = _download_artifact(client, run_id, ap, run_dir)
                obj = _safe_read_json(local)
                if obj is not None:
                    meta_by_run[key] = obj

        if try_load_logged_mlflow_models:
            candidate_model_entities = [
                f"model_entity_{model_label}_{partition}",
                "model",
            ]
            for art in candidate_model_entities:
                loaded = _load_mlflow_logged_model(run_id, art)
                if loaded is not None:
                    models_mlflow[key] = loaded
                    break

        if key not in models_mlflow:
            model_candidates = []
            for ap in artifacts:
                apl = ap.lower()
                if "models/" in apl and (apl.endswith(".joblib") or apl.endswith(".pkl") or apl.endswith(".pickle")):
                    model_candidates.append(ap)

            def _rank_model_path(x):
                xl = x.lower()
                score = 100
                if f"estimator_{model_label.lower()}_{partition.lower()}.joblib" in xl:
                    score = 0
                elif f"model_{model_label.lower()}_{partition.lower()}.joblib" in xl:
                    score = 1
                elif xl.endswith(".joblib"):
                    score = 2
                elif xl.endswith(".pkl") or xl.endswith(".pickle"):
                    score = 3
                return score, len(x)

            model_candidates = sorted(model_candidates, key=_rank_model_path)

            for ap in model_candidates:
                local = _download_artifact(client, run_id, ap, run_dir)
                loaded_model = _safe_load_model(local)
                if loaded_model is not None:
                    models_mlflow[key] = loaded_model
                    break

    score_df_exp = (
        pd.concat(score_rows, axis=0, ignore_index=True).drop_duplicates().reset_index(drop=True)
        if len(score_rows) else None
    )

    leaderboard_exp = (
        pd.concat(leaderboard_rows, axis=0, ignore_index=True).drop_duplicates().reset_index(drop=True)
        if len(leaderboard_rows) else None
    )

    for part, lst in list(results_perm_mae_by_part.items()):
        results_perm_mae_by_part[part] = (
            pd.concat(lst, axis=0, ignore_index=True).drop_duplicates().reset_index(drop=True)
            if len(lst) else pd.DataFrame()
        )

    for part, lst in list(results_perm_deviance_by_part.items()):
        results_perm_deviance_by_part[part] = (
            pd.concat(lst, axis=0, ignore_index=True).drop_duplicates().reset_index(drop=True)
            if len(lst) else pd.DataFrame()
        )

    for part, lst in list(results_shap_share_by_part.items()):
        results_shap_share_by_part[part] = (
            pd.concat(lst, axis=0, ignore_index=True).drop_duplicates().reset_index(drop=True)
            if len(lst) else pd.DataFrame()
        )

    return {
        "runs": runs,
        "score_df_exp": score_df_exp,
        "leaderboard_exp": leaderboard_exp,
        "results_perm_mae_by_part": results_perm_mae_by_part,
        "results_perm_deviance_by_part": results_perm_deviance_by_part,
        "results_shap_share_by_part": results_shap_share_by_part,
        "bkt_by_run": bkt_by_run,
        "X_by_run": X_by_run,
        "features_by_run": features_by_run,
        "models_mlflow": models_mlflow,
        "meta_by_run": meta_by_run,
        "train_fit_dates_by_run": train_fit_dates_by_run,
    }


def build_models_and_X_for_functional_plots(
    *,
    models_mlflow,
    X_by_run,
    features_by_run=None,
    preferred_partition="ALL",
    labels_map=None,
):
    if labels_map is None:
        labels_map = {}

    all_keys = sorted(set(models_mlflow.keys()) | set(X_by_run.keys()))
    all_model_labels = sorted(set(k[0] for k in all_keys))

    models_dict = {}
    X_dict_out = {}
    features_dict = {}

    for model_label in all_model_labels:
        chosen_key = None

        if (model_label, preferred_partition) in models_mlflow and (model_label, preferred_partition) in X_by_run:
            chosen_key = (model_label, preferred_partition)
        else:
            common = [k for k in all_keys if k[0] == model_label and k in models_mlflow and k in X_by_run]
            if len(common):
                common = sorted(common, key=lambda x: (x[1] != preferred_partition, x[1]))
                chosen_key = common[0]

        if chosen_key is None:
            continue

        display_name = labels_map.get(model_label, model_label)
        model_obj = models_mlflow[chosen_key]
        prep = None

        models_dict[display_name] = (model_obj, prep)
        X_dict_out[display_name] = X_by_run[chosen_key]

        if isinstance(features_by_run, dict) and chosen_key in features_by_run:
            features_dict[display_name] = features_by_run[chosen_key]

    return models_dict, X_dict_out, features_dict
import io
import json
import pickle
import joblib
import shutil
import logging
import warnings
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
import mlflow
import mlflow.data
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)

# Optional flavors
try:
    import mlflow.sklearn
    _HAS_MLFLOW_SKLEARN = True
except Exception:
    _HAS_MLFLOW_SKLEARN = False

try:
    import mlflow.lightgbm
    _HAS_MLFLOW_LIGHTGBM = True
except Exception:
    _HAS_MLFLOW_LIGHTGBM = False

try:
    import mlflow.xgboost
    _HAS_MLFLOW_XGBOOST = True
except Exception:
    _HAS_MLFLOW_XGBOOST = False

try:
    import mlflow.statsmodels
    _HAS_MLFLOW_STATSMODELS = True
except Exception:
    _HAS_MLFLOW_STATSMODELS = False

try:
    import mlflow.pyfunc
    _HAS_MLFLOW_PYFUNC = True
except Exception:
    _HAS_MLFLOW_PYFUNC = False


# =========================================================
# Helpers généraux
# =========================================================
def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Period,)):
        return str(obj)
    return str(obj)


def _safe_write_json(obj, path):
    _ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _safe_write_pickle(obj, path):
    _ensure_dir(Path(path).parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _normalize_ds_in_df(df, ds_col="ds"):
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    out = df.copy()
    if ds_col in out.columns:
        out[ds_col] = pd.to_datetime(out[ds_col], errors="coerce")
        out[ds_col] = out[ds_col].dt.to_period("M").dt.to_timestamp(how="start")
    return out


def _subset_partition(df, partition):
    if df is None or not isinstance(df, pd.DataFrame):
        return None
    if "partition" not in df.columns:
        return None
    out = df[df["partition"].astype(str) == str(partition)].copy()
    return out if len(out) else None


def _find_top1_feature(df_part, score_col):
    if df_part is None or len(df_part) == 0 or score_col not in df_part.columns:
        return None
    tmp = df_part.copy().sort_values(score_col, ascending=False)
    row = tmp.iloc[0]
    feat_col = "feature" if "feature" in tmp.columns else None
    if feat_col is None:
        return None
    return str(row[feat_col])


def _list_artifacts_recursive(client, run_id, path=""):
    out = []
    items = client.list_artifacts(run_id, path)
    for item in items:
        if item.is_dir:
            out.extend(_list_artifacts_recursive(client, run_id, item.path))
        else:
            out.append(item.path)
    return out


def _download_artifact(client, run_id, artifact_path, dst_dir):
    _ensure_dir(dst_dir)
    return client.download_artifacts(run_id, artifact_path, dst_dir)


def _unwrap_estimator_from_mlf(maybe_mlf, preferred_key=None):
    base = maybe_mlf

    if hasattr(base, "_base"):
        try:
            base = base._base
        except Exception:
            pass

    if hasattr(base, "models_") and isinstance(getattr(base, "models_"), dict):
        d = base.models_
        if preferred_key is not None and preferred_key in d:
            return _unwrap_estimator_from_mlf(d[preferred_key], preferred_key=None)
        if len(d):
            return _unwrap_estimator_from_mlf(next(iter(d.values())), preferred_key=None)

    for attr in ["model", "estimator", "_model", "_estimator"]:
        if hasattr(base, attr):
            try:
                inner = getattr(base, attr)
                if inner is not None and inner is not base:
                    return _unwrap_estimator_from_mlf(inner, preferred_key=None)
            except Exception:
                pass

    return base


def _serialize_model_candidate(model_obj, base_dir, stem):
    _ensure_dir(base_dir)

    joblib_path = os.path.join(base_dir, f"{stem}.joblib")
    pkl_path = os.path.join(base_dir, f"{stem}.pkl")

    try:
        joblib.dump(model_obj, joblib_path)
        return joblib_path
    except Exception:
        pass

    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(model_obj, f)
        return pkl_path
    except Exception:
        pass

    return None


def _coerce_explainability_obj(obj, model_label=None):
    if obj is None:
        return None

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()

    elif isinstance(obj, dict):
        if model_label is not None and model_label in obj and isinstance(obj[model_label], pd.DataFrame):
            df = obj[model_label].copy()
        elif len(obj) > 0 and all(isinstance(v, pd.DataFrame) for v in obj.values()):
            parts = []
            for k, v in obj.items():
                tmp = v.copy()
                if "model_label" not in tmp.columns:
                    tmp["model_label"] = str(k)
                parts.append(tmp)
            df = pd.concat(parts, axis=0, ignore_index=True)
        else:
            df = pd.DataFrame([obj])

    elif isinstance(obj, (list, tuple)):
        parts = []
        for x in obj:
            if isinstance(x, pd.DataFrame):
                parts.append(x.copy())
            elif isinstance(x, dict):
                parts.append(pd.DataFrame([x]))
        df = pd.concat(parts, axis=0, ignore_index=True) if len(parts) else None
    else:
        return None

    if df is None or len(df) == 0:
        return None

    if model_label is not None and "model_label" in df.columns:
        df = df[df["model_label"].astype(str) == str(model_label)].copy()

    return df if len(df) else None


# =========================================================
# Helpers MLflow dataset / model
# =========================================================
def _infer_signature_safe(model_obj, X_example):
    try:
        from mlflow.models import infer_signature
        if X_example is None or not isinstance(X_example, pd.DataFrame) or len(X_example) == 0:
            return None
        X_small = X_example.head(min(50, len(X_example))).copy()
        y_pred = model_obj.predict(X_small)
        return infer_signature(X_small, y_pred)
    except Exception:
        return None


def _input_example_safe(X_example):
    try:
        if X_example is None or not isinstance(X_example, pd.DataFrame) or len(X_example) == 0:
            return None
        return X_example.head(min(5, len(X_example))).copy()
    except Exception:
        return None


def _looks_like_lightgbm(model_obj):
    name = type(model_obj).__name__.lower()
    mod = getattr(type(model_obj), "__module__", "").lower()
    return ("lightgbm" in mod) or ("lgbm" in name)


def _looks_like_xgboost(model_obj):
    name = type(model_obj).__name__.lower()
    mod = getattr(type(model_obj), "__module__", "").lower()
    return ("xgboost" in mod) or ("xgb" in name)


def _looks_like_statsmodels(model_obj):
    mod = getattr(type(model_obj), "__module__", "").lower()
    return "statsmodels" in mod


def _looks_like_sklearn(model_obj):
    mod = getattr(type(model_obj), "__module__", "").lower()
    return ("sklearn" in mod) or ("scikit_learn" in mod)


class _GenericPyfuncWrapper(mlflow.pyfunc.PythonModel if _HAS_MLFLOW_PYFUNC else object):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            return self.model.predict(model_input)
        return self.model.predict(pd.DataFrame(model_input))


def _build_feast_dataset_name(feast_feature_name, model_label, partition):
    return f"feast_{feast_feature_name}_{model_label}_{partition}"


def _save_dataset_snapshot_for_mlflow(df_dataset, snapshot_dir, snapshot_name):
    """
    Sauvegarde une copie locale parquet du dataset pour fournir à MLflow
    une source concrète et stable, plus robuste pour l'affichage UI.
    """
    _ensure_dir(snapshot_dir)

    df_to_save = df_dataset.copy()
    if "ds" in df_to_save.columns:
        df_to_save["ds"] = pd.to_datetime(df_to_save["ds"], errors="coerce")

    snapshot_path = os.path.join(snapshot_dir, f"{snapshot_name}.parquet")
    df_to_save.to_parquet(snapshot_path, index=False)

    # URI absolu locale
    snapshot_uri = Path(snapshot_path).resolve().as_uri()
    return snapshot_path, snapshot_uri


def _log_feast_dataset_entity_to_mlflow(
    df_dataset,
    *,
    feast_feature_name,
    model_label,
    partition,
    tmp_dir,
    context="training",
    source_dataset_name=None,
):
    """
    Version robuste pour FEAST:
    - snapshot parquet local
    - source URI concrète
    - name explicite
    - log_input(dataset)
    - log_artifact(snapshot)
    """
    try:
        if df_dataset is None or not isinstance(df_dataset, pd.DataFrame) or len(df_dataset) == 0:
            return False, "Empty or invalid source_dataset_df", None

        ds_df = df_dataset.copy()

        # Normalisation légère
        if "ds" in ds_df.columns:
            ds_df["ds"] = pd.to_datetime(ds_df["ds"], errors="coerce")

        dataset_name = (
            str(source_dataset_name)
            if source_dataset_name is not None and str(source_dataset_name).strip() != ""
            else _build_feast_dataset_name(feast_feature_name, model_label, partition)
        )

        snapshot_dir = os.path.join(tmp_dir, "dataset_snapshot")
        snapshot_path, snapshot_uri = _save_dataset_snapshot_for_mlflow(
            ds_df,
            snapshot_dir=snapshot_dir,
            snapshot_name=dataset_name,
        )

        # Construction dataset MLflow
        dataset = mlflow.data.from_pandas(
            ds_df,
            source=snapshot_uri,
            name=dataset_name,
        )

        # Important pour l'UI Dataset
        mlflow.log_input(dataset, context=context)

        # On log aussi le snapshot comme artifact visible
        mlflow.log_artifact(snapshot_path, artifact_path="dataset_snapshot")

        # Petit résumé lisible
        dataset_meta = {
            "dataset_name": dataset_name,
            "feast_feature_name": feast_feature_name,
            "partition": partition,
            "model_label": model_label,
            "n_rows": int(len(ds_df)),
            "n_cols": int(ds_df.shape[1]),
            "columns": list(map(str, ds_df.columns)),
            "snapshot_uri": snapshot_uri,
            "context": context,
        }
        meta_path = os.path.join(snapshot_dir, f"{dataset_name}_meta.json")
        _safe_write_json(dataset_meta, meta_path)
        mlflow.log_artifact(meta_path, artifact_path="dataset_snapshot")

        # Tags utiles
        mlflow.set_tag("dataset_name", dataset_name)
        mlflow.set_tag("dataset_source_kind", "feast_snapshot")
        mlflow.set_tag("feast_feature_name", str(feast_feature_name))
        mlflow.set_tag("dataset_snapshot_uri", snapshot_uri)

        return True, None, dataset_name

    except Exception as e:
        return False, str(e), None


def _log_model_entity_to_mlflow(model_obj, *, model_name, X_example=None):
    signature = _infer_signature_safe(model_obj, X_example)
    input_example = _input_example_safe(X_example)

    if _HAS_MLFLOW_LIGHTGBM and _looks_like_lightgbm(model_obj):
        try:
            mlflow.lightgbm.log_model(
                lgb_model=model_obj,
                name=model_name,
                signature=signature,
                input_example=input_example,
            )
            return "lightgbm"
        except Exception:
            pass

    if _HAS_MLFLOW_XGBOOST and _looks_like_xgboost(model_obj):
        try:
            mlflow.xgboost.log_model(
                xgb_model=model_obj,
                name=model_name,
                signature=signature,
                input_example=input_example,
            )
            return "xgboost"
        except Exception:
            pass

    if _HAS_MLFLOW_STATSMODELS and _looks_like_statsmodels(model_obj):
        try:
            mlflow.statsmodels.log_model(
                statsmodels_model=model_obj,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example,
            )
            return "statsmodels"
        except Exception:
            pass

    if _HAS_MLFLOW_SKLEARN and _looks_like_sklearn(model_obj):
        try:
            mlflow.sklearn.log_model(
                sk_model=model_obj,
                name=model_name,
                signature=signature,
                input_example=input_example,
            )
            return "sklearn"
        except Exception:
            pass

    if _HAS_MLFLOW_SKLEARN:
        try:
            mlflow.sklearn.log_model(
                sk_model=model_obj,
                name=model_name,
                signature=signature,
                input_example=input_example,
            )
            return "sklearn-fallback"
        except Exception:
            pass

    if _HAS_MLFLOW_PYFUNC:
        try:
            mlflow.pyfunc.log_model(
                artifact_path=model_name,
                python_model=_GenericPyfuncWrapper(model_obj),
                signature=signature,
                input_example=input_example,
            )
            return "pyfunc"
        except Exception:
            return None

    return None


def _load_mlflow_logged_model(run_id, artifact_path):
    uri = f"runs:/{run_id}/{artifact_path}"

    if _HAS_MLFLOW_LIGHTGBM:
        try:
            return mlflow.lightgbm.load_model(uri)
        except Exception:
            pass

    if _HAS_MLFLOW_XGBOOST:
        try:
            return mlflow.xgboost.load_model(uri)
        except Exception:
            pass

    if _HAS_MLFLOW_STATSMODELS:
        try:
            return mlflow.statsmodels.load_model(uri)
        except Exception:
            pass

    if _HAS_MLFLOW_SKLEARN:
        try:
            return mlflow.sklearn.load_model(uri)
        except Exception:
            pass

    if _HAS_MLFLOW_PYFUNC:
        try:
            return mlflow.pyfunc.load_model(uri)
        except Exception:
            pass

    return None


# =========================================================
# Figure logging
# =========================================================
def log_matplotlib_figure_to_mlflow(fig, artifact_file, dpi=200, close_after=False):
    mlflow.log_figure(fig, artifact_file)
    if close_after:
        import matplotlib.pyplot as plt
        plt.close(fig)


def save_and_log_matplotlib_figure(fig, artifact_dir, filename, dpi=200, close_after=False):
    _ensure_dir(artifact_dir)
    png_path = os.path.join(artifact_dir, filename)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    mlflow.log_artifact(png_path, artifact_path="plots")

    if close_after:
        import matplotlib.pyplot as plt
        plt.close(fig)

    return png_path


# =========================================================
# Export principal vers MLflow
# =========================================================
def log_experiment_runs_to_mlflow(
    *,
    score_df_exp,
    leaderboard_exp,
    bkt_score,
    meta_models,
    tracking_uri,
    experiment_name,
    feast_feature_name,
    ts_features,
    results_perm_mae_by_part=None,
    results_perm_deviance_by_part=None,
    results_shap_share_by_part=None,
    fitted_models=None,
    train_fit_dates=None,
    X_by_partition=None,
    features_by_partition=None,
    extra_figures=None,
    run_tags=None,
    run_name_fn=None,
    tmp_root="mlflow_tmp",
    log_mlflow_dataset=True,
    log_mlflow_model=True,
    source_dataset_df=None,
    source_dataset_name=None,     # gardé pour compatibilité
    source_dataset_source=None,   # gardé pour compatibilité
    target_col=None,
    n_lags=None,
    exog_cols=None,
):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    df_log = score_df_exp.copy()

    required_cols = {"model_label", "partition", "mae"}
    missing = required_cols - set(df_log.columns)
    if missing:
        raise ValueError(f"score_df_exp missing required columns: {missing}")

    if "model_name" not in df_log.columns:
        df_log["model_name"] = df_log["model_label"].astype(str)

    df_log["model_label"] = df_log["model_label"].astype(str)
    df_log["partition"] = df_log["partition"].astype(str)
    df_log["model_name"] = df_log["model_name"].astype(str)

    for _, row in df_log.iterrows():
        model_label = str(row["model_label"])
        partition = str(row["partition"])
        model_name = str(row["model_name"])

        run_name = str(run_name_fn(row)) if callable(run_name_fn) else f"{model_label} | {partition}"

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("model_label", model_label)
                mlflow.set_tag("partition", partition)
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("run_name_custom", run_name)
                mlflow.set_tag("feast_feature_name", str(feast_feature_name))

                if run_tags:
                    for k, v in run_tags.items():
                        mlflow.set_tag(str(k), str(v))

                mlflow.log_param("model_label", model_label)
                mlflow.log_param("partition", partition)
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("run_name", run_name)

                if ts_features is not None:
                    if isinstance(ts_features, (list, tuple)):
                        mlflow.log_param("n_ts_features", len(ts_features))
                        mlflow.log_param("ts_features", ", ".join(map(str, ts_features)))
                    else:
                        mlflow.log_param("ts_features", str(ts_features))

                if target_col is not None:
                    mlflow.log_param("target_col", str(target_col))

                if n_lags is not None:
                    try:
                        mlflow.log_param("n_lags", int(n_lags))
                    except Exception:
                        mlflow.log_param("n_lags", str(n_lags))

                if exog_cols is not None:
                    try:
                        mlflow.log_param("n_exog_cols", len(exog_cols))
                    except Exception:
                        pass
                    try:
                        mlflow.log_param("exog_cols", ", ".join(map(str, exog_cols)))
                    except Exception:
                        mlflow.log_param("exog_cols", str(exog_cols))

                for col, val in row.items():
                    if col in {"model_label", "partition", "model_name"}:
                        continue
                    if isinstance(val, (int, float, np.integer, np.floating)) and pd.notna(val):
                        try:
                            mlflow.log_metric(col, float(val))
                        except Exception:
                            pass

                tmp_dir = Path(tmp_root) / model_label / partition
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                tmp_dir.mkdir(parents=True, exist_ok=True)

                row_df = pd.DataFrame([row])
                row_csv = tmp_dir / f"score_row_{model_label}_{partition}.csv"
                row_df.to_csv(row_csv, index=False)
                mlflow.log_artifact(str(row_csv), artifact_path="score")

                lb_part = _subset_partition(leaderboard_exp, partition)
                if lb_part is not None:
                    p = tmp_dir / f"leaderboard_{partition}.csv"
                    lb_part.to_csv(p, index=False)
                    mlflow.log_artifact(str(p), artifact_path="leaderboard")

                if results_perm_mae_by_part is not None and partition in results_perm_mae_by_part:
                    dfp = _coerce_explainability_obj(results_perm_mae_by_part[partition], model_label=model_label)
                    if dfp is not None and len(dfp):
                        p = tmp_dir / f"results_perm_mae_{model_label}_{partition}.csv"
                        dfp.to_csv(p, index=False)
                        mlflow.log_artifact(str(p), artifact_path="explainability")
                        score_col = "perm_mae_ratio" if "perm_mae_ratio" in dfp.columns else dfp.columns[-1]
                        top1 = _find_top1_feature(dfp, score_col)
                        if top1 is not None:
                            mlflow.set_tag("perm_mae_top1", top1)

                if results_perm_deviance_by_part is not None and partition in results_perm_deviance_by_part:
                    dfp = _coerce_explainability_obj(results_perm_deviance_by_part[partition], model_label=model_label)
                    if dfp is not None and len(dfp):
                        p = tmp_dir / f"results_perm_dev_{model_label}_{partition}.csv"
                        dfp.to_csv(p, index=False)
                        mlflow.log_artifact(str(p), artifact_path="explainability")
                        score_col = "perm_deviance_ratio" if "perm_deviance_ratio" in dfp.columns else dfp.columns[-1]
                        top1 = _find_top1_feature(dfp, score_col)
                        if top1 is not None:
                            mlflow.set_tag("perm_dev_top1", top1)

                if results_shap_share_by_part is not None and partition in results_shap_share_by_part:
                    dfp = _coerce_explainability_obj(results_shap_share_by_part[partition], model_label=model_label)
                    if dfp is not None and len(dfp):
                        p = tmp_dir / f"results_shap_share_{model_label}_{partition}.csv"
                        dfp.to_csv(p, index=False)
                        mlflow.log_artifact(str(p), artifact_path="explainability")
                        score_col = "shap_share" if "shap_share" in dfp.columns else dfp.columns[-1]
                        top1 = _find_top1_feature(dfp, score_col)
                        if top1 is not None:
                            mlflow.set_tag("shap_top1", top1)

                bkt_part = None
                if isinstance(bkt_score, dict):
                    bkt_part = bkt_score.get((model_label, partition), None)
                elif isinstance(bkt_score, pd.DataFrame):
                    tmp = bkt_score.copy()
                    if "model_label" in tmp.columns and "partition" in tmp.columns:
                        m = (
                            tmp["model_label"].astype(str).eq(model_label)
                            & tmp["partition"].astype(str).eq(partition)
                        )
                        bkt_part = tmp.loc[m].copy()
                    elif "model_label" in tmp.columns:
                        m = tmp["model_label"].astype(str).eq(model_label)
                        bkt_part = tmp.loc[m].copy()

                if isinstance(bkt_part, pd.DataFrame) and len(bkt_part):
                    bkt_part = _normalize_ds_in_df(bkt_part, ds_col="ds")
                    p = tmp_dir / f"bkt_{model_label}_{partition}.parquet"
                    bkt_part.to_parquet(p, index=False)
                    mlflow.log_artifact(str(p), artifact_path="backtest")

                meta_obj = None
                if isinstance(meta_models, dict):
                    meta_obj = meta_models.get((model_label, partition), meta_models.get(model_label, None))
                    if meta_obj is None:
                        meta_obj = meta_models.get("metas", {}).get(model_label, None)

                if meta_obj is not None:
                    p_json = tmp_dir / f"meta_{model_label}_{partition}.json"
                    _safe_write_json(meta_obj, p_json)
                    mlflow.log_artifact(str(p_json), artifact_path="meta")

                    p_pkl = tmp_dir / f"meta_{model_label}_{partition}.pkl"
                    _safe_write_pickle(meta_obj, p_pkl)
                    mlflow.log_artifact(str(p_pkl), artifact_path="meta")

                # =====================================================
                # DATASET FEAST visible dans MLflow UI
                # =====================================================
                dataset_ok = False
                dataset_err = None
                dataset_name_logged = None

                if log_mlflow_dataset and isinstance(source_dataset_df, pd.DataFrame):
                    dataset_ok, dataset_err, dataset_name_logged = _log_feast_dataset_entity_to_mlflow(
                        source_dataset_df,
                        feast_feature_name=feast_feature_name,
                        model_label=model_label,
                        partition=partition,
                        tmp_dir=str(tmp_dir),
                        context="training",
                        source_dataset_name=source_dataset_name,
                    )

                mlflow.set_tag("dataset_logged", str(bool(dataset_ok)).lower())
                if dataset_name_logged is not None:
                    mlflow.set_tag("dataset_name_logged", dataset_name_logged)

                if dataset_err is not None:
                    mlflow.set_tag("dataset_log_error", str(dataset_err)[:500])
                    p = tmp_dir / "dataset_log_error.txt"
                    with open(p, "w", encoding="utf-8") as f:
                        f.write(str(dataset_err))
                    mlflow.log_artifact(str(p), artifact_path="logs")

                Xp2 = None
                if isinstance(X_by_partition, dict):
                    Xp = X_by_partition.get((model_label, partition), None)
                    if Xp is None:
                        Xp = X_by_partition.get((model_label, "ALL"), None)
                    if isinstance(Xp, pd.DataFrame):
                        Xp2 = Xp.copy()
                        if "ds" in Xp2.columns:
                            Xp2["ds"] = pd.to_datetime(Xp2["ds"], errors="coerce")
                        p = tmp_dir / f"X_{model_label}_{partition}.parquet"
                        Xp2.to_parquet(p, index=False)
                        mlflow.log_artifact(str(p), artifact_path="functional")

                if isinstance(features_by_partition, dict):
                    feats = features_by_partition.get((model_label, partition), None)
                    if feats is None:
                        feats = features_by_partition.get((model_label, "ALL"), None)
                    if feats is not None:
                        p = tmp_dir / f"features_{model_label}_{partition}.json"
                        _safe_write_json(list(map(str, feats)), p)
                        mlflow.log_artifact(str(p), artifact_path="functional")

                model_obj = None
                if isinstance(fitted_models, dict):
                    model_obj = fitted_models.get((model_label, partition), None)
                    if model_obj is None:
                        model_obj = fitted_models.get((model_label, "ALL"), None)
                    if model_obj is None:
                        model_obj = fitted_models.get(model_label, None)

                if model_obj is not None:
                    saved_path = _serialize_model_candidate(
                        model_obj=model_obj,
                        base_dir=tmp_dir / "models",
                        stem=f"model_{model_label}_{partition}",
                    )
                    if saved_path is not None:
                        mlflow.log_artifact(str(saved_path), artifact_path="models")

                    try:
                        est = _unwrap_estimator_from_mlf(model_obj, preferred_key=model_label)
                        saved_est = _serialize_model_candidate(
                            model_obj=est,
                            base_dir=tmp_dir / "models",
                            stem=f"estimator_{model_label}_{partition}",
                        )
                        if saved_est is not None:
                            mlflow.log_artifact(str(saved_est), artifact_path="models")

                        if log_mlflow_model:
                            X_example = Xp2 if isinstance(Xp2, pd.DataFrame) else source_dataset_df
                            flavor_used = _log_model_entity_to_mlflow(
                                est,
                                model_name=f"model_entity_{model_label}_{partition}",
                                X_example=X_example,
                            )
                            if flavor_used is not None:
                                mlflow.set_tag("mlflow_model_flavor", flavor_used)
                    except Exception as e:
                        p = tmp_dir / "model_log_error.txt"
                        with open(p, "w", encoding="utf-8") as f:
                            f.write(str(e))
                        mlflow.log_artifact(str(p), artifact_path="logs")

                if isinstance(train_fit_dates, dict):
                    tfd = train_fit_dates.get((model_label, partition), None)
                    if tfd is None:
                        tfd = train_fit_dates.get(model_label, None)
                    if tfd is not None:
                        p = tmp_dir / f"train_fit_dates_{model_label}_{partition}.json"
                        _safe_write_json(tfd, p)
                        mlflow.log_artifact(str(p), artifact_path="meta")

                if isinstance(extra_figures, dict):
                    figs = extra_figures.get((model_label, partition), None)
                    if isinstance(figs, dict):
                        for fname, fig in figs.items():
                            try:
                                mlflow.log_figure(fig, f"plots/{fname}")
                            except Exception:
                                local_fig = tmp_dir / fname
                                fig.savefig(local_fig, dpi=200, bbox_inches="tight")
                                mlflow.log_artifact(str(local_fig), artifact_path="plots")

                out_txt = stdout_buf.getvalue().strip()
                err_txt = stderr_buf.getvalue().strip()

                if out_txt:
                    p = tmp_dir / "stdout.txt"
                    with open(p, "w", encoding="utf-8") as f:
                        f.write(out_txt)
                    mlflow.log_artifact(str(p), artifact_path="logs")

                if err_txt:
                    p = tmp_dir / "stderr.txt"
                    with open(p, "w", encoding="utf-8") as f:
                        f.write(err_txt)
                    mlflow.log_artifact(str(p), artifact_path="logs")


# =========================================================
# Import MLflow complet
# =========================================================
def import_mlflow_experiment(
    *,
    tracking_uri,
    experiment_name,
    dl_dir="mlflow_import",
    prefer_latest_only=True,
    try_load_logged_mlflow_models=True,
):
    _ensure_dir(dl_dir)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment introuvable: {experiment_name}")

    runs = mlflow.search_runs([exp.experiment_id], output_format="pandas")
    if runs.empty:
        raise ValueError("Aucun run trouvé.")

    runs = runs.sort_values("start_time", ascending=False).reset_index(drop=True)

    def _get_col(df, cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    col_model = _get_col(runs, ["params.model_label", "tags.model_label"])
    col_part = _get_col(runs, ["params.partition", "tags.partition"])
    col_name = _get_col(runs, ["params.model_name", "tags.model_name"])

    if col_model is None or col_part is None:
        raise ValueError("Impossible de trouver model_label / partition dans les runs MLflow.")

    runs["model_label"] = runs[col_model].astype(str)
    runs["partition"] = runs[col_part].astype(str)
    runs["model_name"] = runs[col_name].astype(str) if col_name is not None else runs["model_label"].astype(str)

    if prefer_latest_only:
        runs = (
            runs.sort_values("start_time", ascending=False)
            .drop_duplicates(subset=["model_label", "partition"], keep="first")
            .reset_index(drop=True)
        )

    score_rows = []
    leaderboard_rows = []

    results_perm_mae_by_part = {}
    results_perm_deviance_by_part = {}
    results_shap_share_by_part = {}

    bkt_by_run = {}
    X_by_run = {}
    features_by_run = {}
    models_mlflow = {}
    meta_by_run = {}
    train_fit_dates_by_run = {}

    def _safe_read_csv(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    def _safe_read_parquet(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def _safe_read_json(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _safe_load_model(path):
        try:
            if str(path).lower().endswith(".joblib"):
                return joblib.load(path)
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _append_expl_store(store, partition, df, model_label):
        if df is None or len(df) == 0:
            return
        tmp = df.copy()
        if "model_label" not in tmp.columns:
            tmp["model_label"] = str(model_label)
        store.setdefault(partition, [])
        store[partition].append(tmp)

    for _, rr in runs.iterrows():
        run_id = rr["run_id"]
        model_label = str(rr["model_label"])
        partition = str(rr["partition"])
        key = (model_label, partition)

        run_dir = os.path.join(dl_dir, model_label, partition)
        _ensure_dir(run_dir)

        try:
            artifacts = _list_artifacts_recursive(client, run_id, path="")
        except Exception:
            artifacts = []

        for ap in artifacts:
            apl = ap.lower()

            if "score/" in apl and apl.endswith(".csv"):
                local = _download_artifact(client, run_id, ap, run_dir)
                df = _safe_read_csv(local)
                if df is not None and len(df):
                    if "model_label" not in df.columns:
                        df["model_label"] = model_label
                    if "partition" not in df.columns:
                        df["partition"] = partition
                    score_rows.append(df)

            elif "leaderboard/" in apl and apl.endswith(".csv"):
                local = _download_artifact(client, run_id, ap, run_dir)
                df = _safe_read_csv(local)
                if df is not None and len(df):
                    if "partition" not in df.columns:
                        df["partition"] = partition
                    leaderboard_rows.append(df)

            elif "explainability/" in apl and apl.endswith(".csv"):
                local = _download_artifact(client, run_id, ap, run_dir)
                df = _safe_read_csv(local)
                if df is None or len(df) == 0:
                    continue

                if "perm_mae" in apl:
                    _append_expl_store(results_perm_mae_by_part, partition, df, model_label)
                elif "perm_dev" in apl or "perm_deviance" in apl:
                    _append_expl_store(results_perm_deviance_by_part, partition, df, model_label)
                elif "shap" in apl:
                    _append_expl_store(results_shap_share_by_part, partition, df, model_label)

            elif "backtest/" in apl and apl.endswith(".parquet"):
                local = _download_artifact(client, run_id, ap, run_dir)
                df = _safe_read_parquet(local)
                if df is not None:
                    bkt_by_run[key] = df

            elif ("functional/" in apl or "/x_" in apl or apl.startswith("x_")) and apl.endswith(".parquet"):
                local = _download_artifact(client, run_id, ap, run_dir)
                df = _safe_read_parquet(local)
                if df is not None:
                    X_by_run[key] = df

            elif ("functional/" in apl or "features_" in apl) and apl.endswith(".json"):
                local = _download_artifact(client, run_id, ap, run_dir)
                obj = _safe_read_json(local)
                if obj is not None:
                    features_by_run[key] = obj

            elif "meta/" in apl and "train_fit_dates_" in apl and apl.endswith(".json"):
                local = _download_artifact(client, run_id, ap, run_dir)
                obj = _safe_read_json(local)
                if obj is not None:
                    train_fit_dates_by_run[key] = obj

            elif "meta/" in apl and "meta_" in apl and apl.endswith(".json"):
                local = _download_artifact(client, run_id, ap, run_dir)
                obj = _safe_read_json(local)
                if obj is not None:
                    meta_by_run[key] = obj

        if try_load_logged_mlflow_models:
            candidate_model_entities = [
                f"model_entity_{model_label}_{partition}",
                "model",
            ]
            for art in candidate_model_entities:
                loaded = _load_mlflow_logged_model(run_id, art)
                if loaded is not None:
                    models_mlflow[key] = loaded
                    break

        if key not in models_mlflow:
            model_candidates = []
            for ap in artifacts:
                apl = ap.lower()
                if "models/" in apl and (apl.endswith(".joblib") or apl.endswith(".pkl") or apl.endswith(".pickle")):
                    model_candidates.append(ap)

            def _rank_model_path(x):
                xl = x.lower()
                score = 100
                if f"estimator_{model_label.lower()}_{partition.lower()}.joblib" in xl:
                    score = 0
                elif f"model_{model_label.lower()}_{partition.lower()}.joblib" in xl:
                    score = 1
                elif xl.endswith(".joblib"):
                    score = 2
                elif xl.endswith(".pkl") or xl.endswith(".pickle"):
                    score = 3
                return score, len(x)

            model_candidates = sorted(model_candidates, key=_rank_model_path)

            for ap in model_candidates:
                local = _download_artifact(client, run_id, ap, run_dir)
                loaded_model = _safe_load_model(local)
                if loaded_model is not None:
                    models_mlflow[key] = loaded_model
                    break

    score_df_exp = (
        pd.concat(score_rows, axis=0, ignore_index=True).drop_duplicates().reset_index(drop=True)
        if len(score_rows) else None
    )

    leaderboard_exp = (
        pd.concat(leaderboard_rows, axis=0, ignore_index=True).drop_duplicates().reset_index(drop=True)
        if len(leaderboard_rows) else None
    )

    for part, lst in list(results_perm_mae_by_part.items()):
        results_perm_mae_by_part[part] = (
            pd.concat(lst, axis=0, ignore_index=True).drop_duplicates().reset_index(drop=True)
            if len(lst) else pd.DataFrame()
        )

    for part, lst in list(results_perm_deviance_by_part.items()):
        results_perm_deviance_by_part[part] = (
            pd.concat(lst, axis=0, ignore_index=True).drop_duplicates().reset_index(drop=True)
            if len(lst) else pd.DataFrame()
        )

    for part, lst in list(results_shap_share_by_part.items()):
        results_shap_share_by_part[part] = (
            pd.concat(lst, axis=0, ignore_index=True).drop_duplicates().reset_index(drop=True)
            if len(lst) else pd.DataFrame()
        )

    return {
        "runs": runs,
        "score_df_exp": score_df_exp,
        "leaderboard_exp": leaderboard_exp,
        "results_perm_mae_by_part": results_perm_mae_by_part,
        "results_perm_deviance_by_part": results_perm_deviance_by_part,
        "results_shap_share_by_part": results_shap_share_by_part,
        "bkt_by_run": bkt_by_run,
        "X_by_run": X_by_run,
        "features_by_run": features_by_run,
        "models_mlflow": models_mlflow,
        "meta_by_run": meta_by_run,
        "train_fit_dates_by_run": train_fit_dates_by_run,
    }


def build_models_and_X_for_functional_plots(
    *,
    models_mlflow,
    X_by_run,
    features_by_run=None,
    preferred_partition="ALL",
    labels_map=None,
):
    if labels_map is None:
        labels_map = {}

    all_keys = sorted(set(models_mlflow.keys()) | set(X_by_run.keys()))
    all_model_labels = sorted(set(k[0] for k in all_keys))

    models_dict = {}
    X_dict_out = {}
    features_dict = {}

    for model_label in all_model_labels:
        chosen_key = None

        if (model_label, preferred_partition) in models_mlflow and (model_label, preferred_partition) in X_by_run:
            chosen_key = (model_label, preferred_partition)
        else:
            common = [k for k in all_keys if k[0] == model_label and k in models_mlflow and k in X_by_run]
            if len(common):
                common = sorted(common, key=lambda x: (x[1] != preferred_partition, x[1]))
                chosen_key = common[0]

        if chosen_key is None:
            continue

        display_name = labels_map.get(model_label, model_label)
        model_obj = models_mlflow[chosen_key]
        prep = None

        models_dict[display_name] = (model_obj, prep)
        X_dict_out[display_name] = X_by_run[chosen_key]

        if isinstance(features_by_run, dict) and chosen_key in features_by_run:
            features_dict[display_name] = features_by_run[chosen_key]

    return models_dict, X_dict_out, features_dict