import numpy as np
import pandas as pd

from typing import Any, Dict, List, Optional
from dateutil.relativedelta import relativedelta
from mlforecast.utils import PredictionIntervals

# ✅ modules internes corrigés
from Model_training.ModelSpec import ModelSpec
from Data_preparation.time import _ensure_ms, _n_windows_monthly, _slice_cv_block
from Model_training.cross_validation import (
    TrainWindowType,
    _get_train_slice,
    _tune_on_train,
    _build_backtest_cutoffs,
    _eval_cutoffs_with_mlforecast,
)

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
    cv_method: str = "rolling",
    kfold_n_splits: int = 5,
    hv_gap: int = 12,
    features: Optional[List[str]] = None,
    train_window_type: TrainWindowType = "expanding",
    rolling_window_size: Optional[int] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    
    # run modèle par modèle puis merge
    bkts = []
    metas = {}
    bundles = {}

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
            cv_method=cv_method,
            kfold_n_splits=kfold_n_splits,
            hv_gap=hv_gap,
            features=features,
             train_window_type=train_window_type,
            rolling_window_size=rolling_window_size,
        )
        metas[spec.name] = meta_m

        # ✅ bundle explicabilité (si dispo)
        fitted_models = meta_m.get("fitted_models", None)
        train_dates = meta_m.get("train_fit_dates", None) or meta_m.get("train_periods", None)
        features = meta_m.get("features", None)
        preprocs = meta_m.get("preprocs", None)

        if (
            isinstance(fitted_models, list)
            and len(fitted_models) > 0
            and train_dates is not None
            and features is not None
        ):
            bundles[spec.name] = {
                "models": fitted_models,
                "train_fit_dates": list(pd.to_datetime(pd.Index(train_dates))),
                "features": list(features),
                "preprocs": preprocs,
                "params": meta_m,
            }

        if len(bkt_m):
            bkts.append(bkt_m)

    if not bkts:
        return pd.DataFrame(), {
            "error": "aucun modèle n’a produit de backtest",
            "metas": metas,
            "bundles": bundles,
        }

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
    cv_method: str = "rolling",
    kfold_n_splits: int = 5,
    hv_gap: int = 12,
    features: Optional[List[str]] = None,
    train_window_type: TrainWindowType = "expanding",
    rolling_window_size: Optional[int] = None,
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
    exp_end = _ensure_ms(exp_end)

    cutoff_start_all = exp_start - relativedelta(months=h)
    cutoff_end_all = exp_end - relativedelta(months=h)
    total_partitions = _n_windows_monthly(cutoff_start_all, cutoff_end_all)

    # anti-fuite
    ts = ts[ts["ds"] <= exp_end].copy()

    # conformal au backtest (ici ON)
    pi = PredictionIntervals(
        h=h,
        n_windows=int(pi_windows),
        method="conformal_distribution",
    )

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

    fitted_models: list = []
    train_fit_dates: list = []

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
            cv_method=cv_method,
            kfold_n_splits=kfold_n_splits,
            hv_gap=hv_gap,
            train_window_type=train_window_type,
            rolling_window_size=rolling_window_size,
        )

        if best_params is None:
            continue

        params_history.append({
            "model": spec.name,
            "block": block_idx,
            "cutoff_start": cutoff_start_blk,
            "n_windows": int(n_windows_blk),
            "cv_method": cv_method,
            "kfold_n_splits": None if kfold_n_splits is None else int(kfold_n_splits),
            "hv_gap": None if hv_gap is None else int(hv_gap),
            **best_params,
        })
        tune_history.append({
            "model": spec.name,
            "block": block_idx,
            "cutoff_start": cutoff_start_blk,
            "tune_mae": float(tune_mae),
            "cv_method": cv_method,
        })

        # --------- CV predictions (avec PI)
        
        if train_window_type == "expanding":
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
        else:
            cutoffs_blk = _build_backtest_cutoffs(
                ts_blk,
                h=h,
                cutoff_start=cutoff_start_blk,
                n_windows=int(n_windows_blk),
                step_size=int(step_size),
                min_train_n=min_train_n,
                window_type=train_window_type,
                rolling_window_size=rolling_window_size,
            )

            bkt_blk = _eval_cutoffs_with_mlforecast(
                ts_blk,
                spec=spec,
                freq=freq,
                params=best_params,
                h=h,
                cutoffs=cutoffs_blk,
                gap=0,
                train_window_type=train_window_type,
                rolling_window_size=rolling_window_size,
            )

            # colonnes PI vides pour garder une structure proche
            for lv in levels:
                bkt_blk[f"{spec.pred_col}-lo-{lv}"] = np.nan
                bkt_blk[f"{spec.pred_col}-hi-{lv}"] = np.nan
                
                bkt_blk[f"{spec.name}_tune_block"] = block_idx
        bkt_blk[f"{spec.name}_tune_mae"] = float(tune_mae)

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
            # --------- 2) Fallback garanti: 1 modèle par bloc
            mlf_fitted = spec.build_mlf(freq, best_params)
            ts_fit_final = _get_train_slice(
                ts_train_for_tune,
                train_end=cutoff_start_blk,
                window_type=train_window_type,
                rolling_window_size=rolling_window_size,
            )
            mlf_fitted.fit(ts_fit_final, static_features=[])
            fitted_models.append(mlf_fitted)
            train_fit_dates.append(pd.to_datetime(cutoff_start_blk))

        all_bkts.append(bkt_blk)

    if not all_bkts:
        return pd.DataFrame(), {"error": f"{spec.name}: aucun bloc produit"}

    bkt = pd.concat(all_bkts, ignore_index=True)

    # filtre exp exact
    bkt = bkt[(bkt["ds"] >= exp_start) & (bkt["ds"] <= exp_end)].copy()
    bkt = bkt.sort_values(["unique_id", "ds", "cutoff"]).reset_index(drop=True)

    _features = list(features) if features is not None else None

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
        cv_method=cv_method,
        kfold_n_splits=None if kfold_n_splits is None else int(kfold_n_splits),
        hv_gap=None if hv_gap is None else int(hv_gap),
        train_window_type=train_window_type,
        rolling_window_size=rolling_window_size,
        params_history=params_history,
        tune_history=tune_history,
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
