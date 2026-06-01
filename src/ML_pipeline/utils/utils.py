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


# -----------------------------
# Helpers temps
# -----------------------------


# -----------------------------
# Spec modèle
# -----------------------------


# -----------------------------
# Helpers CV custom
# -----------------------------
from typing import Literal

TrainWindowType = Literal["expanding", "sliding"]

def _get_train_slice(
    ts: pd.DataFrame,
    *,
    train_end,
    window_type: TrainWindowType = "expanding",
    rolling_window_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Retourne l'échantillon d'entraînement jusqu'à train_end
    selon :
      - expanding : toutes les dates <= train_end
      - sliding   : uniquement les rolling_window_size derniers mois
    """
    train_end = _ensure_ms(train_end)

    out = ts.copy()
    out["ds"] = (
        pd.to_datetime(out["ds"])
        .dt.to_period("M")
        .dt.to_timestamp(how="start")
    )

    out = out[out["ds"] <= train_end].copy()

    if window_type == "expanding":
        return out

    if window_type == "sliding":
        if rolling_window_size is None or int(rolling_window_size) <= 0:
            raise ValueError(
                "rolling_window_size doit être un entier > 0 quand window_type='sliding'"
            )

        train_start = train_end - relativedelta(months=int(rolling_window_size) - 1)
        out = out[out["ds"] >= train_start].copy()
        return out

    raise ValueError("window_type doit être 'expanding' ou 'sliding'")


def _build_cutoffs_hvblock(
    ts_train: pd.DataFrame,
    *,
    h: int,
    n_windows: int,
    min_train_n: Optional[int] = None,
    gap: int = 12,
    window_type: TrainWindowType = "expanding",
    rolling_window_size: Optional[int] = None,
) -> List[pd.Timestamp]:
    ds_all = sorted(pd.to_datetime(ts_train["ds"]).dropna().unique())
    ds_all = [pd.Timestamp(x).to_period("M").to_timestamp(how="start") for x in ds_all]

    cutoffs = []

    for cutoff in ds_all:
        cutoff = pd.Timestamp(cutoff)

        train_end = cutoff - relativedelta(months=int(gap))
        df_fit = _get_train_slice(
            ts_train,
            train_end=train_end,
            window_type=window_type,
            rolling_window_size=rolling_window_size,
        )

        n_train_eff = len(df_fit)

        test_end = cutoff + relativedelta(months=h)
        has_test = test_end <= ds_all[-1]

        if not has_test:
            continue

        if min_train_n is not None and n_train_eff < int(min_train_n):
            continue

        if n_train_eff <= 0:
            continue

        cutoffs.append(cutoff)

    return cutoffs[-int(n_windows):]

def _build_cutoffs_kfold(
    ts_train: pd.DataFrame,
    *,
    h: int,
    n_splits: int,
    min_train_n: Optional[int] = None,
    window_type: TrainWindowType = "expanding",
    rolling_window_size: Optional[int] = None,
) -> List[pd.Timestamp]:
    ds_all = sorted(pd.to_datetime(ts_train["ds"]).dropna().unique())
    ds_all = [pd.Timestamp(x).to_period("M").to_timestamp(how="start") for x in ds_all]

    eligible = []
    for cutoff in ds_all:
        cutoff = pd.Timestamp(cutoff)

        has_test = (cutoff + relativedelta(months=h)) in ds_all
        if not has_test:
            continue

        df_fit = _get_train_slice(
            ts_train,
            train_end=cutoff,
            window_type=window_type,
            rolling_window_size=rolling_window_size,
        )

        n_train = len(df_fit)

        if min_train_n is None or n_train >= int(min_train_n):
            eligible.append(cutoff)

    if len(eligible) == 0:
        return []

    chunks = np.array_split(np.array(eligible, dtype="datetime64[ns]"), int(n_splits))

    out = []
    for ch in chunks:
        if len(ch):
            out.append(pd.Timestamp(ch[-1]).to_period("M").to_timestamp(how="start"))
    return out

def _build_backtest_cutoffs(
    ts: pd.DataFrame,
    *,
    h: int,
    cutoff_start,
    n_windows: int,
    step_size: int,
    min_train_n: Optional[int] = None,
    window_type: TrainWindowType = "expanding",
    rolling_window_size: Optional[int] = None,
) -> List[pd.Timestamp]:
    ts = ts.copy()
    ts["ds"] = (
        pd.to_datetime(ts["ds"])
        .dt.to_period("M")
        .dt.to_timestamp(how="start")
    )

    ds_all = sorted(ts["ds"].dropna().unique())
    ds_all = [pd.Timestamp(x).to_period("M").to_timestamp(how="start") for x in ds_all]

    cutoff_start = _ensure_ms(cutoff_start)
    out = []

    for j in range(0, int(n_windows), int(step_size)):
        cutoff = cutoff_start + relativedelta(months=j)
        train_end = cutoff
        test_end = cutoff + relativedelta(months=h)

        if test_end not in ds_all and test_end > ds_all[-1]:
            continue

        df_fit = _get_train_slice(
            ts,
            train_end=train_end,
            window_type=window_type,
            rolling_window_size=rolling_window_size,
        )

        if min_train_n is not None and len(df_fit) < int(min_train_n):
            continue

        if len(df_fit) == 0:
            continue

        out.append(cutoff)

    return out

def _eval_cutoffs_with_mlforecast(
    ts_train: pd.DataFrame,
    *,
    spec: ModelSpec,
    freq: str,
    params: Dict[str, Any],
    h: int,
    cutoffs: List[pd.Timestamp],
    gap: int = 0,
    train_window_type: TrainWindowType = "expanding",
    rolling_window_size: Optional[int] = None,
) -> pd.DataFrame:
    ts_train = ts_train.copy()
    ts_train["ds"] = (
        pd.to_datetime(ts_train["ds"])
        .dt.to_period("M")
        .dt.to_timestamp(how="start")
    )

    debug_models = {"RIDGE", "LGBM"}
    do_debug = spec.name in debug_models

    if do_debug:
        print(
            f"\n[DEBUG {spec.name}] _eval_cutoffs_with_mlforecast "
            f"| len(ts_train)={len(ts_train)} | n_cutoffs={len(cutoffs)} "
            f"| h={h} | gap={gap} | train_window_type={train_window_type} "
            f"| rolling_window_size={rolling_window_size}"
        )

    rows = []

    for i, cutoff in enumerate(cutoffs, start=1):
        cutoff = _ensure_ms(cutoff)
        train_end = cutoff - relativedelta(months=int(gap))

        df_fit = _get_train_slice(
            ts_train,
            train_end=train_end,
            window_type=train_window_type,
            rolling_window_size=rolling_window_size,
        )

        df_future_true = ts_train[
            (ts_train["ds"] > train_end) &
            (ts_train["ds"] <= train_end + relativedelta(months=h))
        ].copy()

        if do_debug:
            train_min = df_fit["ds"].min() if len(df_fit) else None
            train_max = df_fit["ds"].max() if len(df_fit) else None
            print(
                f"[DEBUG {spec.name}] cutoff #{i}={cutoff.date()} "
                f"| train_end={train_end.date()} "
                f"| df_fit.shape={df_fit.shape} "
                f"| train_range=({train_min}, {train_max}) "
                f"| df_future_true.shape={df_future_true.shape}"
            )

        if df_fit.empty or df_future_true.empty:
            if do_debug:
                print(
                    f"[DEBUG {spec.name}] cutoff #{i} skipped "
                    f"(df_fit.empty={df_fit.empty}, df_future_true.empty={df_future_true.empty})"
                )
            continue

        exog_cols = [c for c in df_fit.columns if c not in ["unique_id", "ds", "y"]]

        if do_debug:
            print(
                f"[DEBUG {spec.name}] cutoff #{i} "
                f"| n_exog={len(exog_cols)} "
                f"| exog sample={exog_cols[:5]}"
            )

        mlf = spec.build_mlf(freq, params)
        mlf.fit(df_fit, static_features=[])

        future_df = mlf.make_future_dataframe(h)
        future_df["ds"] = (
            pd.to_datetime(future_df["ds"])
            .dt.to_period("M")
            .dt.to_timestamp(how="start")
        )

        if do_debug:
            print(
                f"[DEBUG {spec.name}] cutoff #{i} "
                f"| future_df.shape={future_df.shape} "
                f"| future_df ds range=({future_df['ds'].min()}, {future_df['ds'].max()})"
            )

        X_df = future_df.merge(
            ts_train[["unique_id", "ds"] + exog_cols],
            on=["unique_id", "ds"],
            how="left",
        )

        if do_debug:
            n_missing_total = int(X_df[exog_cols].isna().sum().sum()) if len(exog_cols) else 0
            print(
                f"[DEBUG {spec.name}] cutoff #{i} "
                f"| X_df.shape={X_df.shape} "
                f"| missing_exog_total={n_missing_total}"
            )

            if len(exog_cols):
                missing_by_col = X_df[exog_cols].isna().sum()
                missing_by_col = missing_by_col[missing_by_col > 0]
                if len(missing_by_col):
                    print(
                        f"[DEBUG {spec.name}] cutoff #{i} "
                        f"| missing_by_col={missing_by_col.to_dict()}"
                    )

        if len(exog_cols) and X_df[exog_cols].isna().any().any():
            if do_debug:
                print(f"[DEBUG {spec.name}] cutoff #{i} skipped (missing future exog)")
            continue

        try:
            fcst = mlf.predict(h=h, X_df=X_df)
        except Exception as e:
            if do_debug:
                print(f"[DEBUG {spec.name}] cutoff #{i} predict FAILED -> {repr(e)}")
            continue

        fcst["ds"] = (
            pd.to_datetime(fcst["ds"])
            .dt.to_period("M")
            .dt.to_timestamp(how="start")
        )

        if do_debug:
            print(
                f"[DEBUG {spec.name}] cutoff #{i} "
                f"| fcst.shape={fcst.shape} "
                f"| fcst cols={fcst.columns.tolist()}"
            )

        tmp = df_future_true[["unique_id", "ds", "y"]].merge(
            fcst[["unique_id", "ds", spec.pred_col]],
            on=["unique_id", "ds"],
            how="inner",
        )

        if do_debug:
            print(f"[DEBUG {spec.name}] cutoff #{i} | merged tmp.shape={tmp.shape}")

        if len(tmp):
            tmp["cutoff"] = cutoff
            rows.append(tmp)
            if do_debug:
                print(f"[DEBUG {spec.name}] cutoff #{i} kept")
        else:
            if do_debug:
                print(f"[DEBUG {spec.name}] cutoff #{i} dropped (tmp empty)")

    if do_debug:
        print(f"[DEBUG {spec.name}] total kept rows blocks={len(rows)}")

    if not rows:
        if do_debug:
            print(f"[DEBUG {spec.name}] returning EMPTY cv dataframe")
        return pd.DataFrame(columns=["unique_id", "ds", "cutoff", "y", spec.pred_col])

    out = pd.concat(rows, ignore_index=True)

    if do_debug:
        print(f"[DEBUG {spec.name}] final cv.shape={out.shape}")

    return out

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
    cv_method: str = "rolling",
    kfold_n_splits: int = 5,
    hv_gap: int = 12,
    train_window_type: TrainWindowType = "expanding",
    rolling_window_size: Optional[int] = None,
) -> tuple[Optional[Dict[str, Any]], float]:

    if min_train_n is not None and len(ts_train) < int(min_train_n):
        return None, float("nan")

    if not spec.tunable:
        return (spec.fixed_params or {}), float("nan")

    if not spec.param_space:
        raise ValueError(f"{spec.name}: tunable=True mais param_space=None")

    ts_train = ts_train.copy()
    ts_train["ds"] = (
        pd.to_datetime(ts_train["ds"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp(how="start")
        .dt.normalize()
    )
    ts_train = ts_train.dropna(subset=["ds"]).reset_index(drop=True)

    # conformal au tuning (souvent OFF)
    pi_tune = (
        PredictionIntervals(
            h=h,
            n_windows=min(int(spec.tune_cv_windows), int(pi_windows_cap)),
            method="conformal_distribution",
        )
        if spec.use_conformal_in_tune
        else None
    )

    # générateur d'essais
    if spec.search == "grid":
        sampler = ParameterGrid(spec.param_space)
    else:
        sampler = ParameterSampler(
            spec.param_space,
            n_iter=int(spec.n_iter),
            random_state=int(seed),
        )

    best_params = None
    best_score = np.inf

    for params in sampler:
        params = dict(params)

        # =========================================================
        # 1) Rolling CV
        # =========================================================
        if cv_method == "rolling":
            # expanding: on garde MLForecast natif
            if train_window_type == "expanding":
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

                if len(cv) == 0:
                    continue

                score = mean_absolute_error(cv["y"], cv[spec.pred_col])

            # sliding: évaluation rolling manuelle
            elif train_window_type == "sliding":
                ds_all = sorted(pd.to_datetime(ts_train["ds"]).dropna().unique())
                ds_all = [
                    pd.Timestamp(x).to_period("M").to_timestamp(how="start")
                    for x in ds_all
                ]

                eligible_cutoffs = []

                for cutoff in ds_all:
                    cutoff = pd.Timestamp(cutoff)

                    test_end = cutoff + relativedelta(months=h)
                    has_test = test_end <= ds_all[-1]
                    if not has_test:
                        continue

                    df_fit = _get_train_slice(
                        ts_train,
                        train_end=cutoff,
                        window_type=train_window_type,
                        rolling_window_size=rolling_window_size,
                    )

                    if min_train_n is not None and len(df_fit) < int(min_train_n):
                        continue

                    if len(df_fit) == 0:
                        continue

                    eligible_cutoffs.append(cutoff)

                cutoffs = eligible_cutoffs[-int(spec.tune_cv_windows):]

                if len(cutoffs) == 0:
                    continue

                cv = _eval_cutoffs_with_mlforecast(
                    ts_train,
                    spec=spec,
                    freq=freq,
                    params=params,
                    h=h,
                    cutoffs=cutoffs,
                    gap=0,
                    train_window_type=train_window_type,
                    rolling_window_size=rolling_window_size,
                )

                if len(cv) == 0:
                    continue

                score = mean_absolute_error(cv["y"], cv[spec.pred_col])

            else:
                raise ValueError("train_window_type doit être 'expanding' ou 'sliding'")

        # =========================================================
        # 2) KFold CV custom
        # =========================================================
        elif cv_method == "kfold":
            cutoffs = _build_cutoffs_kfold(
                ts_train,
                h=h,
                n_splits=int(kfold_n_splits),
                min_train_n=min_train_n,
                window_type=train_window_type,
                rolling_window_size=rolling_window_size,
            )

            if len(cutoffs) == 0:
                continue

            cv = _eval_cutoffs_with_mlforecast(
                ts_train,
                spec=spec,
                freq=freq,
                params=params,
                h=h,
                cutoffs=cutoffs,
                gap=0,
                train_window_type=train_window_type,
                rolling_window_size=rolling_window_size,
            )

            if len(cv) == 0:
                continue

            score = mean_absolute_error(cv["y"], cv[spec.pred_col])

        # =========================================================
        # 3) HV-Block CV
        # =========================================================
        elif cv_method == "hvblock":
            cutoffs = _build_cutoffs_hvblock(
                ts_train,
                h=h,
                n_windows=int(spec.tune_cv_windows),
                min_train_n=min_train_n,
                gap=int(hv_gap),
                window_type=train_window_type,
                rolling_window_size=rolling_window_size,
            )

            if len(cutoffs) == 0:
                continue

            cv = _eval_cutoffs_with_mlforecast(
                ts_train,
                spec=spec,
                freq=freq,
                params=params,
                h=h,
                cutoffs=cutoffs,
                gap=int(hv_gap),
                train_window_type=train_window_type,
                rolling_window_size=rolling_window_size,
            )

            if len(cv) == 0:
                continue

            score = mean_absolute_error(cv["y"], cv[spec.pred_col])

        else:
            raise ValueError("cv_method doit être 'rolling', 'kfold' ou 'hvblock'")

        if np.isfinite(score) and score < best_score:
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


# =========================================================
# ✅ EXPLICABILITÉ ROLLING GÉNÉRALE
# Permutation MAE + Permutation Deviance + SHAP share
# =========================================================

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error






import numpy as np
import pandas as pd



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