import numpy as np
import pandas as pd

from typing import Any, Dict, List, Optional, Literal
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid, ParameterSampler
from mlforecast.utils import PredictionIntervals

# ✅ imports corrigés
from Model_training.ModelSpec import ModelSpec
from Data_preparation.time import _ensure_ms

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