import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from Model_evaluation.model_utils import _predict_any, _unwrap_estimator_from_mlf
from Model_evaluation.metrics import mse_deviance


def _to_month_start(s):
    s = pd.to_datetime(s, errors="coerce")
    if isinstance(s, pd.Series):
        return s.dt.to_period("M").dt.to_timestamp(how="start").dt.normalize()
    return pd.Timestamp(s).to_period("M").to_timestamp(how="start").normalize()


def _safe_feature_list(df: pd.DataFrame, target_col: str, date_col: str):
    return [c for c in df.columns if c not in {target_col, date_col}]


def _build_eval_windows(
    *,
    exp_results: dict,
    df_all: pd.DataFrame,
    target_col: str,
    date_col: str = "ds",
    h: int = 12,
    restrict_eval_window=None,
):
    df = df_all.copy().sort_values(date_col).reset_index(drop=True)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).reset_index(drop=True)

    features = list(exp_results["features"])
    models = list(exp_results["models"])
    periods = pd.to_datetime(pd.Index(exp_results["train_periods"]), errors="coerce")

    if len(models) != len(periods):
        raise ValueError("exp_results['models'] et exp_results['train_periods'] doivent avoir la même longueur.")

    if restrict_eval_window is not None:
        start_w, end_w = restrict_eval_window
        start_w = pd.Timestamp(start_w)
        end_w = pd.Timestamp(end_w)
    else:
        start_w, end_w = None, None

    rows = []
    for model_obj, train_end in zip(models, periods):
        if pd.isna(train_end):
            continue

        train_end = pd.Timestamp(train_end).to_period("M").to_timestamp(how="start")
        forecast_date = train_end + pd.DateOffset(months=h)

        if start_w is not None and forecast_date < start_w:
            continue
        if end_w is not None and forecast_date > end_w:
            continue

        hit = df.loc[df[date_col] == forecast_date].copy()
        if hit.empty:
            continue

        miss_feats = [f for f in features if f not in hit.columns]
        if miss_feats:
            continue

        y_true = hit.iloc[0][target_col]
        if pd.isna(y_true):
            continue

        X_row = hit[features].copy()
        if X_row.isna().any(axis=None):
            continue

        rows.append(
            {
                "forecast_date": forecast_date,
                "y_true": float(y_true),
                "X_row": X_row.reset_index(drop=True),
                "model_obj": model_obj,
            }
        )

    return rows, features


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
    date_col: str = "ds",
):
    rows, features = _build_eval_windows(
        exp_results=exp_results,
        df_all=df_all,
        target_col=target_col,
        date_col=date_col,
        h=h,
        restrict_eval_window=restrict_eval_window,
    )

    if len(rows) == 0:
        return pd.DataFrame(columns=["feature", "ratio_mean", "ratio_std", "n_windows"])

    y_true = np.array([r["y_true"] for r in rows], dtype=float)

    base_preds = []
    X_stack = []
    for r in rows:
        pred = _predict_any(r["model_obj"], r["X_row"])
        pred = np.asarray(pred).reshape(-1)[0]
        base_preds.append(float(pred))
        X_stack.append(r["X_row"].iloc[0][features].to_dict())

    base_preds = np.array(base_preds, dtype=float)
    X_df = pd.DataFrame(X_stack, columns=features)

    base_metric = float(metric_fn(y_true, base_preds))
    if not np.isfinite(base_metric):
        raise ValueError("Base metric invalide dans perm_ratio_pseudo_oos.")

    rng = np.random.default_rng(random_state)
    out_rows = []

    for feat in features:
        rep_metrics = []

        for _ in range(n_repeats):
            X_perm = X_df.copy()
            X_perm[feat] = rng.permutation(X_perm[feat].to_numpy())

            perm_preds = []
            for i, r in enumerate(rows):
                X_row_perm = pd.DataFrame([X_perm.iloc[i].to_dict()], columns=features)
                pred = _predict_any(r["model_obj"], X_row_perm)
                pred = np.asarray(pred).reshape(-1)[0]
                perm_preds.append(float(pred))

            perm_preds = np.array(perm_preds, dtype=float)
            perm_metric = float(metric_fn(y_true, perm_preds))
            rep_metrics.append(perm_metric)

        rep_metrics = np.array(rep_metrics, dtype=float)
        ratio = rep_metrics / base_metric if base_metric != 0 else np.full_like(rep_metrics, np.nan)

        out_rows.append(
            {
                "feature": feat,
                "ratio_mean": np.nanmean(ratio),
                "ratio_std": np.nanstd(ratio, ddof=0),
                "n_windows": len(rows),
            }
        )

    return (
        pd.DataFrame(out_rows)
        .sort_values("ratio_mean", ascending=False)
        .reset_index(drop=True)
    )


def shap_share_pseudo_oos(
    *,
    exp_results: dict,
    model_key: str | None = None,
):
    models = list(exp_results["models"])
    features = list(exp_results["features"])

    if len(models) == 0:
        return pd.DataFrame(columns=["feature", "shap_share_mean", "shap_share_std", "n_windows"])

    shares = []

    for model_obj in models:
        est = _unwrap_estimator_from_mlf(model_obj, preferred_key=model_key)

        if hasattr(est, "coef_"):
            w = np.abs(np.asarray(est.coef_, dtype=float)).reshape(-1)
        elif hasattr(est, "feature_importances_"):
            w = np.abs(np.asarray(est.feature_importances_, dtype=float)).reshape(-1)
        else:
            continue

        w = w[: len(features)]
        if len(w) != len(features):
            continue

        total = np.nansum(w)
        if total <= 0 or not np.isfinite(total):
            continue

        shares.append(w / total)

    if len(shares) == 0:
        return pd.DataFrame(columns=["feature", "shap_share_mean", "shap_share_std", "n_windows"])

    arr = np.vstack(shares)

    return (
        pd.DataFrame(
            {
                "feature": features,
                "shap_share_mean": np.nanmean(arr, axis=0),
                "shap_share_std": np.nanstd(arr, axis=0, ddof=0),
                "n_windows": arr.shape[0],
            }
        )
        .sort_values("shap_share_mean", ascending=False)
        .reset_index(drop=True)
    )


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
    ts_local = ts_df.copy()

    bkt_local[date_col] = pd.to_datetime(bkt_local[date_col], errors="coerce")
    ts_local[date_col] = pd.to_datetime(ts_local[date_col], errors="coerce")

    if normalize_month_start:
        bkt_local[date_col] = _to_month_start(bkt_local[date_col])
        ts_local[date_col] = _to_month_start(ts_local[date_col])

    bkt_local = bkt_local.dropna(subset=[date_col]).reset_index(drop=True)
    ts_local = ts_local.dropna(subset=[date_col]).reset_index(drop=True)

    partitions = sorted(bkt_local[partition_col].dropna().astype(str).unique().tolist())

    if verbose:
        print("Partitions détectées:", partitions)

    for partition in partitions:
        if verbose:
            print(f"\n==============================")
            print(f"PARTITION: {partition}")
            print(f"==============================")

        dates_part = bkt_local.loc[
            bkt_local[partition_col].astype(str) == partition,
            date_col,
        ]

        if len(dates_part) == 0:
            continue

        start_p = dates_part.min()
        end_p = dates_part.max()

        ts_part = ts_local.loc[
            (ts_local[date_col] >= start_p) & (ts_local[date_col] <= end_p)
        ].copy()

        if verbose:
            print("Rows used:", len(ts_part))

        results_perm_mae_by_part.setdefault(partition, {})
        results_perm_deviance_by_part.setdefault(partition, {})
        results_shap_share_by_part.setdefault(partition, {})

        if ts_part.empty:
            continue

        for model_name, bundle in meta_models["bundles"].items():
            if verbose:
                print(f"→ Computing {model_name} for {partition}")

            if not isinstance(bundle, dict):
                continue
            if not all(k in bundle for k in ["models", "train_fit_dates", "features"]):
                if verbose:
                    print(f"   ⚠️ Bundle incomplet pour {model_name}.")
                continue

            models_all = list(bundle["models"])
            dates_all = pd.to_datetime(pd.Index(bundle["train_fit_dates"]), errors="coerce")
            feats_all = list(bundle["features"])

            valid_pairs = [(m, d) for m, d in zip(models_all, dates_all) if pd.notna(d) and m is not None]
            valid_pairs = [(m, d) for m, d in valid_pairs if d <= end_p]

            if len(valid_pairs) == 0:
                if verbose:
                    print("   ⚠️ Aucun modèle valide pour cette partition.")
                continue

            models_filtered = [m for m, _ in valid_pairs]
            dates_filtered = [d for _, d in valid_pairs]

            exp = {
                "models": models_filtered,
                "features": feats_all,
                "train_periods": dates_filtered,
            }

            try:
                df_perm_mae = perm_ratio_pseudo_oos(
                    exp_results=exp,
                    df_all=ts_part,
                    target_col=target_col,
                    h=horizon,
                    metric_fn=mean_absolute_error,
                    restrict_eval_window=(str(start_p.date()), str(end_p.date())),
                    model_key=model_name,
                    date_col=date_col,
                )
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ perm_mae failed: {e}")
                df_perm_mae = pd.DataFrame(columns=["feature", "ratio_mean", "ratio_std", "n_windows"])

            try:
                df_perm_dev = perm_ratio_pseudo_oos(
                    exp_results=exp,
                    df_all=ts_part,
                    target_col=target_col,
                    h=horizon,
                    metric_fn=mse_deviance,
                    restrict_eval_window=(str(start_p.date()), str(end_p.date())),
                    model_key=model_name,
                    date_col=date_col,
                )
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ perm_dev failed: {e}")
                df_perm_dev = pd.DataFrame(columns=["feature", "ratio_mean", "ratio_std", "n_windows"])

            try:
                df_shap = shap_share_pseudo_oos(
                    exp_results=exp,
                    model_key=model_name,
                )
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ shap_share failed: {e}")
                df_shap = pd.DataFrame(columns=["feature", "shap_share_mean", "shap_share_std", "n_windows"])

            results_perm_mae_by_part[partition][model_name] = df_perm_mae
            results_perm_deviance_by_part[partition][model_name] = df_perm_dev
            results_shap_share_by_part[partition][model_name] = df_shap

    if verbose:
        print("\n✅ Explainability ready by partition")
        print("Partitions calculées:", list(results_perm_mae_by_part.keys()))

    return (
        results_perm_mae_by_part,
        results_perm_deviance_by_part,
        results_shap_share_by_part,
    )


if __name__ == "__main__":
    main()