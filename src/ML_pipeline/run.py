import sys
import webbrowser
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PIPELINE_ROOT.parent

if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from config.config_loader import load_config
from Data_preparation.data_preparation import load_wide_from_feast, make_ts_from_wide
from Model_training.backtesting import run_backtesting_generic
from Model_training.ModelSpec import build_model_specs
from Model_evaluation.explainability import compute_explainability_by_partition
from Model_evaluation.scoring import build_score_and_explainability_tables
from Model_evaluation.plot_evolution import plot_forecast_evolution
from Model_evaluation.functional_plots import (
    build_functional_inputs_from_meta,
    plot_functional_grid,
    plot_residuals_vs_fitted_grid,
)

from Model_registry.mlflow_registry import (
    log_experiment_runs_to_mlflow,
    register_logged_model_to_registry,
    start_mlflow_server_if_needed,
)


# ============================================================
# HELPERS
# ============================================================
def _auto_select_vars_from_last_partition(
    leaderboard_exp,
    results_shap_share_by_part,
    ts_df,
    *,
    target_partition="2020-end",
    top_n=4,
    fallback_model="LGBM",
):
    best_model = fallback_model

    try:
        if isinstance(leaderboard_exp, pd.DataFrame) and len(leaderboard_exp) > 0:
            lb = leaderboard_exp.copy()

            if "partition" in lb.columns:
                lb = lb.loc[lb["partition"].astype(str) == str(target_partition)].copy()

            if len(lb) > 0:
                for model_col in ["model", "model_name", "Model", "models", "model_label"]:
                    if model_col in lb.columns:
                        best_model = str(lb.iloc[0][model_col])
                        break
    except Exception as e:
        print(f"⚠️ Impossible de déterminer le meilleur modèle depuis leaderboard_exp: {e}")

    print(f"Model selected from partition '{target_partition}': {best_model}")

    shap_obj = None
    try:
        if isinstance(results_shap_share_by_part, dict):
            shap_obj = results_shap_share_by_part.get(target_partition, None)
    except Exception as e:
        print(f"⚠️ Impossible de lire results_shap_share_by_part[{target_partition!r}]: {e}")
        shap_obj = None

    if shap_obj is None:
        print(f"⚠️ Aucune entrée SHAP trouvée pour la partition '{target_partition}'.")
        return best_model, []

    shap_df = None
    try:
        if isinstance(shap_obj, dict):
            shap_df = shap_obj.get(best_model, None)

            if shap_df is None:
                for _, v in shap_obj.items():
                    if isinstance(v, pd.DataFrame) and len(v) > 0:
                        shap_df = v.copy()
                        break

        elif isinstance(shap_obj, pd.DataFrame):
            shap_df = shap_obj.copy()
    except Exception as e:
        print(f"⚠️ Impossible de récupérer le DataFrame SHAP pour {best_model}: {e}")
        shap_df = None

    if shap_df is None or not isinstance(shap_df, pd.DataFrame) or len(shap_df) == 0:
        print(
            f"⚠️ Aucun DataFrame SHAP exploitable pour la partition "
            f"'{target_partition}' et le modèle '{best_model}'."
        )
        return best_model, []

    shap_df = shap_df.copy()

    feat_col = None
    score_col = None

    for c in ["feature", "variable", "feat", "Feature"]:
        if c in shap_df.columns:
            feat_col = c
            break

    for c in [
        "shap_share_mean",
        "shap_share",
        "importance",
        "value",
        "score",
        "mean_abs_shap_share",
    ]:
        if c in shap_df.columns:
            score_col = c
            break

    if feat_col is None:
        shap_df = shap_df.reset_index()
        for c in ["index", "feature", "variable", "feat", "Feature"]:
            if c in shap_df.columns:
                feat_col = c
                break

    if feat_col is None or score_col is None:
        print("⚠️ Colonnes feature/score introuvables dans le DataFrame SHAP.")
        print("Colonnes disponibles:", shap_df.columns.tolist())
        return best_model, []

    try:
        shap_df = shap_df.sort_values(score_col, ascending=False)
    except Exception as e:
        print(f"⚠️ Impossible de trier le DataFrame SHAP sur '{score_col}': {e}")
        return best_model, []

    selected_vars = []
    ts_cols = set(ts_df.columns.astype(str).tolist())

    for feat in shap_df[feat_col].astype(str).tolist():
        if feat in ts_cols and feat not in selected_vars:
            selected_vars.append(feat)
        if len(selected_vars) >= top_n:
            break

    print(f"Selected vars from '{target_partition}' / {best_model}: {selected_vars}")
    return best_model, selected_vars


def _build_long_evolution_df(
    bkt_wide: pd.DataFrame,
    *,
    candidate_models=None,
    date_col: str = "ds",
    y_true_candidates=("y", "y_true", "target", "actual", "observed"),
):
    df = bkt_wide.copy()

    if date_col not in df.columns:
        raise ValueError(f"Colonne date '{date_col}' absente.")

    y_true_col = None
    for c in y_true_candidates:
        if c in df.columns:
            y_true_col = c
            break

    if y_true_col is None:
        raise ValueError(
            f"Aucune colonne y_true trouvée parmi {list(y_true_candidates)}. "
            f"Colonnes disponibles: {df.columns.tolist()}"
        )

    if candidate_models is None:
        candidate_models = ["AR", "LR", "RIDGE", "LGBM"]

    candidate_models = [str(m).upper() for m in candidate_models]

    rows = []
    for model in candidate_models:
        pred_col = None
        pred_candidates = [
            model,
            f"{model}_forecast",
            f"{model}_pred",
            f"{model}_y_hat",
            f"y_hat_{model}",
        ]
        for c in pred_candidates:
            if c in df.columns:
                pred_col = c
                break

        if pred_col is None:
            continue

        lo_col = None
        hi_col = None

        lo_candidates = [
            f"{model}-lo-95",
            f"{model}_lo_95",
            f"{model}_lo",
            f"y_lo_{model}",
        ]
        hi_candidates = [
            f"{model}-hi-95",
            f"{model}_hi_95",
            f"{model}_hi",
            f"y_hi_{model}",
        ]

        for c in lo_candidates:
            if c in df.columns:
                lo_col = c
                break

        for c in hi_candidates:
            if c in df.columns:
                hi_col = c
                break

        tmp_cols = [date_col, y_true_col, pred_col]
        if "partition" in df.columns:
            tmp_cols.append("partition")
        if lo_col is not None:
            tmp_cols.append(lo_col)
        if hi_col is not None:
            tmp_cols.append(hi_col)

        tmp = df[tmp_cols].copy()
        tmp = tmp.rename(
            columns={
                date_col: "ds",
                y_true_col: "y_true",
                pred_col: "y_hat",
            }
        )

        if lo_col is not None:
            tmp = tmp.rename(columns={lo_col: "y_lo"})
        if hi_col is not None:
            tmp = tmp.rename(columns={hi_col: "y_hi"})

        tmp["model_label"] = model
        rows.append(tmp)

    if len(rows) == 0:
        raise ValueError(
            "Aucun modèle détecté dans bkt_models_final. "
            f"Colonnes disponibles: {df.columns.tolist()}"
        )

    out = pd.concat(rows, axis=0, ignore_index=True)
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    out = out.dropna(subset=["ds", "y_true", "y_hat"]).sort_values(["ds", "model_label"])

    if "partition" not in out.columns:
        out["partition"] = "ALL"

    return out


def _extract_shap_long(results_shap_share_by_part) -> pd.DataFrame:
    rows = []

    if not isinstance(results_shap_share_by_part, dict):
        return pd.DataFrame(
            columns=["partition", "model_label", "feature", "shap_share_mean", "shap_share_std", "n_windows"]
        )

    for partition, obj in results_shap_share_by_part.items():
        if isinstance(obj, dict):
            for model_label, df in obj.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue

                tmp = df.copy()
                tmp["partition"] = str(partition)
                tmp["model_label"] = str(model_label)

                if "feature" not in tmp.columns:
                    tmp = tmp.reset_index()
                    if "index" in tmp.columns and "feature" not in tmp.columns:
                        tmp = tmp.rename(columns={"index": "feature"})

                if "shap_share_mean" not in tmp.columns:
                    if "shap_share" in tmp.columns:
                        tmp = tmp.rename(columns={"shap_share": "shap_share_mean"})
                    elif "value" in tmp.columns:
                        tmp = tmp.rename(columns={"value": "shap_share_mean"})

                if "shap_share_std" not in tmp.columns:
                    tmp["shap_share_std"] = pd.NA

                if "n_windows" not in tmp.columns:
                    tmp["n_windows"] = pd.NA

                keep_cols = [
                    "partition", "model_label", "feature",
                    "shap_share_mean", "shap_share_std", "n_windows"
                ]
                for c in keep_cols:
                    if c not in tmp.columns:
                        tmp[c] = pd.NA

                rows.append(tmp[keep_cols])

        elif isinstance(obj, pd.DataFrame) and not obj.empty:
            tmp = obj.copy()
            tmp["partition"] = str(partition)

            if "model_label" not in tmp.columns:
                tmp["model_label"] = "UNKNOWN"

            if "feature" not in tmp.columns:
                tmp = tmp.reset_index()
                if "index" in tmp.columns and "feature" not in tmp.columns:
                    tmp = tmp.rename(columns={"index": "feature"})

            if "shap_share_mean" not in tmp.columns:
                if "shap_share" in tmp.columns:
                    tmp = tmp.rename(columns={"shap_share": "shap_share_mean"})
                elif "value" in tmp.columns:
                    tmp = tmp.rename(columns={"value": "shap_share_mean"})

            if "shap_share_std" not in tmp.columns:
                tmp["shap_share_std"] = pd.NA

            if "n_windows" not in tmp.columns:
                tmp["n_windows"] = pd.NA

            keep_cols = [
                "partition", "model_label", "feature",
                "shap_share_mean", "shap_share_std", "n_windows"
            ]
            for c in keep_cols:
                if c not in tmp.columns:
                    tmp[c] = pd.NA

            rows.append(tmp[keep_cols])

    if not rows:
        return pd.DataFrame(
            columns=["partition", "model_label", "feature", "shap_share_mean", "shap_share_std", "n_windows"]
        )

    out = pd.concat(rows, ignore_index=True)
    out["partition"] = out["partition"].astype(str)
    out["model_label"] = out["model_label"].astype(str)
    out["feature"] = out["feature"].astype(str)
    out["shap_share_mean"] = pd.to_numeric(out["shap_share_mean"], errors="coerce")
    out["shap_share_std"] = pd.to_numeric(out["shap_share_std"], errors="coerce")
    return out


def _build_shap_tables(results_shap_share_by_part, top_k: int = 5):
    shap_long = _extract_shap_long(results_shap_share_by_part)

    if shap_long.empty:
        top5_pivot = pd.DataFrame(
            columns=["model", "partition"] + [f"Rank_{i}" for i in range(1, top_k + 1)]
        )
        return shap_long, top5_pivot

    shap_long = shap_long.sort_values(
        ["partition", "model_label", "shap_share_mean"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    top5_long = (
        shap_long.groupby(["partition", "model_label"], group_keys=False)
        .head(top_k)
        .copy()
    )

    top5_long["rank"] = (
        top5_long.groupby(["partition", "model_label"])["shap_share_mean"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    top5_long["feature_with_pct"] = top5_long.apply(
        lambda r: f"{r['feature']} ({r['shap_share_mean'] * 100:.2f}%)"
        if pd.notna(r["shap_share_mean"])
        else str(r["feature"]),
        axis=1,
    )

    top5_pivot = (
        top5_long.pivot_table(
            index=["model_label", "partition"],
            columns="rank",
            values="feature_with_pct",
            aggfunc="first",
        )
        .reset_index()
    )

    top5_pivot = top5_pivot.rename(columns={"model_label": "model"})
    top5_pivot.columns.name = None

    rename_map = {}
    for c in top5_pivot.columns:
        if isinstance(c, int):
            rename_map[c] = f"Rank_{c}"
    top5_pivot = top5_pivot.rename(columns=rename_map)

    wanted_cols = ["model", "partition"] + [f"Rank_{i}" for i in range(1, top_k + 1)]
    for c in wanted_cols:
        if c not in top5_pivot.columns:
            top5_pivot[c] = pd.NA

    top5_pivot = top5_pivot[wanted_cols].copy()

    partition_order = {
        "1990-1999": 0,
        "2000-2008": 1,
        "2009-2019": 2,
        "2020-end": 3,
        "ALL": 4,
    }
    model_order = {
        "LGBM": 0,
        "LR": 1,
        "RIDGE": 2,
        "AR": 3,
    }

    top5_pivot["_model_order"] = top5_pivot["model"].map(model_order).fillna(999)
    top5_pivot["_part_order"] = top5_pivot["partition"].map(partition_order).fillna(999)

    top5_pivot = (
        top5_pivot.sort_values(["_model_order", "_part_order", "model", "partition"])
        .drop(columns=["_model_order", "_part_order"])
        .reset_index(drop=True)
    )

    return shap_long, top5_pivot


def _export_shap_html_report(
    results_shap_share_by_part,
    output_html,
    *,
    title: str = "SHAP Explainability Report",
    top_k: int = 5,
):
    shap_long, top5_pivot = _build_shap_tables(
        results_shap_share_by_part,
        top_k=top_k,
    )

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    if top5_pivot.empty:
        html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 30px; background: #111; color: #eee; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>Aucune donnée SHAP disponible.</p>
        </body>
        </html>
        """
        output_html.write_text(html, encoding="utf-8")
        return shap_long, top5_pivot, str(output_html)

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 24px;
                background: #0f0f0f;
                color: #f1f1f1;
            }}
            h1, h2 {{
                color: #ffffff;
            }}
            .block {{
                margin-bottom: 36px;
                padding: 18px;
                background: #171717;
                border-radius: 12px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                font-size: 14px;
                background: #111;
            }}
            th, td {{
                border: 1px solid #333;
                padding: 8px 10px;
                text-align: left;
                white-space: nowrap;
            }}
            th {{
                background: #222;
                position: sticky;
                top: 0;
            }}
            tr:nth-child(even) {{
                background: #161616;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>

        <div class="block">
            <h2>Top {top_k} most explanatory features by period</h2>
            {top5_pivot.to_html(index=True, escape=False)}
        </div>
    </body>
    </html>
    """

    output_html.write_text(html, encoding="utf-8")
    return shap_long, top5_pivot, str(output_html)


def _extract_mlflow_objects_from_meta(meta_models):
    bundles = meta_models.get("bundles", {}) if isinstance(meta_models, dict) else {}

    fitted_models_for_mlflow = {}
    train_fit_dates_for_mlflow = {}

    if not isinstance(bundles, dict):
        return None, None

    for model_label, bundle in bundles.items():
        if not isinstance(bundle, dict):
            continue

        models_obj = bundle.get("models", None)
        fit_dates_obj = bundle.get("train_fit_dates", None)

        model_for_log = None

        if isinstance(models_obj, (list, tuple)):
            non_null_models = [m for m in models_obj if m is not None]
            if len(non_null_models) > 0:
                model_for_log = non_null_models[-1]
        elif models_obj is not None:
            model_for_log = models_obj

        if model_for_log is not None:
            fitted_models_for_mlflow[str(model_label)] = model_for_log

        if fit_dates_obj is not None:
            train_fit_dates_for_mlflow[str(model_label)] = fit_dates_obj

    if len(fitted_models_for_mlflow) == 0:
        fitted_models_for_mlflow = None

    if len(train_fit_dates_for_mlflow) == 0:
        train_fit_dates_for_mlflow = None

    return fitted_models_for_mlflow, train_fit_dates_for_mlflow


def _extract_model_intervals_from_bkt(
    bkt_df: pd.DataFrame,
    *,
    model_label: str,
    partition: str | None = None,
    date_col: str = "ds",
):
    if not isinstance(bkt_df, pd.DataFrame) or bkt_df.empty:
        return None

    df = bkt_df.copy()

    if partition is not None and "partition" in df.columns:
        df = df.loc[df["partition"].astype(str) == str(partition)].copy()

    if df.empty or date_col not in df.columns:
        return None

    model = str(model_label).upper()

    pred_candidates = [
        model,
        f"{model}_forecast",
        f"{model}_pred",
        f"{model}_y_hat",
        f"y_hat_{model}",
    ]
    lo_candidates = [
        f"{model}-lo-95",
        f"{model}_lo_95",
        f"{model}_lo",
        f"y_lo_{model}",
    ]
    hi_candidates = [
        f"{model}-hi-95",
        f"{model}_hi_95",
        f"{model}_hi",
        f"y_hi_{model}",
    ]

    pred_col = next((c for c in pred_candidates if c in df.columns), None)
    lo_col = next((c for c in lo_candidates if c in df.columns), None)
    hi_col = next((c for c in hi_candidates if c in df.columns), None)

    if lo_col is None or hi_col is None:
        return None

    out_cols = [date_col, lo_col, hi_col]
    if pred_col is not None:
        out_cols.append(pred_col)
    if "y" in df.columns:
        out_cols.append("y")
    if "partition" in df.columns:
        out_cols.append("partition")

    out = df[out_cols].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).reset_index(drop=True)

    rename_map = {
        date_col: "ds",
        lo_col: "lower",
        hi_col: "upper",
    }
    if pred_col is not None:
        rename_map[pred_col] = "forecast"
    if "y" in out.columns:
        rename_map["y"] = "y_true"

    out = out.rename(columns=rename_map)
    return out


def _build_conformal_artifacts_from_backtest(
    bkt_models_final: pd.DataFrame,
    *,
    model_labels=("RIDGE", "LGBM"),
    partitions=("1990-1999", "2000-2008", "2009-2019", "2020-end"),
):
    conformal_artifacts = {}

    for model_label in model_labels:
        for partition in partitions:
            intervals_df = _extract_model_intervals_from_bkt(
                bkt_models_final,
                model_label=model_label,
                partition=partition,
                date_col="ds",
            )

            if intervals_df is None or intervals_df.empty:
                continue

            conformal_artifacts[(str(model_label), str(partition))] = {
                "intervals_df": intervals_df,
                "meta": {
                    "source": "backtest_existing_intervals",
                    "method": "logged_from_backtest",
                },
            }

    return conformal_artifacts if len(conformal_artifacts) else None


def _prepare_mlflow_logging_payloads(
    *,
    fig_evolution=None,
    fig_functional=None,
    fig_residuals=None,
    X_dict=None,
    features_dict=None,
    partitions=("2020-end",),
    model_keys=("RIDGE", "LGBM"),
):
    extra_figures = {}
    X_by_partition = {}
    features_by_partition = {}

    for part in partitions:
        for model_key in model_keys:
            run_key = (str(model_key), str(part))
            extra_figures[run_key] = {}

            if fig_evolution is not None:
                extra_figures[run_key][f"forecast_evolution_{model_key}_{part}.png"] = fig_evolution

            if fig_functional is not None:
                extra_figures[run_key][f"functional_grid_{model_key}_{part}.png"] = fig_functional

            if fig_residuals is not None:
                extra_figures[run_key][f"residuals_vs_fitted_{model_key}_{part}.png"] = fig_residuals

            if isinstance(X_dict, dict) and model_key in X_dict and X_dict[model_key] is not None:
                X_by_partition[run_key] = X_dict[model_key]

            if isinstance(features_dict, dict) and model_key in features_dict and features_dict[model_key] is not None:
                features_by_partition[run_key] = features_dict[model_key]

    if len(extra_figures) == 0:
        extra_figures = None
    if len(X_by_partition) == 0:
        X_by_partition = None
    if len(features_by_partition) == 0:
        features_by_partition = None

    return extra_figures, X_by_partition, features_by_partition


# ============================================================
# MAIN
# ============================================================
def main():
    requirement_path = PIPELINE_ROOT / "config" / "requirement.json"
    model_settings_path = PIPELINE_ROOT / "config" / "model_settings.json"

    config = load_config(requirement_path)
    model_config = load_config(model_settings_path)

    model_specs = build_model_specs(model_config)

    series_ids = config["data"]["series_ids"]
    start = config["data"]["start"]
    end = config["data"]["end"]

    raw_target_col = config["data"]["target"]
    pipeline_target_col = "y"

    lags = config["feature_engineering"]["lags"]
    freq = config["forecast"]["frequency"]

    if isinstance(lags, (list, tuple)) and len(lags) > 0:
        main_lag = int(lags[0])
    else:
        main_lag = int(lags)

    print("\nLoading data from Feast...\n")

    df_wide = load_wide_from_feast(
        feature_ref="stationary_value:value",
        series_ids=series_ids,
        start=start,
        end=end,
        freq=freq,
    )

    print("df_wide shape:", df_wide.shape)
    print("df_wide max date:", df_wide.index.max())

    print("\nBuilding features...\n")

    ts_df, exog_cols = make_ts_from_wide(
        df_wide=df_wide,
        target_col=raw_target_col,
        lags=lags,
        include_target_lags=True,
    )

    print("ts_df shape:", ts_df.shape)
    print("n exog:", len(exog_cols))

    print("\n--- LAST ROWS ---")
    print(ts_df.tail(4))

    print("\n--- COLUMNS ---")
    print(ts_df.columns.tolist())

    print("\nDate min:", ts_df["ds"].min())
    print("Date max:", ts_df["ds"].max())

    h = config["forecast"]["horizon"]
    step_size = config["forecast"]["step_size"]

    pi_windows = config["prediction_intervals"]["n_windows"]
    levels = config["prediction_intervals"]["levels"]

    exp_start = pd.Timestamp(config["experiment_period"]["start"])
    exp_end = pd.Timestamp(config["experiment_period"]["end"])

    cv_method = config["cross_validation"]["method"]
    hv_gap = config["cross_validation"]["hv_gap"]
    kfold_n_splits = config["cross_validation"]["kfold_n_splits"]

    seed = config["reproducibility"]["seed"]

    min_train_n = config["training"]["min_train_n"]
    train_window_type = config["training"]["train_window_type"]
    rolling_window_size = config["training"]["rolling_window_size"]

    print("\nRunning backtesting...\n")

    bkt_models_final, meta_models = run_backtesting_generic(
        ts=ts_df,
        model_specs=model_specs,
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
        features=exog_cols,
        train_window_type=train_window_type,
        rolling_window_size=rolling_window_size,
    )

    print("\nBacktesting DONE\n")
    print("bkt_models_final shape:", bkt_models_final.shape)

    print("\n--- HEAD ---")
    print(bkt_models_final.head())

    print("\n--- META KEYS ---")
    print(meta_models.keys())

    print("\n================ DEBUG OUTPUT ================\n")

    print("COLUMNS:")
    print(bkt_models_final.columns.tolist())

    print("\nMODELS DETECTED:")
    print("RIDGE cols:", [c for c in bkt_models_final.columns if "RIDGE" in c])
    print("LGBM cols:", [c for c in bkt_models_final.columns if "LGBM" in c])

    pd.set_option("display.max_columns", None)
    print("\nHEAD (FULL):")
    print(bkt_models_final.head())

    print("\nNULL CHECK:")
    print(bkt_models_final.isna().sum())

    print("\nDATE RANGE:")
    print("min ds:", bkt_models_final["ds"].min())
    print("max ds:", bkt_models_final["ds"].max())

    if "RIDGE_tune_mae" in bkt_models_final.columns:
        print("\nRIDGE tune mae mean:", bkt_models_final["RIDGE_tune_mae"].mean())

    if "LGBM_tune_mae" in bkt_models_final.columns:
        print("LGBM tune mae mean:", bkt_models_final["LGBM_tune_mae"].mean())

    print("\n=============================================\n")

    print("\nBuilding partitions...\n")

    bkt_models_final = bkt_models_final.copy()
    bkt_models_final["ds"] = pd.to_datetime(bkt_models_final["ds"], errors="coerce")

    bkt_models_final["partition"] = pd.cut(
        bkt_models_final["ds"],
        bins=[
            pd.Timestamp("1990-01-01"),
            pd.Timestamp("2000-01-01"),
            pd.Timestamp("2009-01-01"),
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2100-01-01"),
        ],
        labels=[
            "1990-1999",
            "2000-2008",
            "2009-2019",
            "2020-end",
        ],
        right=False,
    )

    print(
        "Partitions found:",
        sorted(bkt_models_final["partition"].dropna().astype(str).unique().tolist())
    )

    print("\nRunning explainability...\n")

    (
        results_perm_mae_by_part,
        results_perm_deviance_by_part,
        results_shap_share_by_part,
    ) = compute_explainability_by_partition(
        bkt_score=bkt_models_final,
        ts_df=ts_df,
        meta_models=meta_models,
        target_col=pipeline_target_col,
        date_col="ds",
        partition_col="partition",
        horizon=h,
        normalize_month_start=True,
        verbose=True,
    )

    print("\nBuilding scoring + explainability tables...\n")

    long_sc2, score_df, score_df_exp, leaderboard_exp = build_score_and_explainability_tables(
        bkt_score=bkt_models_final,
        results_perm_mae_by_part=results_perm_mae_by_part,
        results_perm_deviance_by_part=results_perm_deviance_by_part,
        results_shap_share_by_part=results_shap_share_by_part,
        models=["RIDGE", "LGBM"],
        target_col=pipeline_target_col,
        partition_col="partition",
        top_k=2,
        verbose=True,
    )

    print("\n=== FINAL LEADERBOARD ===")
    print(leaderboard_exp)

    print("\nBuilding SHAP HTML report...\n")

    shap_long_all = None
    shap_top5_table = None
    shap_report_path = None

    try:
        output_dir = PIPELINE_ROOT / "outputs"
        shap_long_all, shap_top5_table, shap_report_path = _export_shap_html_report(
            results_shap_share_by_part=results_shap_share_by_part,
            output_html=output_dir / "shap_explainability_report.html",
            title=f"SHAP Explainability Report — {raw_target_col}",
            top_k=5,
        )
        print(f"✅ SHAP report saved to: {shap_report_path}")
        webbrowser.open_new_tab(Path(shap_report_path).resolve().as_uri())
    except Exception as e:
        print(f"⚠️ SHAP HTML report skipped: {e}")

    print("\nBuilding global evolution plot...\n")

    fig_evolution = None
    metrics_by_part = None
    evolution_df = None

    try:
        candidate_models = ["RIDGE", "LGBM"]

        evolution_df = _build_long_evolution_df(
            bkt_wide=bkt_models_final,
            candidate_models=candidate_models,
            date_col="ds",
            y_true_candidates=("y", "y_true"),
        )

        detected_models = tuple(
            evolution_df["model_label"].dropna().astype(str).unique().tolist()
        )
        print("Detected models for evolution plot:", detected_models)

        fig_evolution, metrics_by_part = plot_forecast_evolution(
            combined_bkt=evolution_df,
            series_name=raw_target_col,
            partition=None,
            models=detected_models,
            show_partition_lines=True,
            annotate_partition_metrics=True,
            annotate_fontsize=10,
        )
    except Exception as e:
        print(f"⚠️ Global evolution plot skipped: {e}")

    print("\nBuilding functional plots...\n")

    _pretty_label_from_feat = {
        f"SP500_lag{main_lag}": "S&P 500",
        f"INDPRO_lag{main_lag}": "Industrial production",
        f"TB3MS_lag{main_lag}": "3M Treasury bill",
        f"BUSLOANS_lag{main_lag}": "Business loans",
        f"DPCERA3M086SBEA_lag{main_lag}": "Consumption",
        f"RPI_lag{main_lag}": "Real personal income",
        f"M2SL_lag{main_lag}": "Money supply (M2)",
        f"CPIAUCSL_lag{main_lag}": "CPI",
        f"OILPRICEX_lag{main_lag}": "Oil price",
        f"UNRATE_lag{main_lag}": f"Unemployment lag {main_lag}",
    }

    best_model_for_plot, selected_vars = _auto_select_vars_from_last_partition(
        leaderboard_exp=leaderboard_exp,
        results_shap_share_by_part=results_shap_share_by_part,
        ts_df=ts_df,
        target_partition="2020-end",
        top_n=4,
        fallback_model="LGBM",
    )

    print("Selected vars for functional plots:", selected_vars)

    fig = None
    axes = None
    models_dict = None
    X_dict = None
    features_dict = None

    if len(selected_vars) == 0:
        print("No matching selected_vars found from last partition. Functional plots skipped.")
    else:
        functional_start_date = "1990-01-01"
        functional_end_date = "2019-08-01"

        functional_model_order = ("RIDGE", "LGBM")

        models_dict, X_dict, features_dict = build_functional_inputs_from_meta(
            meta_models=meta_models,
            ts_data=ts_df,
            end_date=functional_end_date,
            date_col="ds",
            model_order=functional_model_order,
        )

        print("Models used for functional plots:", list(models_dict.keys()))
        print("Variables used for functional plots:", selected_vars)

        fig, axes = plot_functional_grid(
            models_dict=models_dict,
            X_dict=X_dict,
            selected_vars=selected_vars,
            start_date=functional_start_date,
            end_date=functional_end_date,
            date_col="ds",
            poly_deg_lgbm=3,
            figsize_per_cell=(3.1, 2.7),
            sharey=False,
            max_n=2000,
            random_state=42,
            _pretty_label_from_feat=_pretty_label_from_feat,
            use_pretty_labels=True,
        )

    print("\nBuilding residuals vs fitted plot...\n")

    fig_residuals = None
    axes_residuals = None

    try:
        if evolution_df is None:
            evolution_df = _build_long_evolution_df(
                bkt_wide=bkt_models_final,
                candidate_models=["RIDGE", "LGBM"],
                date_col="ds",
                y_true_candidates=("y", "y_true"),
            )

        fig_residuals, axes_residuals = plot_residuals_vs_fitted_grid(
            combined_bkt=evolution_df,
            date_col="ds",
            y_true_col="y_true",
            y_hat_col="y_hat",
            model_col="model_label",
            start_date="1990-01-01",
            threshold=5.0,
            fitted_low_threshold=-3.0,
            lowess_frac=0.30,
            figsize_per_cell=(6.0, 5.0),
            n_cols=2,
        )

    except Exception as e:
        print(f"⚠️ Residuals vs fitted plot skipped: {e}")

    print("\nExtracting conformal artifacts from backtest...\n")

    conformal_artifacts = None
    try:
        conformal_artifacts = _build_conformal_artifacts_from_backtest(
            bkt_models_final=bkt_models_final,
            model_labels=("RIDGE", "LGBM"),
            partitions=("1990-1999", "2000-2008", "2009-2019", "2020-end"),
        )

        if conformal_artifacts is None:
            print("⚠️ No conformal intervals found in backtest columns.")
        else:
            print("✅ Conformal artifacts prepared for keys:")
            print(list(conformal_artifacts.keys()))
    except Exception as e:
        print(f"⚠️ Conformal extraction skipped: {e}")
        conformal_artifacts = None

    print("\nPreparing MLflow figure/function payloads...\n")

    extra_figures, X_by_partition, features_by_partition = _prepare_mlflow_logging_payloads(
        fig_evolution=fig_evolution,
        fig_functional=fig,
        fig_residuals=fig_residuals,
        X_dict=X_dict,
        features_dict=features_dict,
        partitions=("2020-end",),
        model_keys=("RIDGE", "LGBM"),
    )

    print("extra_figures is None:", extra_figures is None)
    print("X_by_partition is None:", X_by_partition is None)
    print("features_by_partition is None:", features_by_partition is None)

    if isinstance(extra_figures, dict):
        print("extra_figures keys:", list(extra_figures.keys()))
    if isinstance(X_by_partition, dict):
        print("X_by_partition keys:", list(X_by_partition.keys()))
    if isinstance(features_by_partition, dict):
        print("features_by_partition keys:", list(features_by_partition.keys()))

    print("\nExporting to MLflow...\n")

    run_refs = None
    registry_info = None

    try:
        tracking_uri = "http://127.0.0.1:5000"
        experiment_name = "Robustness_Test_Data_Simulation"

        bundles = meta_models.get("bundles", {}) if isinstance(meta_models, dict) else {}
        print("meta_models keys:", meta_models.keys() if isinstance(meta_models, dict) else type(meta_models))
        print("bundles keys:", bundles.keys() if isinstance(bundles, dict) else type(bundles))

        if isinstance(bundles, dict) and "RIDGE" in bundles and isinstance(bundles["RIDGE"], dict):
            print("RIDGE bundle keys:", bundles["RIDGE"].keys())
            print("RIDGE models type:", type(bundles["RIDGE"].get("models", None)))
        if isinstance(bundles, dict) and "LGBM" in bundles and isinstance(bundles["LGBM"], dict):
            print("LGBM bundle keys:", bundles["LGBM"].keys())
            print("LGBM models type:", type(bundles["LGBM"].get("models", None)))

        start_mlflow_server_if_needed(
            tracking_uri=tracking_uri,
            backend_store_uri="sqlite:///mlflow.db",
            default_artifact_root="./mlruns",
            host="127.0.0.1",
            port="5000",
            wait_seconds=15,
        )

        fitted_models_for_mlflow, train_fit_dates_for_mlflow = _extract_mlflow_objects_from_meta(meta_models)

        print("fitted_models_for_mlflow is None:", fitted_models_for_mlflow is None)
        print("train_fit_dates_for_mlflow is None:", train_fit_dates_for_mlflow is None)

        if isinstance(fitted_models_for_mlflow, dict):
            print("fitted_models_for_mlflow keys:", fitted_models_for_mlflow.keys())
            for k, v in fitted_models_for_mlflow.items():
                print(f"model for MLflow [{k}] type:", type(v))

        if isinstance(train_fit_dates_for_mlflow, dict):
            print("train_fit_dates_for_mlflow keys:", train_fit_dates_for_mlflow.keys())

        run_refs = log_experiment_runs_to_mlflow(
            score_df_exp=score_df_exp,
            leaderboard_exp=leaderboard_exp,
            bkt_score=bkt_models_final,
            meta_models=meta_models,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            feast_feature_name="stationary_value:value",
            ts_features=exog_cols,
            results_perm_mae_by_part=results_perm_mae_by_part,
            results_perm_deviance_by_part=results_perm_deviance_by_part,
            results_shap_share_by_part=results_shap_share_by_part,
            fitted_models=fitted_models_for_mlflow,
            train_fit_dates=train_fit_dates_for_mlflow,
            X_by_partition=X_by_partition,
            features_by_partition=features_by_partition,
            conformal_artifacts=conformal_artifacts,
            conformal_alpha=0.05,
            extra_figures=extra_figures,
            run_tags={"project": raw_target_col},
            run_name_fn=lambda row: f"{row['model_label']} | {row['partition']}",
            tmp_root="mlflow_tmp",
            log_mlflow_dataset=True,
            log_mlflow_model=True,
            source_dataset_df=ts_df,
            source_dataset_name="ts_df_stationary",
            target_col=pipeline_target_col,
            n_lags=main_lag,
            exog_cols=exog_cols,
            return_run_refs=True,
        )

        print("✅ MLflow export done")
        print("run_refs:", run_refs)

        model_to_register = ("LGBM", "2020-end")

        if run_refs is not None and model_to_register in run_refs:
            ref = run_refs[model_to_register]

            if ref["artifact_path"] is not None:
                registry_info = register_logged_model_to_registry(
                    tracking_uri=tracking_uri,
                    run_id=ref["run_id"],
                    artifact_path=ref["artifact_path"],
                    registered_model_name="unrate_forecast_model",
                    description="Forecast model registered from pipeline run",
                    tags={
                        "model_label": model_to_register[0],
                        "partition": model_to_register[1],
                    },
                    alias="champion",
                )
                print("✅ Registry done:", registry_info)
            else:
                print("⚠️ No logged model artifact found for registry.")
        else:
            print("⚠️ Requested model not found in run_refs.")

    except Exception as e:
        print(f"⚠️ MLflow / Registry skipped: {e}")

    if plt.get_fignums():
        plt.show()

    return (
        bkt_models_final,
        meta_models,
        score_df,
        score_df_exp,
        leaderboard_exp,
        results_perm_mae_by_part,
        results_perm_deviance_by_part,
        results_shap_share_by_part,
        shap_long_all,
        shap_top5_table,
        shap_report_path,
        fig_evolution,
        metrics_by_part,
        fig,
        axes,
        models_dict,
        X_dict,
        features_dict,
        fig_residuals,
        axes_residuals,
        conformal_artifacts,
        run_refs,
        registry_info,
    )


# ============================================================
# ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    main()