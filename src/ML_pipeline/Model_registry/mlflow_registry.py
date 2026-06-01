import os
import io
import json
import pickle
import joblib
import shutil
import logging
import warnings
import time
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import subprocess
from urllib.request import urlopen

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
# MLflow server bootstrap
# =========================================================
def start_mlflow_server_if_needed(
    tracking_uri="http://127.0.0.1:5000",
    backend_store_uri="sqlite:///mlflow.db",
    default_artifact_root="./mlruns",
    host="127.0.0.1",
    port="5000",
    wait_seconds=15,
):
    def _is_server_up(uri):
        try:
            with urlopen(uri, timeout=2):
                return True
        except Exception:
            return False

    if _is_server_up(tracking_uri):
        print(f"✅ MLflow server already running at {tracking_uri}")
        return {
            "started": False,
            "tracking_uri": tracking_uri,
            "backend_store_uri": backend_store_uri,
            "default_artifact_root": default_artifact_root,
        }

    print(f"🚀 Starting MLflow server at {tracking_uri} ...")

    subprocess.Popen(
        [
            "mlflow",
            "server",
            "--backend-store-uri", backend_store_uri,
            "--default-artifact-root", default_artifact_root,
            "--host", str(host),
            "--port", str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(Path.cwd()),
        shell=False,
    )

    for _ in range(int(wait_seconds)):
        if _is_server_up(tracking_uri):
            print(f"✅ MLflow server started at {tracking_uri}")
            return {
                "started": True,
                "tracking_uri": tracking_uri,
                "backend_store_uri": backend_store_uri,
                "default_artifact_root": default_artifact_root,
            }
        time.sleep(1)

    print(f"⚠️ MLflow server launch requested but not reachable yet at {tracking_uri}")
    return {
        "started": True,
        "tracking_uri": tracking_uri,
        "backend_store_uri": backend_store_uri,
        "default_artifact_root": default_artifact_root,
        "warning": "server_not_reachable_yet",
    }


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


def _coerce_conformal_residuals_df(obj):
    if obj is None:
        return None

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    elif isinstance(obj, (list, tuple, np.ndarray)):
        arr = np.asarray(obj).reshape(-1)
        df = pd.DataFrame({"abs_residual": arr})
    elif isinstance(obj, dict):
        if "residuals_df" in obj and isinstance(obj["residuals_df"], pd.DataFrame):
            df = obj["residuals_df"].copy()
        elif "abs_residual" in obj:
            vals = np.asarray(obj["abs_residual"]).reshape(-1)
            df = pd.DataFrame({"abs_residual": vals})
        else:
            return None
    else:
        return None

    if len(df) == 0:
        return None

    if "abs_residual" not in df.columns:
        # essaye une conversion simple si une seule colonne
        if df.shape[1] == 1:
            df = df.copy()
            df.columns = ["abs_residual"]
        else:
            return None

    df["abs_residual"] = pd.to_numeric(df["abs_residual"], errors="coerce")
    df = df.dropna(subset=["abs_residual"]).reset_index(drop=True)

    return df if len(df) else None


def _coerce_conformal_intervals_df(obj):
    if obj is None:
        return None
    if not isinstance(obj, pd.DataFrame):
        return None

    df = obj.copy()
    required_any = [
        {"ds", "lower", "upper"},
        {"ds", "lo_95", "hi_95"},
        {"ds", "forecast", "lower", "upper"},
        {"ds", "y_hat", "lower", "upper"},
    ]

    cols = set(df.columns)
    if not any(req.issubset(cols) for req in required_any):
        return None

    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df["ds"] = df["ds"].dt.to_period("M").dt.to_timestamp(how="start")

    return df.reset_index(drop=True)


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
    _ensure_dir(snapshot_dir)

    df_to_save = df_dataset.copy()
    if "ds" in df_to_save.columns:
        df_to_save["ds"] = pd.to_datetime(df_to_save["ds"], errors="coerce")

    snapshot_path = os.path.join(snapshot_dir, f"{snapshot_name}.parquet")
    df_to_save.to_parquet(snapshot_path, index=False)

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
    try:
        if df_dataset is None or not isinstance(df_dataset, pd.DataFrame) or len(df_dataset) == 0:
            return False, "Empty or invalid source_dataset_df", None

        ds_df = df_dataset.copy()

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

        dataset = mlflow.data.from_pandas(
            ds_df,
            source=snapshot_uri,
            name=dataset_name,
        )

        mlflow.log_input(dataset, context=context)
        mlflow.log_artifact(snapshot_path, artifact_path="dataset_snapshot")

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
    conformal_artifacts=None,           # NEW
    conformal_alpha=0.05,               # NEW
    extra_figures=None,
    run_tags=None,
    run_name_fn=None,
    tmp_root="mlflow_tmp",
    log_mlflow_dataset=True,
    log_mlflow_model=True,
    source_dataset_df=None,
    source_dataset_name=None,
    source_dataset_source=None,
    target_col=None,
    n_lags=None,
    exog_cols=None,
    return_run_refs=False,
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

    run_refs = {}

    for _, row in df_log.iterrows():
        model_label = str(row["model_label"])
        partition = str(row["partition"])
        model_name = str(row["model_name"])

        run_name = str(run_name_fn(row)) if callable(run_name_fn) else f"{model_label} | {partition}"

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            with mlflow.start_run(run_name=run_name) as active_run:
                current_run_id = active_run.info.run_id

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
                # CONFORMAL LOGGING (NEW)
                # =====================================================
                conformal_obj = None
                if isinstance(conformal_artifacts, dict):
                    conformal_obj = conformal_artifacts.get((model_label, partition), None)
                    if conformal_obj is None:
                        conformal_obj = conformal_artifacts.get((model_label, "ALL"), None)
                    if conformal_obj is None:
                        conformal_obj = conformal_artifacts.get(model_label, None)

                if conformal_obj is not None:
                    try:
                        mlflow.log_param("conformal_alpha", float(conformal_alpha))
                    except Exception:
                        mlflow.log_param("conformal_alpha", str(conformal_alpha))

                    mlflow.set_tag("conformal_logged", "true")

                    q95 = None
                    if isinstance(conformal_obj, dict) and "q95" in conformal_obj:
                        try:
                            q95 = float(conformal_obj["q95"])
                            mlflow.log_metric("conformal_q95", q95)
                            mlflow.set_tag("conformal_has_q95", "true")
                        except Exception:
                            mlflow.set_tag("conformal_has_q95", "false")
                    else:
                        mlflow.set_tag("conformal_has_q95", "false")

                    residuals_df = None
                    intervals_df = None
                    conformal_meta = None

                    if isinstance(conformal_obj, dict):
                        residuals_df = _coerce_conformal_residuals_df(conformal_obj.get("residuals_df"))
                        if residuals_df is None and "abs_residual" in conformal_obj:
                            residuals_df = _coerce_conformal_residuals_df(conformal_obj["abs_residual"])

                        intervals_df = _coerce_conformal_intervals_df(conformal_obj.get("intervals_df"))
                        conformal_meta = conformal_obj.get("meta", None)
                    else:
                        residuals_df = _coerce_conformal_residuals_df(conformal_obj)

                    if residuals_df is not None and len(residuals_df):
                        p = tmp_dir / f"conformal_residuals_{model_label}_{partition}.csv"
                        residuals_df.to_csv(p, index=False)
                        mlflow.log_artifact(str(p), artifact_path="conformal")
                        mlflow.log_metric("conformal_n_residuals", float(len(residuals_df)))
                        mlflow.set_tag("conformal_has_residuals", "true")
                    else:
                        mlflow.set_tag("conformal_has_residuals", "false")

                    if intervals_df is not None and len(intervals_df):
                        p = tmp_dir / f"prediction_intervals_{model_label}_{partition}.parquet"
                        intervals_df.to_parquet(p, index=False)
                        mlflow.log_artifact(str(p), artifact_path="conformal")
                        mlflow.log_metric("conformal_n_intervals", float(len(intervals_df)))
                        mlflow.set_tag("conformal_has_intervals", "true")
                    else:
                        mlflow.set_tag("conformal_has_intervals", "false")

                    meta_payload = {
                        "model_label": model_label,
                        "partition": partition,
                        "alpha": float(conformal_alpha) if conformal_alpha is not None else None,
                        "q95": q95,
                    }
                    if isinstance(conformal_meta, dict):
                        meta_payload.update(conformal_meta)

                    p = tmp_dir / f"conformal_meta_{model_label}_{partition}.json"
                    _safe_write_json(meta_payload, p)
                    mlflow.log_artifact(str(p), artifact_path="conformal")

                else:
                    mlflow.set_tag("conformal_logged", "false")

                    # fallback: si bkt_part a déjà des bornes, on les logge aussi
                    if isinstance(bkt_part, pd.DataFrame) and len(bkt_part):
                        cols = set(bkt_part.columns)
                        if {"ds", "lower", "upper"}.issubset(cols) or {"ds", "lo_95", "hi_95"}.issubset(cols):
                            try:
                                p = tmp_dir / f"prediction_intervals_{model_label}_{partition}.parquet"
                                bkt_part.to_parquet(p, index=False)
                                mlflow.log_artifact(str(p), artifact_path="conformal")
                                mlflow.set_tag("conformal_has_intervals", "true")
                                mlflow.set_tag("conformal_logged_from_backtest", "true")
                            except Exception:
                                pass

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

                artifact_path_logged = None

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
                            artifact_path_logged = f"model_entity_{model_label}_{partition}"
                            flavor_used = _log_model_entity_to_mlflow(
                                est,
                                model_name=artifact_path_logged,
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

                run_refs[(model_label, partition)] = {
                    "run_id": current_run_id,
                    "artifact_path": artifact_path_logged,
                    "run_name": run_name,
                }

    if return_run_refs:
        return run_refs


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
    conformal_by_run = {}

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

            elif "conformal/" in apl and "residual" in apl and apl.endswith(".csv"):
                local = _download_artifact(client, run_id, ap, run_dir)
                df = _safe_read_csv(local)
                conformal_by_run.setdefault(key, {})
                conformal_by_run[key]["residuals_df"] = df

            elif "conformal/" in apl and ("interval" in apl) and (apl.endswith(".parquet") or apl.endswith(".csv")):
                local = _download_artifact(client, run_id, ap, run_dir)
                df = _safe_read_parquet(local) if apl.endswith(".parquet") else _safe_read_csv(local)
                conformal_by_run.setdefault(key, {})
                conformal_by_run[key]["intervals_df"] = df

            elif "conformal/" in apl and apl.endswith(".json"):
                local = _download_artifact(client, run_id, ap, run_dir)
                obj = _safe_read_json(local)
                conformal_by_run.setdefault(key, {})
                conformal_by_run[key]["meta"] = obj

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

        # récupérer q95 depuis metrics si présent
        try:
            run = client.get_run(run_id)
            if "conformal_q95" in run.data.metrics:
                conformal_by_run.setdefault(key, {})
                conformal_by_run[key]["q95"] = run.data.metrics["conformal_q95"]
            if "conformal_alpha" in run.data.params:
                conformal_by_run.setdefault(key, {})
                conformal_by_run[key]["alpha"] = run.data.params["conformal_alpha"]
        except Exception:
            pass

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
        "conformal_by_run": conformal_by_run,  # NEW
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


# =========================================================
# Helpers Registry
# =========================================================
def get_latest_run_ref_for_model_partition(
    *,
    tracking_uri,
    experiment_name,
    model_label,
    partition,
):
    mlflow.set_tracking_uri(tracking_uri)

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment introuvable: {experiment_name}")

    runs = mlflow.search_runs([exp.experiment_id], output_format="pandas")
    if runs.empty:
        raise ValueError("Aucun run trouvé.")

    col_model = "params.model_label" if "params.model_label" in runs.columns else "tags.model_label"
    col_part = "params.partition" if "params.partition" in runs.columns else "tags.partition"

    sub = runs[
        runs[col_model].astype(str).eq(str(model_label))
        & runs[col_part].astype(str).eq(str(partition))
    ].copy()

    if sub.empty:
        raise ValueError(f"Aucun run trouvé pour model={model_label}, partition={partition}")

    sub = sub.sort_values("start_time", ascending=False)
    run_id = str(sub.iloc[0]["run_id"])
    artifact_path = f"model_entity_{model_label}_{partition}"

    return {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_label": str(model_label),
        "partition": str(partition),
    }


# =========================================================
# MLflow Model Registry
# =========================================================
def _ensure_registered_model_exists(client, registered_model_name, description=None):
    try:
        client.get_registered_model(registered_model_name)
        return False
    except Exception:
        client.create_registered_model(
            name=str(registered_model_name),
            description=description or f"Registered model for {registered_model_name}",
        )
        return True


def register_logged_model_to_registry(
    *,
    tracking_uri,
    run_id,
    artifact_path,
    registered_model_name,
    description=None,
    tags=None,
    alias=None,
    wait_until_ready=True,
    max_wait_seconds=20,
):
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    _ensure_registered_model_exists(
        client,
        registered_model_name=registered_model_name,
        description=description,
    )

    model_uri = f"runs:/{run_id}/{artifact_path}"

    mv = mlflow.register_model(
        model_uri=model_uri,
        name=str(registered_model_name),
    )

    version = str(mv.version)

    if wait_until_ready:
        waited = 0
        while waited < max_wait_seconds:
            try:
                mv2 = client.get_model_version(
                    name=str(registered_model_name),
                    version=version,
                )
                if str(getattr(mv2, "status", "")).upper() == "READY":
                    break
            except Exception:
                pass
            time.sleep(1)
            waited += 1

    if tags:
        for k, v in tags.items():
            try:
                client.set_model_version_tag(
                    name=str(registered_model_name),
                    version=version,
                    key=str(k),
                    value=str(v),
                )
            except Exception:
                pass

    if alias:
        try:
            client.set_registered_model_alias(
                name=str(registered_model_name),
                alias=str(alias),
                version=version,
            )
        except Exception:
            pass

    return {
        "registered_model_name": str(registered_model_name),
        "version": version,
        "run_id": str(run_id),
        "artifact_path": str(artifact_path),
        "model_uri": model_uri,
        "alias": alias,
    }


def promote_model_to_alias(
    *,
    tracking_uri,
    registered_model_name,
    version,
    alias="production",
):
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    client.set_registered_model_alias(
        name=str(registered_model_name),
        alias=str(alias),
        version=str(version),
    )

    return {
        "model": str(registered_model_name),
        "version": str(version),
        "alias": str(alias),
    }


def load_model_from_registry(
    *,
    registered_model_name,
    alias="production",
):
    model_uri = f"models:/{registered_model_name}@{alias}"

    if _HAS_MLFLOW_LIGHTGBM:
        try:
            return mlflow.lightgbm.load_model(model_uri)
        except Exception:
            pass

    if _HAS_MLFLOW_SKLEARN:
        try:
            return mlflow.sklearn.load_model(model_uri)
        except Exception:
            pass

    if _HAS_MLFLOW_XGBOOST:
        try:
            return mlflow.xgboost.load_model(model_uri)
        except Exception:
            pass

    if _HAS_MLFLOW_STATSMODELS:
        try:
            return mlflow.statsmodels.load_model(model_uri)
        except Exception:
            pass

    if _HAS_MLFLOW_PYFUNC:
        return mlflow.pyfunc.load_model(model_uri)

    raise RuntimeError(f"Impossible de charger {model_uri}")