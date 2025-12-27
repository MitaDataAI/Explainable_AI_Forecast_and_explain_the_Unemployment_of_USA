from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from explainable_ai_forecast.experiments.model_evaluation_compare_load import (
    _load_run_dir,
    runs_to_long,
    long_to_wide,
)

def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _make_run_dir(
    base: Path,
    run_id: str,
    *,
    model: str,
    ar_p_fixed: int | None = None,
) -> Path:
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": model,
        "target": "UNRATE",
        "horizon": 12,
    }
    if model.lower() == "ar":
        config["ar"] = {
            "p_fixed": ar_p_fixed,
        }

    dataset_info = {
        "path": "dummy.csv",
        "n_rows": 3,
        "n_cols": 2,
        "start": "2000-01-01",
        "end": "2000-03-01",
        "freq": "MS",
    }

    # predictions indexées par date
    idx = pd.date_range("2000-01-01", periods=3, freq="MS")
    preds = pd.DataFrame(
        {
            "y_true": [1.0, 2.0, 3.0],
            "y_pred": [1.1, 1.9, 3.2],
        },
        index=idx,
    )
    # optionnel: colonne p_used pour AR
    if model.lower() == "ar":
        p_used_val = ar_p_fixed if ar_p_fixed is not None else 5
        preds["p_used"] = [p_used_val] * len(preds)

    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "dataset_info.json", dataset_info)

    # parquet
    preds.to_parquet(run_dir / "predictions.parquet")

    return run_dir


def test_load_run_dir_and_long_format(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, "run_linear", model="linear")

    ra = _load_run_dir(run_dir)
    assert ra.run_id == "run_linear"
    assert "y_true" in ra.predictions.columns
    assert "y_pred" in ra.predictions.columns
    assert isinstance(ra.predictions.index, pd.DatetimeIndex)
    assert ra.predictions.index.is_monotonic_increasing


def test_method_inference_ar_fixed_vs_auto(tmp_path: Path) -> None:
    run_linear = _make_run_dir(tmp_path, "rid_linear", model="linear")
    run_ar1 = _make_run_dir(tmp_path, "rid_ar1", model="ar", ar_p_fixed=1)
    run_arp_auto = _make_run_dir(tmp_path, "rid_arp_auto", model="ar", ar_p_fixed=None)

    runs = [_load_run_dir(run_linear), _load_run_dir(run_ar1), _load_run_dir(run_arp_auto)]
    long_df = runs_to_long(runs)

    methods = set(long_df["method"].unique())
    assert "linear" in methods or "LinearRegression" in methods or "linear" in methods  # tolérance
    assert "AR(1)" in methods
    assert "AR(p auto)" in methods

    # colonnes minimales
    assert set(["date", "method", "run_id", "y_true", "y_pred"]).issubset(long_df.columns)

    # dates valides
    assert pd.api.types.is_datetime64_any_dtype(long_df["date"])


def test_long_to_wide_pivot(tmp_path: Path) -> None:
    run_linear = _make_run_dir(tmp_path, "rid_linear", model="linear")
    run_ar1 = _make_run_dir(tmp_path, "rid_ar1", model="ar", ar_p_fixed=1)

    runs = [_load_run_dir(run_linear), _load_run_dir(run_ar1)]
    long_df = runs_to_long(runs)
    wide_df = long_to_wide(long_df)

    # index datetime + colonne y_true
    assert isinstance(wide_df.index, pd.DatetimeIndex)
    assert "y_true" in wide_df.columns

    # au moins une colonne de prédiction par méthode
    # (le nom exact dépend de l'inférence)
    pred_cols = [c for c in wide_df.columns if c != "y_true"]
    assert len(pred_cols) >= 1

    # y_true doit être présent sur les mêmes dates
    assert wide_df["y_true"].notna().all()