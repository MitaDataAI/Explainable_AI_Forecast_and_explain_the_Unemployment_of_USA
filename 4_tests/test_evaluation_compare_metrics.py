from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from explainable_ai_forecast.experiments.model_evaluation_compare_metrics import (
    compare_runs_metrics,
)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _make_run_dir(base: Path, run_id: str, *, model: str, p_fixed: int | None = None) -> Path:
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {"model": model, "target": "UNRATE", "horizon": 12}
    if model.lower() == "ar":
        config["ar"] = {"p_fixed": p_fixed}

    dataset_info = {"freq": "MS"}

    # dates couvrant au moins une "fenêtre" dans metrics_by_windows
    idx = pd.date_range("1983-01-01", periods=60, freq="MS")
    preds = pd.DataFrame(
        {
            "y_true": range(60),
            "y_pred": [x + 0.5 for x in range(60)],
        },
        index=idx,
    )
    preds.to_parquet(run_dir / "predictions.parquet")

    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "dataset_info.json", dataset_info)

    return run_dir


def test_compare_runs_metrics_smoke(tmp_path: Path) -> None:
    root = tmp_path
    _make_run_dir(root, "rid_linear", model="linear")
    _make_run_dir(root, "rid_ar1", model="ar", p_fixed=1)
    _make_run_dir(root, "rid_auto", model="ar", p_fixed=None)

    res = compare_runs_metrics(root, ["rid_linear", "rid_ar1", "rid_auto"])

    assert not res.per_method.empty
    assert "AR(1)" in res.per_method.index
    assert "AR(p auto)" in res.per_method.index

    # au moins une colonne de métrique
    assert res.per_method.shape[1] >= 1