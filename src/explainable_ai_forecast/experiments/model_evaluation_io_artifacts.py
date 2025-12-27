from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


def _stable_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def make_run_id(config: dict, dataset_info: dict) -> str:
    return _stable_hash({"config": config, "dataset": dataset_info})


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_predictions_parquet(path: str | Path, preds: pd.DataFrame) -> None:
    preds.to_parquet(path, index=True)