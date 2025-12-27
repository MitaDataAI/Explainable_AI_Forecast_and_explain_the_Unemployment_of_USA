from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from explainable_ai_forecast.experiments.model_evaluation_compare_load import (
    load_runs,
    runs_to_long,
)
from explainable_ai_forecast.experiments.model_evaluation_metrics import (
    metrics_by_windows,
)


@dataclass(frozen=True)
class CompareMetricsResult:
    per_method: pd.DataFrame
    raw_windows: dict


def _flatten_window_metrics(metrics_windows: dict, *, method: str) -> dict:
    row: dict = {"method": method}
    for win_name, m in (metrics_windows or {}).items():
        if isinstance(m, dict):
            for k, v in m.items():
                row[f"{win_name}.{k}"] = v
    return row


def compare_runs_metrics(
    artifacts_root: str | Path,
    compare_id: str,
    run_ids: Iterable[str],
) -> CompareMetricsResult:
    artifacts_root = Path(artifacts_root)

    runs = load_runs(artifacts_root, run_ids)
    long_df = runs_to_long(runs)

    # === persist predictions_long at comparison level
    compare_dir = artifacts_root / "comparison" / compare_id
    compare_dir.mkdir(parents=True, exist_ok=True)

    long_df.to_csv(compare_dir / "predictions_long.csv", index=False)
    long_df.to_parquet(compare_dir / "predictions_long.parquet", index=False)

    rows = []
    raw = {}

    for run_id in sorted(long_df["run_id"].unique()):
        sub = long_df[long_df["run_id"] == run_id].copy()

        base_method = str(sub["method"].iloc[0])
        label = f"{base_method}__{run_id[:8]}"

        pred_df = sub.set_index("date")[["y_true", "y_pred"]].sort_index()

        mw = metrics_by_windows(pred_df)
        raw[label] = mw
        rows.append(_flatten_window_metrics(mw, method=label))

    out = pd.DataFrame(rows).set_index("method").sort_index()
    return CompareMetricsResult(per_method=out, raw_windows=raw)