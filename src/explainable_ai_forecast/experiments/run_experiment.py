from __future__ import annotations

import argparse
import json
from pathlib import Path

from explainable_ai_forecast.experiments.data import load_features
from explainable_ai_forecast.experiments.features import make_supervised
from explainable_ai_forecast.experiments.models.registry import make_model
from explainable_ai_forecast.experiments.backtest import pseudo_oos_expanding
from explainable_ai_forecast.experiments.metrics import regression_metrics
from explainable_ai_forecast.experiments.io_artifacts import (
    ensure_dir,
    make_run_id,
    save_json,
    save_predictions_parquet,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--date-col", type=str, default="date")
    p.add_argument("--target", type=str, default="UNRATE")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--min-train-n", type=int, default=36)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--artifacts-dir", type=str, default="artifacts/experiments")
    p.add_argument("--model", type=str, default="linear")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = {
        "data_path": args.data_path,
        "date_col": args.date_col,
        "target": args.target,
        "horizon": args.horizon,
        "min_train_n": args.min_train_n,
        "start": args.start,
        "end": args.end,
        "model": args.model,
    }

    df, info = load_features(args.data_path, date_col=args.date_col)
    dataset_info = info.__dict__

    sup = make_supervised(df, target=args.target, horizon=args.horizon, add_target_lags=[args.horizon])

    model = make_model(args.model)

    bt = pseudo_oos_expanding(
        sup.X,
        sup.y_future,
        model=model,
        min_train_n=args.min_train_n,
        start=args.start,
        end=args.end,
    )

    metrics = regression_metrics(bt.predictions)

    run_id = make_run_id(config, dataset_info)
    run_dir = ensure_dir(Path(args.artifacts_dir) / run_id)

    save_json(run_dir / "config.json", config)
    save_json(run_dir / "dataset_info.json", dataset_info)
    save_json(run_dir / "metrics.json", metrics)
    save_predictions_parquet(run_dir / "predictions.parquet", bt.predictions)

    print(f"[OK] Run saved to: {run_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()