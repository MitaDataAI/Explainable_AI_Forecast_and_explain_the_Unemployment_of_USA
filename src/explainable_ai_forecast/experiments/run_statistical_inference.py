"""
Run — Statistical inference FROM SHAP snapshots (OOS)

Exemple:
python -m explainable_ai_forecast.experiments.run_statistical_inference_from_snapshots `
  --artifacts_dir artifacts/experiments/comparison/8a88d7fbc4d5 `
  --snapshots_path artifacts/experiments/comparison/8a88d7fbc4d5/shap_snapshots.csv `
  --date_col date `
  --y_col y_true `
  --model LINREG `
  --cov_type HC1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from explainable_ai_forecast.experiments.model_evaluation_statistical_inference import (
    statistical_inference_table_from_snapshots,
)


def _load_snapshots(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Snapshots introuvable: {path}")

    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Extension snapshots non supportée: {path.suffix}")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    out = df.copy()
    out.columns = [" | ".join([str(x) for x in c]).strip() for c in out.columns.values]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts_dir", type=str, required=True)
    p.add_argument("--snapshots_path", type=str, required=True, help="CSV/Parquet des SHAP snapshots (OOS).")
    p.add_argument("--model", type=str, default=None, help="Filtre un modèle (si colonne 'model' existe).")
    p.add_argument("--date_col", type=str, default="date")
    p.add_argument("--y_col", type=str, default="y_true")
    p.add_argument("--restrict_eval_start", type=str, default=None)
    p.add_argument("--restrict_eval_end", type=str, default=None)
    p.add_argument("--cov_type", type=str, default="HC1")
    p.add_argument("--out_name", type=str, default="statistical_inference_shapley_regression.csv")
    args = p.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df_snap = _load_snapshots(Path(args.snapshots_path))

    restrict = None
    if args.restrict_eval_start or args.restrict_eval_end:
        restrict = (args.restrict_eval_start or "", args.restrict_eval_end or "")

    tbl = statistical_inference_table_from_snapshots(
        df_snap,
        model=args.model,
        date_col=args.date_col,
        y_col=args.y_col,
        restrict_eval_window=restrict,
        cov_type=args.cov_type,
        model_label=args.model or "AllModels",
    )

    out_dir = artifacts_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    _flatten_columns(tbl).to_csv(out_path, index=True)

    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()