from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from explainable_ai_forecast.experiments.data_validation import load_features
from explainable_ai_forecast.experiments.data_preparation_features import make_supervised
from explainable_ai_forecast.experiments.data_preparation_preprocess import PreprocSpec
from explainable_ai_forecast.experiments.model_training_backtest import (
    ARSpec,
    backtest_ar_expanding,
    pseudo_oos_expanding,
)
from explainable_ai_forecast.experiments.models.registry import make_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--date-col", type=str, default="date")
    p.add_argument("--target", type=str, default="UNRATE")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--min-train-n", type=int, default=36)

    # bornes (optionnelles)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)

    # où écrire comparison/<run_id>/...
    p.add_argument("--compare-dir", type=str, required=True)

    # préproc (pour modèles sklearn-like)
    p.add_argument("--winsor-level", type=float, default=0.01)
    p.add_argument("--no-normalize", action="store_true")

    # snapshots
    p.add_argument("--save-snapshots", action="store_true")
    p.add_argument("--snapshots-subdir", type=str, default="oos_snapshots")

    return p.parse_args()


def _filter_by_dates(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if df.empty:
        return df
    out = df
    if start is not None:
        out = out.loc[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out.index <= pd.Timestamp(end)]
    return out


def _to_long(pred_df: pd.DataFrame, method_label: str) -> pd.DataFrame:
    out = pred_df.reset_index().rename(columns={"index": "date"})
    if "date" not in out.columns:
        out = out.rename(columns={out.columns[0]: "date"})
    out["method"] = method_label
    return out[["date", "method", "y_pred", "y_true"]]


def main() -> None:
    args = parse_args()

    compare_dir = Path(args.compare_dir)
    compare_dir.mkdir(parents=True, exist_ok=True)

    snapshots_root = compare_dir / args.snapshots_subdir
    if args.save_snapshots:
        snapshots_root.mkdir(parents=True, exist_ok=True)

    preproc_spec = PreprocSpec(
        winsor_level=float(args.winsor_level),
        normalize=(not args.no_normalize),
    )

    # -------------------------
    # Load + supervised dataset
    # -------------------------
    df, info = load_features(args.data_path, date_col=args.date_col)

    sup = make_supervised(
        df,
        target=args.target,
        horizon=args.horizon,
        add_target_lags=[args.horizon],  # ex: UNRATE_lag12 si horizon=12
    )

    # -------------------------
    # Models
    # -------------------------
    # AR1 (p fixe = 1)
    ar1_spec = ARSpec(
        h=args.horizon,
        min_train_n=args.min_train_n,
        trend="c",
        p_fixed=1,
        p_grid=range(1, 13),
        cv_anchor="1983-01-01",
        cv_update_every_months=36,
        use_bagging=True,
        B_boot=30,
        L_block=12,
        seed=123,
        save_snapshots=bool(args.save_snapshots),
        snapshots_root=str(snapshots_root) if args.save_snapshots else None,
        method_name="AR1",
    )
    bt_ar1 = backtest_ar_expanding(sup.y_current, ar1_spec)
    pred_ar1 = bt_ar1.predictions.copy()
    pred_ar1.index = pred_ar1.index - pd.DateOffset(months=args.horizon)
    pred_ar1.index.name = "date"
    pred_ar1 = _filter_by_dates(pred_ar1, args.start, args.end)

    # ARP (p auto)
    arp_spec = ARSpec(
        h=args.horizon,
        min_train_n=args.min_train_n,
        trend="c",
        p_fixed=None,
        p_grid=range(1, 13),
        cv_anchor="1983-01-01",
        cv_update_every_months=36,
        use_bagging=True,
        B_boot=30,
        L_block=12,
        seed=123,
        save_snapshots=bool(args.save_snapshots),
        snapshots_root=str(snapshots_root) if args.save_snapshots else None,
        method_name="ARP",
    )
    bt_arp = backtest_ar_expanding(sup.y_current, arp_spec)
    pred_arp = bt_arp.predictions.copy()
    pred_arp.index = pred_arp.index - pd.DateOffset(months=args.horizon)
    pred_arp.index.name = "date"
    pred_arp = _filter_by_dates(pred_arp, args.start, args.end)

    # LINREG (modèle registry)
    lin_model = make_model("linear")
    bt_lin = pseudo_oos_expanding(
        sup.X,
        sup.y_future,
        model=lin_model,
        min_train_n=args.min_train_n,
        start=args.start,
        end=args.end,
        preproc=preproc_spec,
        save_snapshots=bool(args.save_snapshots),
        snapshots_root=snapshots_root if args.save_snapshots else None,
        method_name="LINREG",
    )
    pred_lin = bt_lin.predictions

    # -------------------------
    # Save predictions_long.csv
    # -------------------------
    df_long = pd.concat(
        [
            _to_long(pred_ar1, "AR(1)"),
            _to_long(pred_arp, "AR(p auto)"),
            _to_long(pred_lin, "linear"),
        ],
        ignore_index=True,
    )
    df_long["date"] = pd.to_datetime(df_long["date"])
    df_long = df_long.sort_values(["date", "method"]).reset_index(drop=True)

    out_pred_long = compare_dir / "predictions_long.csv"
    df_long.to_csv(out_pred_long, index=False)

    print("✅ Compare backtest terminé")
    print(f"✅ predictions_long.csv: {out_pred_long}")
    if args.save_snapshots:
        print(f"✅ snapshots: {snapshots_root}")
    print(df_long.head(10))


if __name__ == "__main__":
    main()