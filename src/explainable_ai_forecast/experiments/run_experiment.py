from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.explainable_ai_forecast.experiments.data_validation import load_features
from src.explainable_ai_forecast.experiments.data_preparation_features import make_supervised
from src.explainable_ai_forecast.experiments.data_preparation_preprocess import PreprocSpec
from src.explainable_ai_forecast.experiments.model_evaluation_io_artifacts import (
    ensure_dir,
    make_run_id,
    save_json,
    save_predictions_parquet,
)
from src.explainable_ai_forecast.experiments.model_evaluation_metrics import (
    format_window_report,
    metrics_by_windows,
)
from src.explainable_ai_forecast.experiments.model_training_backtest import (
    ARSpec,
    backtest_ar_expanding,
    pseudo_oos_expanding,
)
from src.explainable_ai_forecast.experiments.models.registry import make_model


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
    p.add_argument("--model", type=str, default="linear")  # "linear" ou "ar"

    # ---- préprocessing (avancé) : utilisé pour modèles type sklearn (ex: linear) ----
    p.add_argument("--winsor-level", type=float, default=0.01)
    p.add_argument("--no-normalize", action="store_true", help="Désactive la normalisation (z-score).")

    # ---- AR options ----
    p.add_argument(
        "--ar-p",
        type=int,
        default=None,
        help="Ordre p fixe pour AR(p). Si absent -> sélection automatique.",
    )
    p.add_argument(
        "--ar-auto-p",
        action="store_true",
        help="Force la sélection automatique de p (ignore --ar-p).",
    )

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


def main() -> None:
    args = parse_args()

    preproc_spec = PreprocSpec(
        winsor_level=float(args.winsor_level),
        normalize=(not args.no_normalize),
    )

    # Choix p fixe vs auto
    use_auto_p = bool(args.ar_auto_p) or (args.ar_p is None)
    p_fixed = None if use_auto_p else int(args.ar_p)

    config = {
        "data_path": args.data_path,
        "date_col": args.date_col,
        "target": args.target,
        "horizon": args.horizon,
        "min_train_n": args.min_train_n,
        "start": args.start,
        "end": args.end,
        "model": args.model,
        "preprocess": {
            "winsor_level": preproc_spec.winsor_level,
            "normalize": preproc_spec.normalize,
        },
        "ar": {
            "trend": "c",
            "p_fixed": p_fixed,  # ✅ NEW
            "p_grid": list(range(1, 13)),  # utilisé seulement si p_fixed is None
            "cv_anchor": "1983-01-01",
            "cv_update_every_months": 36,
            "use_bagging": True,
            "B_boot": 30,
            "L_block": 12,
            "seed": 123,
        },
    }

    df, info = load_features(args.data_path, date_col=args.date_col)
    dataset_info = info.__dict__

    # Dataset supervisé (sert aux modèles X/y_future; pour AR on réutilise y_current)
    sup = make_supervised(
        df,
        target=args.target,
        horizon=args.horizon,
        add_target_lags=[args.horizon],  # ex: UNRATE_lag12 si horizon=12
    )

    # -------------------------
    # Backtest selon le modèle
    # -------------------------
    if args.model.lower() == "ar":
        # AR travaille sur y(t) (y_current). On produit des prévisions y(t+h).
        ar_spec = ARSpec(
            h=args.horizon,
            min_train_n=args.min_train_n,
            trend=config["ar"]["trend"],
            p_fixed=config["ar"]["p_fixed"],  # ✅ NEW
            p_grid=range(1, 13),
            cv_anchor=config["ar"]["cv_anchor"],
            cv_update_every_months=config["ar"]["cv_update_every_months"],
            use_bagging=config["ar"]["use_bagging"],
            B_boot=config["ar"]["B_boot"],
            L_block=config["ar"]["L_block"],
            seed=config["ar"]["seed"],
        )

        bt_ar = backtest_ar_expanding(sup.y_current, ar_spec)

        # IMPORTANT:
        # Ton pipeline "linear" indexe les prédictions par t_end (date de features),
        # alors que le backtest AR renvoie la date forecastée t_fore = t_end + h.
        # On remet l'index au format t_end pour rester compatible avec tes métriques/artefacts actuels.
        pred_df = bt_ar.predictions.copy()
        pred_df.index = pred_df.index - pd.DateOffset(months=args.horizon)
        pred_df.index.name = "date"

        pred_df = _filter_by_dates(pred_df, args.start, args.end)
        bt_predictions = pred_df

    else:
        # Modèles "sklearn-like" (ex: linear)
        model = make_model(args.model)

        bt = pseudo_oos_expanding(
            sup.X,
            sup.y_future,
            model=model,
            min_train_n=args.min_train_n,
            start=args.start,
            end=args.end,
            preproc=preproc_spec,  # préproc avancé (fit train, apply train+forecast)
        )
        bt_predictions = bt.predictions

    # -------------------------
    # Évaluation + sauvegardes
    # -------------------------
    metrics = metrics_by_windows(bt_predictions)
    report = format_window_report(metrics)

    run_id = make_run_id(config, dataset_info)
    run_dir = ensure_dir(Path(args.artifacts_dir) / run_id)

    save_json(run_dir / "config.json", config)
    save_json(run_dir / "dataset_info.json", dataset_info)
    save_json(run_dir / "metrics_windows.json", metrics)
    save_predictions_parquet(run_dir / "predictions.parquet", bt_predictions)

    print(f"[OK] Run saved to: {run_dir}")
    print(report)


if __name__ == "__main__":
    main()