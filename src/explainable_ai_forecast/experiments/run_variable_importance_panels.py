"""
Run: génère le graphique "importance panels" (SHAP share + Perm abs error + Perm deviance)
à partir du CSV agrégé: xai_perm_shap_all.csv

Exemple:
python -m explainable_ai_forecast.experiments.run_importance_panels \
  --artifacts_dir artifacts/experiments/comparison/8a88d7fbc4d5 \
  --top_k 15
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from explainable_ai_forecast.experiments.model_evaluation_variable_importance_panels import (
    plot_importance_panels,
)


def _build_dicts_from_aggregated_csv(df: pd.DataFrame):
    """
    Convertit le CSV agrégé en dicts attendus par plot_importance_panels:
    - shapleys: {model: Series(index=feature, values=shapley_share)}
    - perm_abs: {model: DataFrame(variable, perm_score_ratio_mean, perm_score_ratio_std)}
    - perm_dev: idem
    """
    required = {"model", "feature", "mean_perm_abs_error", "mean_perm_deviance", "shapley_share"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans xai_perm_shap_all.csv: {sorted(missing)}")

    shapleys = {}
    perm_abs = {}
    perm_dev = {}

    for model_name, g in df.groupby("model"):
        g = g.copy()

        # SHAP: on prend shapley_share par feature
        shapleys[model_name] = (
            g.set_index("feature")["shapley_share"]
            .astype(float)
            .sort_values(ascending=False)
        )

        # Perm abs error: on mappe mean_perm_abs_error vers perm_score_ratio_mean
        # (std indisponible dans ton CSV => NaN)
        perm_abs[model_name] = pd.DataFrame(
            {
                "variable": g["feature"].astype(str).values,
                "perm_score_ratio_mean": g["mean_perm_abs_error"].astype(float).values,
                "perm_score_ratio_std": [float("nan")] * len(g),
            }
        )

        # Perm deviance
        perm_dev[model_name] = pd.DataFrame(
            {
                "variable": g["feature"].astype(str).values,
                "perm_score_ratio_mean": g["mean_perm_deviance"].astype(float).values,
                "perm_score_ratio_std": [float("nan")] * len(g),
            }
        )

    return shapleys, perm_abs, perm_dev


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--artifacts_dir",
        type=str,
        required=True,
        help="Dossier d'artefacts (ex: artifacts/experiments/comparison/<run_id>)",
    )
    p.add_argument(
        "--csv_name",
        type=str,
        default="xai_perm_shap_all.csv",
        help="Nom du CSV agrégé dans artifacts_dir",
    )
    p.add_argument("--top_k", type=int, default=15, help="Top K features à afficher")
    p.add_argument(
        "--out_name",
        type=str,
        default="importance_panels.png",
        help="Nom du fichier image output",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="DPI de sauvegarde",
    )
    args = p.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    in_csv = artifacts_dir / args.csv_name
    if not in_csv.exists():
        raise FileNotFoundError(f"CSV introuvable: {in_csv}")

    df = pd.read_csv(in_csv)

    shapleys, perm_abs, perm_dev = _build_dicts_from_aggregated_csv(df)

    fig = plot_importance_panels(
        shapleys=shapleys,
        perm_abs=perm_abs,
        perm_dev=perm_dev,
        top_k=args.top_k,
    )

    out_dir = artifacts_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()