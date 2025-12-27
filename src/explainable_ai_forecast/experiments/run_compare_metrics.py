from __future__ import annotations

from pathlib import Path
import argparse

from explainable_ai_forecast.experiments.model_evaluation_compare_metrics import (
    compare_runs_metrics,
)

# Racine des artifacts
ARTIFACTS_ROOT = Path("artifacts/experiments")

# 👉 mets ici TES vrais run_ids
RUN_IDS = [
    "91e987618f30f5f4",   # Linear
    "1543452ae9c1c8c8",   # AR(1)
    "9a2891c636365b0c",   # AR(p auto)
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--compare-id", type=str, required=True)
    args = p.parse_args()

    compare_id = args.compare_id

    # === appel logique
    res = compare_runs_metrics(
        artifacts_root=ARTIFACTS_ROOT,
        compare_id=compare_id,
        run_ids=RUN_IDS,
    )

    # === sauvegarde métriques (comme avant)
    out_dir = ARTIFACTS_ROOT / "comparison" / compare_id
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "comparison_metrics.csv"
    out_parquet = out_dir / "comparison_metrics.parquet"

    res.per_method.to_csv(out_csv)
    res.per_method.to_parquet(out_parquet)

    print("\n=== COMPARAISON PAR MODÈLE ===")
    print(res.per_method.round(4))

    print("\n✅ Fichiers sauvegardés :")
    print(f" - {out_csv}")
    print(f" - {out_parquet}")


if __name__ == "__main__":
    main()