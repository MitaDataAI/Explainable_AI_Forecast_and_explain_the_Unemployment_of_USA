from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import json
import pandas as pd


# ----------------------------
# Types
# ----------------------------

@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    config: dict
    dataset_info: dict
    predictions: pd.DataFrame  # index=date, cols at least y_true,y_pred


# ----------------------------
# Low-level IO
# ----------------------------

def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_run_dir(run_dir: Path) -> RunArtifacts:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir introuvable: {run_dir}")

    config = _read_json(run_dir / "config.json")
    dataset_info = _read_json(run_dir / "dataset_info.json")

    pred_path = run_dir / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.parquet introuvable dans {run_dir}")

    preds = pd.read_parquet(pred_path)

    # On force l'index date si besoin
    if "date" in preds.columns:
        preds["date"] = pd.to_datetime(preds["date"], errors="coerce")
        preds = preds.set_index("date")
    preds.index = pd.to_datetime(preds.index, errors="coerce")

    # Colonnes attendues
    if "y_true" not in preds.columns or "y_pred" not in preds.columns:
        raise ValueError(f"Colonnes manquantes dans predictions: {preds.columns.tolist()}")

    preds = preds.sort_index()

    run_id = run_dir.name
    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        config=config,
        dataset_info=dataset_info,
        predictions=preds,
    )


def load_runs(artifacts_root: str | Path, run_ids: Iterable[str]) -> list[RunArtifacts]:
    root = Path(artifacts_root)
    out: list[RunArtifacts] = []
    for rid in run_ids:
        out.append(_load_run_dir(root / rid))
    return out


# ----------------------------
# Standardisation "long"
# ----------------------------

def _infer_method_name(ra: RunArtifacts) -> str:
    """
    Nom lisible du modèle pour comparaison.
    Exemples:
      - linear -> 'LinearRegression'
      - ar + p_fixed=1 -> 'AR(1)'
      - ar + auto -> 'AR(p auto)'
    """
    cfg = ra.config or {}
    m = str(cfg.get("model", "model")).lower()

    if m == "ar":
        ar_cfg = cfg.get("ar", {}) or {}
        p_fixed = ar_cfg.get("p_fixed", None)
        if p_fixed is None:
            return "AR(p auto)"
        return f"AR({int(p_fixed)})"

    # fallback (linear, ridge, lgbm...)
    return str(cfg.get("model", "Model"))


def runs_to_long(runs: list[RunArtifacts]) -> pd.DataFrame:
    """
    Retourne un DF long: date, method, y_true, y_pred (+ p_used si présent).
    """
    frames = []
    for ra in runs:
        df = ra.predictions.copy()
        df = df.reset_index().rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        method = _infer_method_name(ra)
        df["method"] = method
        df["run_id"] = ra.run_id

        cols = ["date", "method", "run_id", "y_true", "y_pred"]
        if "p_used" in df.columns:
            cols.append("p_used")

        frames.append(df[cols])

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["date", "y_true", "y_pred"]).sort_values(["date", "method"]).reset_index(drop=True)
    return out

def long_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot en wide: index=date, colonnes=method, plus colonne y_true.
    """
    df = long_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # y_true unique par date (si plusieurs, on prend médiane)
    true_by_date = df.groupby("date")["y_true"].median().rename("y_true")

    wide = df.pivot_table(index="date", columns="method", values="y_pred", aggfunc="mean").sort_index()
    wide = wide.join(true_by_date, how="left")
    return wide