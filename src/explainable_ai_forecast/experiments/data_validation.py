# 3_scripts/experiments/data.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class DatasetInfo:
    path: str
    n_rows: int
    n_cols: int
    start: str
    end: str
    freq: str


def _to_month_start_index(idx: pd.Index) -> pd.DatetimeIndex:
    """Convertit un index en DatetimeIndex ancré au début de mois (MS)."""
    dt = pd.to_datetime(idx, errors="coerce")
    if dt.isna().any():
        bad = idx[dt.isna()][:5]
        raise ValueError(f"Index date non convertible (exemples): {list(bad)}")
    return dt.to_period("M").to_timestamp(how="start")


def ensure_ms_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Force un DatetimeIndex de fréquence mensuelle (MS), comme dans ton notebook.
    - Si date_col est présent, il devient l'index.
    - L'index est converti en "début de mois".
    - La série est reindexée en fréquence MS (asfreq) pour homogénéiser.
    """
    out = df.copy()

    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        if out[date_col].isna().any():
            bad = out.loc[out[date_col].isna()].head(5)
            raise ValueError(
                f"Colonne '{date_col}' contient des dates non parseables. "
                f"Exemples:\n{bad}"
            )
        out = out.set_index(date_col)
    else:
        # date_col absent -> on suppose que l'index est la date
        if out.index.name is None:
            out.index.name = date_col

    out.index = _to_month_start_index(out.index)

    # Enlève doublons d'index (sinon asfreq échoue ou introduit de l'ambiguïté)
    if out.index.has_duplicates:
        dup = out.index[out.index.duplicated()].unique()[:5]
        raise ValueError(f"Index date contient des doublons (exemples): {list(dup)}")

    out = out.sort_index()

    # Force fréquence MS (introduit des NaN si des mois manquent, ce qui est utile à détecter)
    out = out.asfreq("MS")

    return out


def load_features(
    path: str,
    date_col: str = "date",
    *,
    index_freq: str = "MS",
) -> tuple[pd.DataFrame, DatasetInfo]:
    """
    Charge un fichier features final (csv/parquet) et renvoie (df, info).
    df : index mensuel MS, trié, prêt pour la suite du pipeline expérimental.

    NB: index_freq est gardé pour extensibilité; ici on utilise MS.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fichier introuvable: {p}")

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Format non supporté: {p.suffix} (attendu: .csv ou .parquet)")

    df = ensure_ms_index(df, date_col=date_col)

    info = DatasetInfo(
        path=str(p),
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        start=str(df.index.min().date()) if len(df) else "NA",
        end=str(df.index.max().date()) if len(df) else "NA",
        freq=index_freq,
    )
    return df, info