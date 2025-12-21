from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SupervisedData:
    """
    Dataset supervisé pour une prévision à horizon h.

    - X        : features observées au temps t
    - y_future : cible au temps t+h
    - y_current: cible au temps t (utile pour diagnostics)
    """
    X: pd.DataFrame
    y_future: pd.Series
    y_current: pd.Series
    target: str
    horizon: int


def make_supervised(
    df: pd.DataFrame,
    *,
    target: str,
    horizon: int,
    add_target_lags: Iterable[int] = (),
    drop_cols: Iterable[str] = (),
) -> SupervisedData:
    """
    Transforme un DataFrame mensuel (index MS) en problème supervisé horizon h.

    Définition:
      - X(t): variables disponibles à t
      - y_future(t) = y(t+h)

    Optionnel:
      - Ajout de lags de la cible dans X (ex: UNRATE_lag12 = y(t-12))

    Notes:
      - Aucune fuite temporelle: la cible future est obtenue via shift(-horizon).
      - Les NaN induits (lags, horizon) sont conservés volontairement.
    """
    # --- validations de base ---
    if target not in df.columns:
        raise KeyError(
            f"Target '{target}' absente du DataFrame. "
            f"Colonnes disponibles (extrait): {list(df.columns)[:20]}"
        )

    if horizon <= 0:
        raise ValueError("horizon doit être strictement positif")

    df2 = df.copy()

    # --- suppression optionnelle de colonnes ---
    drop_cols = list(drop_cols)
    for col in drop_cols:
        if col not in df2.columns:
            raise KeyError(f"drop_cols contient une colonne absente: '{col}'")
    if drop_cols:
        df2 = df2.drop(columns=drop_cols)

    # --- cible courante ---
    y_current = df2[target].astype(float)

    # --- features de base ---
    X = df2.drop(columns=[target]).astype(float)

    # --- ajout des lags de la cible ---
    for lag in add_target_lags:
        if lag <= 0:
            raise ValueError("Chaque lag doit être strictement positif")
        X[f"{target}_lag{lag}"] = y_current.shift(lag)

    # --- cible future à horizon h ---
    y_future = y_current.shift(-horizon)

    return SupervisedData(
        X=X,
        y_future=y_future,
        y_current=y_current,
        target=target,
        horizon=horizon,
    )