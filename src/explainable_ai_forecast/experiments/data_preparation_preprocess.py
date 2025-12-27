from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocSpec:
    winsor_level: float = 0.01
    normalize: bool = True


@dataclass(frozen=True)
class FittedPreproc:
    spec: PreprocSpec
    lower: pd.Series  # bornes winsor (par feature)
    upper: pd.Series
    mean: Optional[pd.Series]  # pour standardisation
    std: Optional[pd.Series]


def fit_preproc(X_train: pd.DataFrame, spec: PreprocSpec) -> FittedPreproc:
    """
    Fit du preprocessing sur le TRAIN uniquement:
    - winsorisation: clip aux quantiles [q, 1-q]
    - normalisation: standardisation (mean/std) (optionnel)

    Hypothèse: X_train colonnes numériques.
    """
    if X_train.empty:
        raise ValueError("X_train est vide")

    q = float(spec.winsor_level)
    if not (0.0 <= q < 0.5):
        raise ValueError("winsor_level doit être dans [0, 0.5)")

    # quantiles par feature (ignore NaN)
    lower = X_train.quantile(q)
    upper = X_train.quantile(1.0 - q)

    # Fit stats de scaling sur données winsorisées
    X_w = X_train.clip(lower=lower, upper=upper, axis=1)

    if spec.normalize:
        mean = X_w.mean(skipna=True)
        std = X_w.std(skipna=True, ddof=0).replace(0.0, 1.0)  # éviter division par 0
    else:
        mean = None
        std = None

    return FittedPreproc(spec=spec, lower=lower, upper=upper, mean=mean, std=std)


def apply_preproc(X: pd.DataFrame, fitted: FittedPreproc, *, fillna_value: Optional[float] = None) -> pd.DataFrame:
    """
    Apply preprocessing:
    - clip winsor (avec bornes train)
    - standardize (avec mean/std train), si activé
    - optionnel: fillna (utile pour x_fore 1-ligne)
    """
    if X.empty:
        return X.copy()

    out = X.copy()

    out = out.clip(lower=fitted.lower, upper=fitted.upper, axis=1)

    if fitted.spec.normalize:
        out = (out - fitted.mean) / fitted.std

    if fillna_value is not None:
        out = out.fillna(fillna_value)

    return out