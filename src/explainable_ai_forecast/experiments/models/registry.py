from __future__ import annotations

from sklearn.linear_model import LinearRegression


def make_model(name: str, params: dict | None = None):
    params = params or {}
    name = name.lower().strip()

    if name in {"linear", "linear_regression", "linreg"}:
        return LinearRegression(**params)

    raise ValueError(f"Modèle inconnu: {name}")