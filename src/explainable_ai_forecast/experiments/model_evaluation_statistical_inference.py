"""
Statistical inference — Shapley regressions FROM SNAPSHOTS (OOS)

Objectif:
- Ne PAS recalculer SHAP.
- Utiliser un fichier snapshot déjà produit (SHAP OOS).
- Construire un panel Phi_t (t x features) et estimer:
    y_true = alpha + sum_j beta_j^S * phi_{j,t} + eps_t
- Exporter une table avec βˢ, p-value, sig, Γˢ (Shapley share)

Formats de snapshots supportés (auto-détection):
A) Long format (recommandé):
    date, model, feature, shap_value, y_true   (y_true optionnel si fourni ailleurs)
B) Wide format:
    date, model, y_true, SHAP_<feature1>, SHAP_<feature2>, ...

Remarque:
- On filtre par model si fourni (sinon on prend tout ce qui existe).
"""

from __future__ import annotations

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import statsmodels.api as sm


# -------------------------
# Utilitaires
# -------------------------
def _sigstars(p: float) -> str:
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out


def _infer_long_or_wide(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if {"feature", "shap_value"}.issubset(cols):
        return "long"
    # wide: au moins une colonne shap-like
    if any(c.lower().startswith("shap_") for c in cols):
        return "wide"
    raise ValueError(
        "Format snapshots non reconnu. Attendu soit long (feature, shap_value), "
        "soit wide (colonnes shap_*)."
    )


def _phi_y_from_snapshots(
    df_snap: pd.DataFrame,
    *,
    model: Optional[str],
    date_col: str,
    y_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Retourne:
    - Phi: DataFrame index=date, colonnes=features, valeurs=phi
    - y: Series index=date, valeurs=y_true
    """
    fmt = _infer_long_or_wide(df_snap)

    df = _ensure_datetime(df_snap, date_col)

    if model is not None and "model" in df.columns:
        df = df.loc[df["model"].astype(str) == str(model)].copy()

    if fmt == "long":
        # y_true : pris par date (et model si présent)
        if y_col not in df.columns:
            raise ValueError(f"Colonne y_col='{y_col}' introuvable dans snapshots long.")

        # Pivot shap
        Phi = (
            df.pivot_table(index=date_col, columns="feature", values="shap_value", aggfunc="mean")
            .sort_index()
        )

        # y
        y = (
            df.groupby(date_col)[y_col]
            .mean()
            .sort_index()
        )

    else:  # wide
        if y_col not in df.columns:
            raise ValueError(f"Colonne y_col='{y_col}' introuvable dans snapshots wide.")

        shap_cols = [c for c in df.columns if c.lower().startswith("shap_")]
        if not shap_cols:
            raise ValueError("Aucune colonne shap_* trouvée dans snapshots wide.")

        Phi = (
            df.set_index(date_col)[shap_cols]
            .sort_index()
            .rename(columns=lambda c: c[5:] if c.lower().startswith("shap_") else c)
        )

        y = df.set_index(date_col)[y_col].sort_index()

    # alignement + nettoyage NaN
    common_idx = Phi.index.intersection(y.index)
    Phi = Phi.loc[common_idx]
    y = y.loc[common_idx]

    # drop lignes avec NaN
    mask = (~Phi.isna().any(axis=1)) & (~y.isna())
    Phi = Phi.loc[mask]
    y = y.loc[mask]

    if len(Phi) < 10:
        raise RuntimeError(f"Trop peu d'observations après filtrage: n={len(Phi)}")

    return Phi, y


def _shap_shares(Phi: pd.DataFrame) -> pd.Series:
    abs_mean = Phi.abs().mean(axis=0)
    tot = abs_mean.sum()
    shares = abs_mean / tot if tot > 0 else abs_mean
    return shares.sort_values(ascending=False)


# -------------------------
# Régression + Table
# -------------------------
def shapley_regression_from_phi(
    y: pd.Series,
    Phi: pd.DataFrame,
    cov_type: str = "HC1",
) -> pd.DataFrame:
    X = sm.add_constant(Phi.to_numpy())
    res = sm.OLS(y.to_numpy(), X).fit(cov_type=cov_type)

    out = pd.DataFrame(
        {
            "βˢ": res.params[1:],
            "std_err": res.bse[1:],
            "t": res.tvalues[1:],
            "p-value": res.pvalues[1:],
        },
        index=Phi.columns,
    )
    out["sig"] = out["p-value"].map(_sigstars)
    return out


def statistical_inference_table_from_snapshots(
    df_snapshots: pd.DataFrame,
    *,
    model: Optional[str] = None,
    date_col: str = "date",
    y_col: str = "y_true",
    restrict_eval_window: Optional[Tuple[str, str]] = None,
    cov_type: str = "HC1",
    model_label: Optional[str] = None,
) -> pd.DataFrame:
    """
    Retourne une table type “Table III”:
      βˢ, p-value, sig, Γˢ (Shapley share)
    """
    Phi, y = _phi_y_from_snapshots(df_snapshots, model=model, date_col=date_col, y_col=y_col)

    # restriction fenêtre (optionnel)
    if restrict_eval_window is not None:
        start, end = restrict_eval_window
        if start:
            Phi = Phi.loc[Phi.index >= pd.to_datetime(start)]
            y = y.loc[y.index >= pd.to_datetime(start)]
        if end:
            Phi = Phi.loc[Phi.index <= pd.to_datetime(end)]
            y = y.loc[y.index <= pd.to_datetime(end)]

    reg = shapley_regression_from_phi(y, Phi, cov_type=cov_type)
    shares = _shap_shares(Phi).rename("Γˢ")

    tbl = pd.concat([reg[["βˢ", "p-value", "sig"]], shares], axis=1)
    tbl = tbl.sort_values("Γˢ", ascending=False)

    label = model_label or (model if model is not None else "AllModels")
    tbl.columns = pd.MultiIndex.from_product([[label], tbl.columns])
    return tbl