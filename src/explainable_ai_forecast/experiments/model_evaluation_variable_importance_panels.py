"""
Graphique : Shapley share + Permutation importance (MAE) + Permutation importance (Deviance)

But:
- Centraliser le tracé dans un module importable.
- Permettre un appel depuis un run script (ex: run_feature_importance_allinone.py ou run_compare_report_pdf.py).
"""

from __future__ import annotations

from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# UTILITAIRES
# =========================

def _normalize_shap(shap_obj: Union[pd.DataFrame, pd.Series]) -> pd.Series:
    """Normalise une importance SHAP (somme = 1)."""
    if isinstance(shap_obj, pd.Series):
        s = shap_obj.astype(float)
        tot = s.sum()
        return s / tot if tot > 0 else s

    if isinstance(shap_obj, pd.DataFrame):
        df = shap_obj.copy()
        if "feature" in df.columns and df.index.name != "feature":
            df = df.set_index("feature")

        for col in ["share", "value", "abs_mean", "mean_abs_shap", "importance", "shap"]:
            if col in df.columns:
                s = df[col].astype(float)
                tot = s.sum()
                return s / tot if tot > 0 else s

        raise ValueError("Le DataFrame SHAP doit contenir une colonne valide (ex: 'share').")

    raise TypeError("shap_obj doit être une pandas Series ou DataFrame.")


def _prep_perm_df(imp_df: pd.DataFrame) -> pd.DataFrame:
    """Formate la permutation importance pour uniformité."""
    need = {"variable", "perm_score_ratio_mean"}
    if not need.issubset(imp_df.columns):
        raise ValueError("imp_df doit contenir 'variable' et 'perm_score_ratio_mean'.")

    out = imp_df[["variable", "perm_score_ratio_mean"]].copy()
    out = out.rename(columns={"perm_score_ratio_mean": "mean"})
    out["std"] = imp_df["perm_score_ratio_std"].values if "perm_score_ratio_std" in imp_df.columns else np.nan
    return out.set_index("variable")


def _unify_feature_axis(
    shapleys: Optional[Dict[str, Union[pd.DataFrame, pd.Series]]],
    perm_abs: Optional[Dict[str, pd.DataFrame]],
    perm_dev: Optional[Dict[str, pd.DataFrame]],
    top_k: Optional[int],
) -> list[str]:
    """Crée un ordre commun de features, basé sur la moyenne SHAP prioritaire."""
    features = set()

    if shapleys:
        for df in shapleys.values():
            features.update(_normalize_shap(df).index.tolist())

    if perm_abs:
        for df in perm_abs.values():
            features.update(_prep_perm_df(df).index.tolist())

    if perm_dev:
        for df in perm_dev.values():
            features.update(_prep_perm_df(df).index.tolist())

    if not features:
        return []

    score = pd.Series(0.0, index=sorted(features))
    n = pd.Series(0, index=score.index)

    if shapleys:
        for df in shapleys.values():
            s = _normalize_shap(df).reindex(score.index).fillna(0.0)
            score += s
            n += (s > 0).astype(int)

    # fallback: si pas de SHAP exploitable, on peut ordonner par perm_abs
    if (n == 0).all() and perm_abs:
        for df in perm_abs.values():
            p = _prep_perm_df(df).reindex(score.index)
            s = p["mean"].fillna(0.0)
            score += s
            n += (s > 0).astype(int)

    n = n.replace(0, 1)
    order = (score / n).sort_values(ascending=False).index.tolist()
    if top_k:
        order = order[:top_k]
    return order


def _model_styles() -> dict:
    """Couleurs + marqueurs pour chaque modèle."""
    return {
        "LinearRegression": dict(color="#2ca02c", marker="o"),
        "Ridge": dict(color="#1f77b4", marker="s"),
        "LightGBM": dict(color="#ff7f0e", marker="^"),
    }


def _jitter_offsets(models: list[str], width: float = 0.22) -> dict[str, float]:
    """Décalage horizontal léger entre modèles pour éviter le chevauchement."""
    if len(models) <= 1:
        return {models[0]: 0.0} if models else {}
    offs = np.linspace(-width, width, len(models))
    return {m: float(o) for m, o in zip(models, offs)}


# =========================
# API PUBLIQUE
# =========================

def plot_importance_panels(
    *,
    shapleys: Optional[Dict[str, Union[pd.DataFrame, pd.Series]]] = None,
    perm_abs: Optional[Dict[str, pd.DataFrame]] = None,
    perm_dev: Optional[Dict[str, pd.DataFrame]] = None,
    top_k: Optional[int] = None,
    figsize=(18, 4.2),
    rotate=55,
    ylabels=("Shapley share", "Mean permutation values (Absolute error)", "Mean permutation values (Deviance)"),
    jitter_width=0.22,
):
    """
    Trace 1 à 3 panneaux alignés sur les mêmes features:
    - SHAP share
    - Permutation importance (MAE)
    - Permutation importance (Deviance)
    """
    order = _unify_feature_axis(shapleys, perm_abs, perm_dev, top_k)
    if not order:
        raise ValueError("Aucune feature à afficher.")

    panels = []
    if shapleys is not None:
        panels.append("shap")
    if perm_abs is not None:
        panels.append("perm_abs")
    if perm_dev is not None:
        panels.append("perm_dev")

    fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharex=True)
    if len(panels) == 1:
        axes = [axes]

    model_names = []
    if shapleys:
        model_names += list(shapleys.keys())
    if perm_abs:
        model_names += list(perm_abs.keys())
    if perm_dev:
        model_names += list(perm_dev.keys())
    model_names = sorted(list(set(model_names)))

    styles = _model_styles()
    x = np.arange(len(order))
    offsets = _jitter_offsets(model_names, width=jitter_width)

    for ax, pane in zip(axes, panels):
        for xi in x:
            ax.axvline(xi, color="lightgray", lw=0.7, ls="--", alpha=0.5)

        for m in model_names:
            st = styles.get(m, dict(color="k", marker="o"))
            xv = x + offsets.get(m, 0.0)

            if pane == "shap" and shapleys and m in shapleys:
                s = _normalize_shap(shapleys[m]).reindex(order).fillna(0.0).values
                ax.plot(
                    xv, s,
                    linestyle="none",
                    marker=st["marker"],
                    markersize=6,
                    markerfacecolor="none",
                    markeredgewidth=1.4,
                    color=st["color"],
                    label=m,
                )

            elif pane == "perm_abs" and perm_abs and m in perm_abs:
                p = _prep_perm_df(perm_abs[m]).reindex(order)
                ax.errorbar(
                    xv, p["mean"].values,
                    yerr=p["std"].values,
                    fmt=st["marker"],
                    ms=6,
                    mfc="none",
                    mew=1.4,
                    ecolor=st["color"],
                    elinewidth=1.0,
                    capsize=3,
                    linestyle="none",
                    color=st["color"],
                    label=m,
                )

            elif pane == "perm_dev" and perm_dev and m in perm_dev:
                p = _prep_perm_df(perm_dev[m]).reindex(order)
                ax.errorbar(
                    xv, p["mean"].values,
                    yerr=p["std"].values,
                    fmt=st["marker"],
                    ms=6,
                    mfc="none",
                    mew=1.4,
                    ecolor=st["color"],
                    elinewidth=1.0,
                    capsize=3,
                    linestyle="none",
                    color=st["color"],
                    label=m,
                )

        ax.set_ylabel(
            ylabels[0] if pane == "shap"
            else (ylabels[1] if pane == "perm_abs" else ylabels[2])
        )
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=rotate, ha="right")
        ax.set_ylim(bottom=0)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D([], [], marker=styles[m]["marker"], linestyle="none",
                   color=styles[m]["color"], label=m, markerfacecolor="none",
                   markeredgewidth=1.4)
        for m in model_names
        if m in styles
    ]
    
    if handles:
        axes[0].legend(handles=handles, loc="upper left", frameon=False)

    plt.tight_layout()
    return fig