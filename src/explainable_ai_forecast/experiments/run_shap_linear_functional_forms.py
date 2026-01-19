# ============================================================
# SHAP (linéaire) — Functional-form plots (GRID)
# - Lit les snapshots OOS sauvés par pseudo_oos_expanding
# - Calcule phi_j(t) = (x_j(t) - mean_train_j(t)) * beta_j(t)
# - Trace un grid (rows=variables, col=Linear)
#
# Usage:
#   python -m explainable_ai_forecast.experiments.run_shap_linear_functional_form_grid \
#       --snapshots-root /path/to/comparison/run_x/oos_snapshots \
#       --method LINREG \
#       --output /path/to/out/shap_linear_grid.png \
#       --vars UNRATE_lags_12 INDPRO "S&P500" BUSLOANS TB3MS RPI
# ============================================================

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------- setup DARK THEME (global)
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#0d0d0d",
    "axes.facecolor":   "#111111",
    "savefig.facecolor":"#0d0d0d",
    "text.color":       "#e5e5e5",
    "axes.edgecolor":   "#bfbfbf",
    "axes.labelcolor":  "#e5e5e5",
    "xtick.color":      "#d8d8d8",
    "ytick.color":      "#d8d8d8",
    "grid.color":       "#5a5a5a",
})
_DARK_LINE = "#ffffff"     # lignes de tendance
_DARK_ZERO = "#7a7a7a"     # axes 0
_DARK_NA   = "#a5a5a5"     # texte NA
_DARK_GRID = (0.25, ":")   # alpha, style

# palette contrastée pour fond noir
_DARK_COLORS = (
    "#7fdfff",  # cyan
    "#ffd166",  # jaune
    "#ff7f7f",  # rouge clair
    "#9d7aff",  # violet
    "#8bd17c",  # vert
    "#f29ae1",  # rose
)


# ---------------------- Alias & résolution de noms
_alias_map = {
    "UNRATE_lags_12": ["UNRATE_lag12", "UNRATE_lags12", "lag12_UNRATE", "UNRATE_L12", "UNRATE_LAG_12", "UNRATE_LAG12"],
    "S&P500": ["SP500", "S_P_500", "SANDP500", "SP_500", "S&P_500"],
    "OILPRICE": ["OIL_PRICE", "CRUDE_OIL", "WTI", "BRENT"],
    "BUSLOANS": ["BUS_LOANS", "BUSINESS_LOANS", "LOANS_BUSINESS"],
    "INDPRO": ["IND_PRO", "IND_PRODUCTION", "IP_INDEX"],
    "TB3MS": ["TB3M", "T3M", "TBILL3M", "TBILL_3M"],
    "RPI": ["PERSONAL_INCOME", "PI", "REAL_PI"],
}


def _resolve_feature_name(col_list, feat) -> Optional[str]:
    """Résout un alias de variable vers une colonne existante."""
    if feat in col_list:
        return feat

    for alt in _alias_map.get(feat, []):
        if alt in col_list:
            return alt

    # fallback "normalize" (enlève symboles)
    f = re.sub(r"[^A-Z0-9]", "", str(feat).upper())
    for col in col_list:
        c = re.sub(r"[^A-Z0-9]", "", str(col).upper())
        if c == f:
            return col

    # hack spécifique UNRATE lag12
    if "UNRATE" in f and ("12" in f):
        for col in col_list:
            cu = str(col).upper()
            if "UNRATE" in cu and ("LAG12" in cu or ("LAG" in cu and "12" in cu)):
                return col

    return None


# ---------------------- Snapshot loading + linear SHAP
def _linear_phi_one_snapshot(X_train_used: pd.DataFrame,
                             X_fore_used: pd.DataFrame,
                             model) -> Tuple[pd.Series, pd.Series]:
    """
    Retourne:
      - x (observed values) : Series features
      - phi : Series features, phi_j = (x_j - mean_train_j) * beta_j
    """
    if X_fore_used.shape[0] != 1:
        # on ne s'attend qu'à un forecast point par snapshot
        X_fore_used = X_fore_used.iloc[[0]]

    mu = X_train_used.mean(axis=0)
    x = X_fore_used.iloc[0]

    if not hasattr(model, "coef_"):
        raise ValueError(f"Le modèle n'a pas coef_: {type(model).__name__}")

    beta = pd.Series(np.asarray(model.coef_).ravel(), index=X_train_used.columns)
    # alignement strict
    beta = beta.reindex(x.index)

    phi = (x - mu.reindex(x.index)) * beta
    return x, phi


def _iter_snapshot_dirs(snapshots_root: Path, method: str) -> List[Path]:
    """
    snapshots_root/
      YYYY-MM/
        METHOD/
          model.joblib, X_train_used.parquet, X_fore_used.parquet, meta.json, ...
    """
    out = []
    for month_dir in sorted([p for p in snapshots_root.iterdir() if p.is_dir()]):
        snap_dir = month_dir / method
        if snap_dir.exists() and snap_dir.is_dir():
            out.append(snap_dir)
    return out


def load_linear_oos_series(snapshots_root: Path, method: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construit deux DataFrames alignés :
      X_all  : index=date, cols=features, valeurs=x_fore_used
      PHI_all: index=date, cols=features, valeurs=phi (linéaire exact)
    """
    snap_dirs = _iter_snapshot_dirs(snapshots_root, method)
    if not snap_dirs:
        raise FileNotFoundError(f"Aucun snapshot trouvé dans {snapshots_root} pour method={method}")

    xs = []
    phis = []
    dates = []

    for sd in snap_dirs:
        model_path = sd / "model.joblib"
        xtr_path = sd / "X_train_used.parquet"
        xfc_path = sd / "X_fore_used.parquet"
        meta_path = sd / "meta.json"

        if not (model_path.exists() and xtr_path.exists() and xfc_path.exists() and meta_path.exists()):
            # snapshot incomplet -> skip
            continue

        model = joblib.load(model_path)
        X_train_used = pd.read_parquet(xtr_path)
        X_fore_used = pd.read_parquet(xfc_path)
        meta = pd.read_json(meta_path, typ="series")

        t = pd.Timestamp(meta.get("date"))
        x, phi = _linear_phi_one_snapshot(X_train_used, X_fore_used, model)

        xs.append(x)
        phis.append(phi)
        dates.append(t)

    if not dates:
        raise RuntimeError(f"Snapshots trouvés mais aucun exploitable (fichiers manquants) dans {snapshots_root}/{method}")

    X_all = pd.DataFrame(xs, index=pd.DatetimeIndex(dates)).sort_index()
    PHI_all = pd.DataFrame(phis, index=pd.DatetimeIndex(dates)).sort_index()
    X_all.index.name = "date"
    PHI_all.index.name = "date"

    # garde uniquement l'intersection des colonnes (sécurité)
    common_cols = [c for c in X_all.columns if c in PHI_all.columns]
    X_all = X_all.loc[:, common_cols]
    PHI_all = PHI_all.loc[:, common_cols]

    return X_all, PHI_all


# ---------------------- Plot (grid)
def plot_functional_grid_linear(
    X_all: pd.DataFrame,
    PHI_all: pd.DataFrame,
    selected_vars: List[str],
    *,
    poly_deg: int = 1,
    scatter_alpha: float = 0.70,
    s: int = 18,
    colors=_DARK_COLORS,
    figsize_per_cell: Tuple[float, float] = (3.2, 2.8),
    sharey: bool = False,
):
    """
    Grid 1 colonne (Linear), rows=variables.
    """
    model_name = "Linear regression"
    n_rows, n_cols = len(selected_vars), 1

    fig_w = max(6.0, n_cols * figsize_per_cell[0])
    fig_h = max(6.0, n_rows * figsize_per_cell[1])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharey=sharey)
    fig.patch.set_facecolor("#0d0d0d")

    if n_rows == 1:
        axes = np.array([axes])
    axes = axes.reshape(-1, 1)

    col_index_map = {col: j for j, col in enumerate(X_all.columns)}
    color_pts = colors[0 % len(colors)]
    deg = int(poly_deg)

    for r, feat in enumerate(selected_vars):
        ax = axes[r, 0]
        ax.set_facecolor("#111111")

        resolved = _resolve_feature_name(X_all.columns, feat)

        if resolved is None or resolved not in col_index_map:
            ax.text(
                0.5, 0.5, "NA",
                ha="center", va="center",
                fontsize=11, fontweight="bold", color=_DARK_NA,
                transform=ax.transAxes
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0:
                ax.set_title(model_name, fontsize=9, pad=4, color="#e5e5e5")
            ax.set_ylabel(feat, fontsize=9, color="#e5e5e5")
            continue

        j = col_index_map[resolved]
        x = X_all.iloc[:, j].to_numpy()
        y = PHI_all.iloc[:, j].to_numpy()

        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        ax.scatter(
            x, y,
            alpha=scatter_alpha, s=s,
            facecolors="#0d0d0d",
            edgecolors=color_pts,
            linewidths=0.8
        )

        if len(x) > deg + 1:
            try:
                coefs = np.polyfit(x, y, deg=deg)
                x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                y_line = np.polyval(coefs, x_line)
                ax.plot(x_line, y_line, color=_DARK_LINE, linewidth=1.2)
            except Exception:
                pass

        ax.axhline(0, color=_DARK_ZERO, lw=0.9)
        ax.axvline(0, color=_DARK_ZERO, lw=0.9)
        ax.grid(axis="y", linestyle=_DARK_GRID[1], alpha=_DARK_GRID[0])

        if r == 0:
            ax.set_title(model_name, fontsize=9, pad=4, color="#e5e5e5")
        ax.set_ylabel(feat, fontsize=9, color="#e5e5e5")
        if r == n_rows - 1:
            ax.set_xlabel("Observed values (x_fore_used)", fontsize=8, color="#e5e5e5")

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#bfbfbf")
        ax.spines["bottom"].set_color("#bfbfbf")

    plt.tight_layout(h_pad=0.9, w_pad=0.9)
    return fig, axes


# ---------------------- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshots-root", type=str, required=True,
                   help="Racine des snapshots (ex: comparison/<run_id>/oos_snapshots)")
    p.add_argument("--method", type=str, default="LINREG",
                   help="Nom du method_name utilisé lors du backtest (ex: LINREG, RIDGE, etc.)")
    p.add_argument("--output", type=str, required=True,
                   help="Chemin de sortie image (png/pdf)")
    p.add_argument("--vars", nargs="*", default=None,
                   help="Liste de variables à tracer. Si vide -> toutes les colonnes de X_fore_used.")
    p.add_argument("--poly-deg", type=int, default=1,
                   help="Degré du fit (linéaire=1 conseillé)")
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--sharey", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    snapshots_root = Path(args.snapshots_root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X_all, PHI_all = load_linear_oos_series(snapshots_root, args.method)

    selected_vars = args.vars if args.vars else list(X_all.columns)

    fig, _ = plot_functional_grid_linear(
        X_all=X_all,
        PHI_all=PHI_all,
        selected_vars=selected_vars,
        poly_deg=int(args.poly_deg),
        sharey=bool(args.sharey),
    )

    fig.savefig(out_path, dpi=int(args.dpi))
    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()