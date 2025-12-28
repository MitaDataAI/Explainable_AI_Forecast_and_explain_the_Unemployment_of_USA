from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Snapshot utilities
# -----------------------------

def _collect_snapshots(snapshots_root: Path, method: str) -> list[Path]:
    """
    snapshots_root/YYYY-MM/METHOD/
    """
    out: list[Path] = []
    if not snapshots_root.exists():
        raise FileNotFoundError(f"Snapshots introuvables: {snapshots_root}")

    for ym_dir in sorted([p for p in snapshots_root.iterdir() if p.is_dir()]):
        mdir = ym_dir / method
        if mdir.exists() and mdir.is_dir():
            out.append(mdir)

    if not out:
        raise FileNotFoundError(f"Aucun snapshot trouvé pour method='{method}' dans {snapshots_root}")
    return out


def _read_snapshot_date(mdir: Path, date_key: str = "date") -> pd.Timestamp:
    """
    1) meta.json[date_key] si présent
    2) fallback: dossier parent YYYY-MM -> YYYY-MM-01
    """
    meta_path = mdir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict) and date_key in meta:
                return pd.to_datetime(meta[date_key])
        except Exception:
            pass

    ym = mdir.parent.name  # "YYYY-MM"
    return pd.to_datetime(f"{ym}-01")


def _read_snapshot_X(mdir: Path) -> pd.DataFrame:
    x_path = mdir / "X_fore_used.parquet"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing: {x_path}")
    X = pd.read_parquet(x_path)
    if len(X) != 1:
        # en général c'est 1 ligne (la date t)
        # si plusieurs, on prend la première ligne
        X = X.iloc[:1].copy()
    return X


def _build_x_long_from_snapshots(
    compare_dir: Path,
    snapshots_subdir: str,
    method: str,
    date_key: str,
) -> pd.DataFrame:
    """
    Retourne un DF long: date, model, feature, x_value
    (x_value = valeur observée de la feature au snapshot)
    """
    snapshots_root = compare_dir / snapshots_subdir
    snap_dirs = _collect_snapshots(snapshots_root, method)

    rows = []
    for mdir in snap_dirs:
        dt = _read_snapshot_date(mdir, date_key=date_key)
        X = _read_snapshot_X(mdir)
        for feat, val in X.iloc[0].items():
            rows.append(
                {
                    "date": dt,
                    "model": method,
                    "feature": str(feat),
                    "x_value": float(val) if pd.notna(val) else np.nan,
                }
            )

    df_x = pd.DataFrame(rows)
    df_x["date"] = pd.to_datetime(df_x["date"])
    return df_x


# -----------------------------
# Plot
# -----------------------------

def _plot_grid(
    df: pd.DataFrame,
    selected_vars: list[str],
    model_order: list[str],
    *,
    poly_deg_linear: int = 1,
    scatter_alpha: float = 0.55,
    s: float = 14.0,
    figsize_per_cell: tuple[float, float] = (3.1, 2.6),
    sharey: bool = False,
) -> plt.Figure:
    """
    df contient: date, model, feature, x_value, shap_value
    """
    n_rows, n_cols = len(selected_vars), len(model_order)
    fig_w = n_cols * figsize_per_cell[0]
    fig_h = n_rows * figsize_per_cell[1]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharey=sharey)
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for c, m in enumerate(model_order):
        df_m = df[df["model"] == m]
        for r, feat in enumerate(selected_vars):
            ax = axes[r, c]
            sub = df_m[df_m["feature"] == feat].copy()

            x = sub["x_value"].to_numpy(dtype=float)
            y = sub["shap_value"].to_numpy(dtype=float)
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]

            # points
            ax.scatter(x, y, alpha=scatter_alpha, s=s, facecolors="white", edgecolors="black", linewidths=0.6)

            # droite (polyfit deg=1)
            if len(x) >= (poly_deg_linear + 2):
                try:
                    coefs = np.polyfit(x, y, deg=poly_deg_linear)
                    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                    y_line = np.polyval(coefs, x_line)
                    ax.plot(x_line, y_line, linewidth=1.2)
                except Exception:
                    pass

            # habillage
            ax.axhline(0, color="lightgray", lw=0.8)
            ax.grid(axis="y", linestyle=":", alpha=0.25)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

            if r == 0:
                ax.set_title(m, fontsize=9, pad=4)
            if c == 0:
                ax.set_ylabel(feat, fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel("Observed values", fontsize=8)

    plt.tight_layout(h_pad=0.8, w_pad=0.8)
    return fig


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--compare-dir", type=str, required=True)
    p.add_argument("--snapshots-subdir", type=str, default="oos_snapshots")

    # SHAP snapshots (déjà exporté)
    p.add_argument(
        "--shap-snapshots",
        type=str,
        default="shap_snapshots_long.csv",
        help="CSV long: date, model, feature, shap_value, y_true",
    )

    # modèles à inclure (doivent exister dans oos_snapshots et dans le CSV SHAP)
    p.add_argument("--methods", type=str, default="LINREG", help="Ex: 'LINREG,RIDGE,GBDT'")

    # variables à tracer
    p.add_argument("--vars", type=str, required=True, help="Ex: 'INDPRO,SP500,BUSLOANS,M2SL,TB3MS'")

    p.add_argument("--date-key", type=str, default="date", help="Clé date dans meta.json si présente")
    p.add_argument("--out-name", type=str, default="functional_form_grid.png")
    p.add_argument("--dpi", type=int, default=160)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    compare_dir = Path(args.compare_dir)

    shap_path = Path(args.shap_snapshots)
    if not shap_path.exists():
        shap_path = compare_dir / args.shap_snapshots
    if not shap_path.exists():
        raise FileNotFoundError(f"SHAP snapshots introuvable: {shap_path}")

    df_shap = pd.read_csv(shap_path)
    if "date" not in df_shap.columns:
        raise ValueError("Le CSV SHAP doit contenir une colonne 'date'.")
    df_shap["date"] = pd.to_datetime(df_shap["date"])

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    selected_vars = [v.strip() for v in args.vars.split(",") if v.strip()]

    # On construit X_long (valeurs observées) pour chaque méthode
    x_parts = []
    for m in methods:
        df_x = _build_x_long_from_snapshots(
            compare_dir=compare_dir,
            snapshots_subdir=args.snapshots_subdir,
            method=m,
            date_key=args.date_key,
        )
        x_parts.append(df_x)
    df_x_all = pd.concat(x_parts, axis=0, ignore_index=True)

    # Join X values with shap_value
    # (df_shap contient model=LINREG etc. si ton export l’a écrit comme ça)
    df = df_shap.merge(df_x_all, on=["date", "model", "feature"], how="inner")

    # Filtrer sur vars demandées
    df = df[df["feature"].isin(selected_vars)].copy()
    if df.empty:
        raise ValueError(
            "Join vide. Vérifie:\n"
            "- que 'model' dans shap_snapshots_long.csv correspond à tes méthodes (LINREG...)\n"
            "- que 'date' est cohérent (meta.json date ou YYYY-MM)\n"
            "- que les noms de features correspondent (SP500 vs S&P500 etc.)"
        )

    # ordre colonnes = methods dans l'ordre donné
    model_order = methods

    fig = _plot_grid(
        df=df,
        selected_vars=selected_vars,
        model_order=model_order,
        poly_deg_linear=1,
        scatter_alpha=0.55,
        s=14,
        figsize_per_cell=(3.1, 2.6),
        sharey=False,
    )

    out_dir = compare_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
