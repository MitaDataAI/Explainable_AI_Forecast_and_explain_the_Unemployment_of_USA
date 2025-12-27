from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from explainable_ai_forecast.experiments.model_evaluation_compare_load import (
    load_runs,
    runs_to_long,
)

# ============================================================
# CONFIG
# ============================================================

ARTIFACTS_ROOT = "artifacts/experiments"

RUN_IDS = [
    "91e987618f30f5f4",  # linear
    "1543452ae9c1c8c8",  # AR(1)
    "9a2891c636365b0c",  # AR(p auto)
]

H = 12
MODE = "aligne"  # "aligne" ou "decale"

OUT_DIR = Path("artifacts/experiments/comparison/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# PLOTS UTILS
# ============================================================

def build_wide_from_long(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme un DataFrame long (date, method, pred, true)
    en un tableau large : une colonne par méthode + la vérité.
    """
    df = df_long.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    true_by_date = df.groupby("date")["true"].median().rename("true")
    wide = df.pivot_table(index="date", columns="method", values="pred", aggfunc="mean").sort_index()
    wide = wide.join(true_by_date, how="left")

    print("✅ wide prêt :", wide.shape, "| Colonnes :", wide.columns.tolist())
    return wide


def make_color_map(methods: list[str]) -> dict:
    """
    Crée une palette de couleurs cohérente pour chaque méthode.
    """
    cmap = {}
    base_colors = plt.cm.get_cmap("tab10", len(methods))
    for i, m in enumerate(methods):
        cmap[m] = base_colors(i)
    cmap["true"] = "white"  # la vérité en blanc (fond noir)
    return cmap


def get_split_dates(segments):
    """
    Extrait les dates de séparation à partir d'une liste de segments :
    [(start, end, label), ...] → liste unique et triée de dates.
    """
    dates = sorted({pd.to_datetime(start) for (start, _, _) in segments})
    return dates


def plot_two_global_charts(
    df_long: pd.DataFrame,
    segments: list[tuple[str, str | None, str]],
    *,
    H: int = 12,
    mode: str = "aligne",   # "aligne" ou "decale"
    save: bool = True,
    title_prefix: str = "",
    out_dir: Path | None = None,
):
    """
    Trace deux graphiques globaux :
    1) Prévisions des modèles
    2) Évolution des erreurs absolues
    avec les dates de séparation affichées sur les deux.
    """
    plt.style.use("dark_background")

    wide = build_wide_from_long(df_long).dropna(subset=["true"])
    methods = [c for c in wide.columns if c != "true"]
    colors = make_color_map(methods)
    split_dates = get_split_dates(segments)

    # 1) Prévisions globales
    plt.figure(figsize=(12, 5.2))
    plt.plot(wide.index, wide["true"], label="true", color=colors["true"], linewidth=2.2)

    for m in methods:
        if mode == "aligne":
            plt.plot(wide.index, wide[m], label=m, color=colors[m], linewidth=1.5)
        elif mode == "decale":
            plt.plot(
                wide.index - pd.DateOffset(months=H),
                wide[m],
                label=f"{m} (−{H}m)",
                color=colors[m],
                linewidth=1.5,
            )
        else:
            raise ValueError("mode inconnu (utiliser 'aligne' ou 'decale')")

    for d in split_dates:
        plt.axvline(d, linestyle="--", alpha=0.6)

    plt.title(f"{title_prefix}Prévisions des modèles (mode={mode}, h={H})".strip())
    plt.xlabel("Date")
    plt.ylabel("Niveau")
    plt.grid(alpha=0.3)
    plt.legend(ncols=2)
    plt.tight_layout()

    if save:
        out_dir = out_dir or Path(".")
        p = out_dir / "global_forecasts.png"
        plt.savefig(p, dpi=150)
        print(f"💾 Figure enregistrée → {p}")

    plt.show()

    # 2) Erreurs absolues
    errs = pd.DataFrame({m: (wide["true"] - wide[m]).abs() for m in methods}, index=wide.index)

    plt.figure(figsize=(12, 5.2))
    for m in methods:
        plt.plot(errs.index, errs[m], label=f"|err| {m}", color=colors[m], linewidth=1.5)

    for d in split_dates:
        plt.axvline(d, linestyle="--", alpha=0.6)

    plt.title(f"{title_prefix}Évolution des erreurs absolues".strip())
    plt.xlabel("Date")
    plt.ylabel("|Erreur|")
    plt.grid(alpha=0.3)
    plt.legend(ncols=2)
    plt.tight_layout()

    if save:
        out_dir = out_dir or Path(".")
        p = out_dir / "global_errors.png"
        plt.savefig(p, dpi=150)
        print(f"💾 Figure enregistrée → {p}")

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    runs = load_runs(ARTIFACTS_ROOT, RUN_IDS)
    long_df = runs_to_long(runs)

    # On construit df_pred_long au format attendu par ton code :
    # colonnes: date, method, pred, true
    df_pred_long = (
        long_df.rename(columns={"y_pred": "pred", "y_true": "true"})
               [["date", "method", "pred", "true", "run_id"]]
               .copy()
    )

    # Pour éviter les doublons de "method" (si tu relances plusieurs fois),
    # tu peux rendre method unique par run_id :
    df_pred_long["method"] = df_pred_long["method"] + "__" + df_pred_long["run_id"].str[:8]

    segments = [
        ("1990-01-01", "1999-12-31", "1990-1999"),
        ("2000-01-01", "2008-07-31", "2000-2008"),
        ("2008-08-01", "2019-11-30", "2008-2019"),
        ("2019-12-01", None,         "2019-fin"),
    ]

    plot_two_global_charts(
        df_pred_long[["date", "method", "pred", "true"]],
        segments,
        H=H,
        mode=MODE,
        save=True,
        title_prefix="",
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()