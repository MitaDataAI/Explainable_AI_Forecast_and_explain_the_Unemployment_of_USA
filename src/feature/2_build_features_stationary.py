# --------------------------------------------------
# Bootstrap PYTHONPATH (AVANT tout import projet)
# --------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "configs").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

if not (PROJECT_ROOT / "configs").exists():
    raise RuntimeError("Impossible de trouver la racine projet (dossier 'configs' introuvable).")

sys.path.append(str(PROJECT_ROOT))

# (optionnel debug)
print("PROJECT_ROOT =", PROJECT_ROOT)
print("EXISTS configs? =", (PROJECT_ROOT / "configs").exists())

# --------------------------------------------------
# Imports standards
# --------------------------------------------------
import numpy as np
import pandas as pd

# --------------------------------------------------
# Imports projet
# --------------------------------------------------
from feature_core import (
    load_from_postgres,
    select_series_long,
    summarize_long,
    save_long_to_parquet,  # ✅ parquet
)

from configs.feature_engineering import (
    SERIES_COLS,
    STATIONARITY_RULES,
    VARIABLE_LABELS,
    REPORT_FILENAME,  # ex: "stationarity_transformations_report.csv"
    DROP_FIRST_N,     # ex: 12
)

# --------------------------------------------------
# Nom du fichier de sortie stationnarisé (fallback safe)
# --------------------------------------------------
STATIONARY_LONG_FILENAME = getattr(
    __import__("configs.feature_engineering", fromlist=["x"]),
    "STATIONARY_LONG_FILENAME",
    "unemployment_features_stationary_long.csv",
)


def _apply_diff_long(df_s: pd.DataFrame, order: int, lags: int) -> pd.DataFrame:
    out = df_s.copy()
    v = out["value"]
    for _ in range(order):
        v = v.diff(lags)
    out["value"] = v
    return out


def _apply_logdiff_long(df_s: pd.DataFrame, order: int, lags: int) -> pd.DataFrame:
    out = df_s.copy()
    v = out["value"]

    if (v <= 0).any():
        bad_dates = out.loc[v <= 0, "date"].head(5).astype(str).tolist()
        raise ValueError(
            f"{out['series_id'].iloc[0]} contient des valeurs <= 0 (ex dates {bad_dates}). "
            f"Impossible d'appliquer log()."
        )

    v = np.log(v)
    for _ in range(order):
        v = v.diff(lags)

    out["value"] = v
    return out


def _transformation_to_text(method: str, order: int, lags: int) -> str:
    if method == "none":
        return "none"

    if method == "diff" and order == 1:
        base = "changes"
    elif method == "logdiff" and order == 1:
        base = "log changes"
    elif method == "diff" and order == 2:
        base = "second order changes"
    elif method == "logdiff" and order == 2:
        base = "second order log changes"
    else:
        base = f"{method} (order={order})"

    return f"{base} (Δ{lags})"


def build_stationarity_report_long(rules: dict, labels: dict, include_none: bool = False) -> pd.DataFrame:
    rows = []
    for series_id, rule in rules.items():
        if series_id == "_default":
            continue

        method = rule.get("method", rule.get("transform"))
        order = int(rule.get("order", 0))
        lags = int(rule.get("lags", 1))
        text = _transformation_to_text(method, order, lags)

        if (not include_none) and text == "none":
            continue

        rows.append(
            {
                "Variable": labels.get(series_id, series_id),
                "Transformation": text,
                "Name in the FRED-MD database": series_id,
            }
        )

    df = pd.DataFrame(rows)

    # ordre SERIES_COLS
    order_map = {sid: i for i, sid in enumerate(SERIES_COLS)}
    df["_order"] = df["Name in the FRED-MD database"].map(order_map).fillna(9999).astype(int)
    return df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def _drop_first_n_rows_per_series(df_long: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Règle demandée :
    - calcule le rang (0,1,2,...) dans chaque série triée par date
    - supprime TOUJOURS les n premières lignes de chaque série (peu importe NaN ou non)
    - ne supprime pas les NaN ailleurs (milieu/fin)
    """
    if n is None:
        n = 0
    n = int(n)
    if n <= 0:
        return df_long

    df_long = df_long.sort_values(["series_id", "date"]).copy()
    df_long["_rn"] = df_long.groupby("series_id").cumcount()

    before = len(df_long)
    df_out = df_long[df_long["_rn"] >= n].drop(columns=["_rn"]).reset_index(drop=True)

    dropped = before - len(df_out)
    if dropped:
        print(f"[stationarity-long] Dropped {dropped} rows (first {n} observations per series).")

    return df_out


def apply_stationarity_transformations_long(
    df_long: pd.DataFrame,
    rules: dict = STATIONARITY_RULES,
) -> pd.DataFrame:
    required = {"date", "series_id", "value"}
    if not required.issubset(df_long.columns):
        raise ValueError(f"Colonnes attendues {required}, colonnes trouvées {list(df_long.columns)}")

    df_long = df_long.copy()
    df_long["date"] = pd.to_datetime(df_long["date"])
    df_long = df_long.sort_values(["series_id", "date"])

    default_rule = rules.get("_default")
    if default_rule is None:
        raise KeyError("Règle _default manquante dans STATIONARITY_RULES.")

    out_parts = []

    for sid, df_s in df_long.groupby("series_id", sort=False):
        rule = rules.get(sid, default_rule)
        method = rule.get("method", rule.get("transform"))
        order = int(rule.get("order", 1))
        lags = int(rule.get("lags", 1))

        if method == "none":
            out = df_s.copy()
        elif method == "diff":
            out = _apply_diff_long(df_s, order=order, lags=lags)
        elif method == "logdiff":
            out = _apply_logdiff_long(df_s, order=order, lags=lags)
        else:
            raise ValueError(f"Transformation inconnue '{method}' pour {sid}")

        out_parts.append(out)

    df_out = pd.concat(out_parts, ignore_index=True)

    # --------------------------------------------------
    # RÈGLE DEMANDÉE : supprimer les N premières lignes de chaque série
    # --------------------------------------------------
    df_out = _drop_first_n_rows_per_series(df_out, n=DROP_FIRST_N)

    return df_out


def main():
    # 1) Charger les données (LONG, comme SQL)
    df_long = load_from_postgres()

    # 2) Filtrer les séries utiles (config)
    df_sel = select_series_long(df_long, SERIES_COLS)

    # 3) Rapport des transformations (lisible)
    df_report = build_stationarity_report_long(
        rules=STATIONARITY_RULES,
        labels=VARIABLE_LABELS,
        include_none=False,
    )
    print("\n=== STATIONARITY TRANSFORMATIONS (REPORT) ===")
    print(df_report.to_string(index=False))

    # 3bis) Sauvegarde du report -> PARQUET
    report_parquet = Path(REPORT_FILENAME).with_suffix(".parquet").name
    report_path = save_long_to_parquet(
        df_long=df_report,
        script_file=__file__,
        filename=report_parquet,
    )
    print(f"\nRapport (PARQUET) sauvegardé ici :\n{report_path}")

    # 4) Appliquer la stationnarité en LONG (sans pivot)
    df_stationary_long = apply_stationarity_transformations_long(df_sel)

    # 5) Vérification (LONG)
    summarize_long(df_stationary_long)

    # 6) Sauvegarde du dataset final -> PARQUET
    stationary_parquet = Path(STATIONARY_LONG_FILENAME).with_suffix(".parquet").name
    output_path = save_long_to_parquet(
        df_long=df_stationary_long,
        script_file=__file__,
        filename=stationary_parquet,
    )
    print(f"\nFeatures STATIONNARISÉES (LONG) (PARQUET) sauvegardées ici :\n{output_path}")


if __name__ == "__main__":
    main()