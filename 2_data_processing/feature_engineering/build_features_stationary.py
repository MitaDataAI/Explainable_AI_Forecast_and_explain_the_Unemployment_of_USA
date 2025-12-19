import numpy as np
import pandas as pd

from feature_core import (
    load_from_postgres,
    select_series_long,
    long_to_wide,
    summarize_wide,
    save_wide_to_csv,
)

from config import (
    STATIONARITY_RULES,
    SERIES_COLS,
    OUTPUT_FILENAME,
    VARIABLE_LABELS,
)


def _validate_expected_columns(df, expected_cols):
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"\nColonnes manquantes dans df_wide : {missing}\n"
            f"Colonnes disponibles : {list(df.columns)}\n"
            f"➡️ Vérifie les series_id présents dans la base PostgreSQL."
        )


def _apply_diff(series, order: int, lags: int):
    """Apply order-times differencing with step = lags (Δ_lags)."""
    out = series
    for _ in range(order):
        out = out.diff(lags)
    return out


def _apply_logdiff(series, order: int, lags: int, col_name: str):
    """Apply log then differencing; stop if non-positive values exist."""
    if (series <= 0).any():
        bad_idx = series.index[series <= 0][:5].tolist()
        raise ValueError(
            f"{col_name} contient des valeurs <= 0 (ex: index {bad_idx}). "
            f"Impossible d'appliquer log()."
        )
    out = np.log(series)
    out = _apply_diff(out, order=order, lags=lags)
    return out


def _transformation_to_text(method: str, order: int, lags: int) -> str:
    """
    Convertit (method, order, lags) en texte lisible humain.
    Exemples :
      diff,1,12    -> changes (Δ12)
      logdiff,1,3  -> log changes (Δ3)
      diff,2,1     -> second order changes (Δ1)
      logdiff,2,3  -> second order log changes (Δ3)
      none         -> none
    """
    if method == "none":
        return "none"

    base = None
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


def build_stationarity_report(rules, labels, include_none: bool = False) -> pd.DataFrame:
    """
    Construit un tableau "rapport" :
      Variable | Transformation | Name in the FRED-MD database
    """
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

    # On garde l'ordre de SERIES_COLS (plus logique qu'un tri alphabétique)
    df = pd.DataFrame(rows)
    order_map = {sid: i for i, sid in enumerate(SERIES_COLS)}
    df["_order"] = df["Name in the FRED-MD database"].map(order_map).fillna(9999).astype(int)
    df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    return df


def apply_stationarity_transformations(df, rules=STATIONARITY_RULES):
    """
    Application déterministe des transformations de stationnarité,
    pilotée par config (supporte _default + method/lags/order).

    - Tri temporel avant diff()
    - Stoppe explicitement si colonne manquante (pour SERIES_COLS)
    - Stoppe si log demandé mais valeurs <= 0
    """
    # On valide uniquement les séries attendues (pas _default)
    expected_cols = [c for c in SERIES_COLS if c in df.columns]
    _validate_expected_columns(df, expected_cols=SERIES_COLS)

    df = df.sort_index().copy()

    default_rule = rules.get("_default")

    # On applique aux colonnes du df (pas aux clés du dict)
    for col in df.columns:
        rule = rules.get(col, default_rule)
        if rule is None:
            raise KeyError(f"Aucune règle trouvée pour {col} et pas de règle _default.")

        method = rule.get("method", rule.get("transform"))
        order = int(rule.get("order", 1))
        lags = int(rule.get("lags", 1))

        if method == "diff":
            df[col] = _apply_diff(df[col], order=order, lags=lags)

        elif method == "logdiff":
            df[col] = _apply_logdiff(df[col], order=order, lags=lags, col_name=col)

        elif method == "none":
            df[col] = df[col]

        else:
            raise ValueError(f"Transformation inconnue '{method}' pour {col}")

    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        print(f"[stationarity] Dropped {dropped} rows after transformations.")

    return df


def main():
    # 1) Chargement des données brutes (long)
    df_long = load_from_postgres()

    # 2) Sélection des séries macro (piloté par config)
    df_sel = select_series_long(df_long, SERIES_COLS)

    # 3) Long → Wide
    df_wide = long_to_wide(df_sel)

    # 🔎 Debug utile (si problème futur)
    print("\nColonnes df_wide :", list(df_wide.columns))

    # 3bis) Rapport lisible humain
    df_report = build_stationarity_report(
        rules=STATIONARITY_RULES,
        labels=VARIABLE_LABELS,
        include_none=False,  # mets True si tu veux inclure USREC
    )

    print("\n=== STATIONARITY TRANSFORMATIONS (REPORT) ===")
    print(df_report.to_string(index=False))

    # (optionnel) sauvegarder le rapport
    report_path = save_wide_to_csv(
        df_wide=df_report,
        script_file=__file__,
        filename="stationarity_transformations_report.csv",
    )
    print(f"\nRapport des transformations sauvegardé ici :\n{report_path}")

    # 4) Stationnarisation
    df_features = apply_stationarity_transformations(df_wide)

    # 5) Vérification
    summarize_wide(df_features)

    # 6) Sauvegarde dataset final
    output_path = save_wide_to_csv(
        df_wide=df_features,
        script_file=__file__,
        filename=OUTPUT_FILENAME,
    )

    print(f"\nFeatures STATIONNARISÉES sauvegardées ici :\n{output_path}")


if __name__ == "__main__":
    main()