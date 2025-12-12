import numpy as np
from feature_core import (
    load_from_postgres,
    select_series_long,
    long_to_wide,
    summarize_wide,
    save_wide_to_csv,
)


def apply_stationarity_transformations(df):
    """
    Application déterministe des transformations de stationnarité
    (décidées dans les notebooks).

    Stoppe explicitement si une colonne attendue est manquante.
    """
    expected_cols = [
        "UNRATE",
        "INDPRO",
        "RPI",
        "SP500",
        "DPCERA3M086SBEA",
        "TB3MS",
        "BUSLOANS",
        "CPIAUCSL",
        "M2SL",
        "OILPRICEX",
        "USREC",
    ]

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"\nColonnes manquantes dans df_wide : {missing}\n"
            f"Colonnes disponibles : {list(df.columns)}\n"
            f"➡️ Vérifie les series_id présents dans la base PostgreSQL."
        )

    df = df.copy()

    # -------------------------
    # Variable cible
    # -------------------------
    # UNRATE : variation en points
    df["UNRATE"] = df["UNRATE"].diff(1)

    # -------------------------
    # Variables I(1) : log-diff
    # -------------------------
    for col in ["INDPRO", "RPI", "SP500", "DPCERA3M086SBEA"]:
        df[col] = np.log(df[col]).diff(1)

    # -------------------------
    # Variable en taux
    # -------------------------
    df["TB3MS"] = df["TB3MS"].diff(1)

    # -------------------------
    # Variables I(2)
    # -------------------------
    for col in ["BUSLOANS", "CPIAUCSL", "M2SL", "OILPRICEX"]:
        df[col] = df[col].diff(1).diff(1)

    # USREC : variable binaire → inchangée

    return df.dropna()


def main():
    # 1) Chargement des données brutes (long)
    df_long = load_from_postgres()

    # 2) Sélection des séries macro
    col_names = [
        "UNRATE",
        "TB3MS",
        "RPI",
        "INDPRO",
        "DPCERA3M086SBEA",
        "SP500",
        "BUSLOANS",
        "CPIAUCSL",
        "OILPRICEX",
        "M2SL",
        "USREC",
    ]

    df_sel = select_series_long(df_long, col_names)

    # 3) Long → Wide
    df_wide = long_to_wide(df_sel)

    # 🔎 Debug utile (si problème futur)
    print("\nColonnes df_wide :", list(df_wide.columns))

    # 4) Stationnarisation
    df_features = apply_stationarity_transformations(df_wide)

    # 5) Vérification
    summarize_wide(df_features)

    # 6) Sauvegarde
    output_path = save_wide_to_csv(
        df_wide=df_features,
        script_file=__file__,
        filename="unemployment_features_stationary.csv",
    )

    print(f"\nFeatures STATIONNARISÉES sauvegardées ici :\n{output_path}")


if __name__ == "__main__":
    main()