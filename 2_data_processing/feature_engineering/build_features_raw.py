from feature_core import (
    load_from_postgres,
    select_series_long,
    long_to_wide,
    summarize_wide,
    save_wide_to_csv,
)


def main():
    # 1) Chargement des données brutes
    df_long = load_from_postgres()

    # 2) Sélection des séries
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

    # 3) Long → Wide (AUCUNE transformation)
    df_wide = long_to_wide(df_sel)

    # 4) Vérification
    summarize_wide(df_wide)

    # 5) Sauvegarde
    output_path = save_wide_to_csv(
        df_wide=df_wide,
        script_file=__file__,
        filename="unemployment_features_raw.csv",
    )

    print(f"\nFeatures RAW sauvegardées ici :\n{output_path}")


if __name__ == "__main__":
    main()