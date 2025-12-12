import getpass
import psycopg
import pandas as pd
import numpy as np
from pathlib import Path


def load_from_postgres(
    host: str = "127.0.0.1",
    port: int = 5432,
    dbname: str = "unemployment_usa",
    user: str = "postgres",
    table: str = "unemployment_dataset_usa",
) -> pd.DataFrame:
    """
    Connexion PostgreSQL → chargement de la table en DataFrame (RAM) → fermeture.
    Attend : date, series_id, value
    """
    password = getpass.getpass(f"Mot de passe PostgreSQL ({user}): ")

    conn = psycopg.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
    )

    query = f"SELECT date, series_id, value FROM {table};"
    df = pd.read_sql(query, conn)
    conn.close()

    return df


def select_series_long(df: pd.DataFrame, series_list: list[str]) -> pd.DataFrame:
    """
    Filtre le format long pour ne garder que les series_id souhaités.
    """
    required = {"date", "series_id", "value"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Colonnes attendues {required}, colonnes trouvées {list(df.columns)}"
        )

    return df[df["series_id"].isin(series_list)].copy()


def long_to_wide(df_long: pd.DataFrame, aggfunc: str = "first") -> pd.DataFrame:
    """
    Transforme long → wide (index=date, colonnes=series_id).
    """
    return (
        df_long
        .pivot_table(
            index="date",
            columns="series_id",
            values="value",
            aggfunc=aggfunc
        )
        .sort_index()
    )


def summarize_wide(df_wide: pd.DataFrame, top_n: int = 10) -> None:
    """
    Résumé rapide du DataFrame wide.
    """
    print("\n=== RÉSUMÉ DATASET (WIDE) ===")
    print("Shape :", df_wide.shape)
    print("Colonnes :", list(df_wide.columns))

    print(f"\nNaN par colonne (top {top_n}) :")
    print(df_wide.isna().sum().sort_values(ascending=False).head(top_n))

    print("\nAperçu :")
    print(df_wide.head())


def apply_stationarity_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique les transformations de stationnarité décidées en amont.
    Les noms de colonnes sont volontairement simples (sans _d1, _d2).
    """
    df = df.copy()

    # Variable cible : variation du taux de chômage
    if "UNRATE" in df.columns:
        df["UNRATE"] = df["UNRATE"].diff(1)

    # Variables I(1) : log-diff
    for col in ["INDPRO", "RPI", "SP500", "DPCERA3M086SBEA"]:
        if col in df.columns:
            df[col] = np.log(df[col]).diff(1)

    # Variable en taux : diff simple
    if "TB3MS" in df.columns:
        df["TB3MS"] = df["TB3MS"].diff(1)

    # Variables I(2) : double différenciation
    for col in ["BUSLOANS", "CPIAUCSL", "M2SL", "OILPRICEx"]:
        if col in df.columns:
            df[col] = df[col].diff(1).diff(1)

    # USREC : variable binaire → inchangée

    return df


def main():
    # 1) Chargement des données brutes
    df = load_from_postgres()

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
        "OILPRICEx",
        "M2SL",
        "USREC",
    ]

    df_sel = select_series_long(df, col_names)

    # 3) Long → Wide
    df_wide = long_to_wide(df_sel)

    # 4) Vérification
    summarize_wide(df_wide)

    # 5) Stationnarisation (application directe)
    df_features = apply_stationarity_transformations(df_wide)

    # Suppression des lignes induites par les différenciations
    df_features = df_features.dropna()

    print("\n=== FEATURES FINALES ===")
    print("Shape :", df_features.shape)
    print("Colonnes :", list(df_features.columns))
    print(df_features.head())

    # 6) Chemin relatif vers le dossier features
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "1_data" / "processed" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "unemployment_features.csv"

    # 7) Sauvegarde finale (CSV)
    df_features.to_csv(output_path, index=True)  # index = date
    print(f"\nDonnées finales sauvegardées ici :\n{output_path}")


if __name__ == "__main__":
    main()