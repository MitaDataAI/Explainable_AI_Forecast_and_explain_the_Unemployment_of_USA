# feature_core.py
# Code commun réutilisable pour :
# - charger la table PostgreSQL en format long (date, series_id, value)
# - sélectionner des séries
# - transformer long -> wide
# - (optionnel) sauvegarder en CSV avec un chemin relatif projet

import getpass
from pathlib import Path
import psycopg
import pandas as pd


def load_from_postgres(
    host: str = "127.0.0.1",
    port: int = 5432,
    dbname: str = "unemployment_usa",
    user: str = "postgres",
    table: str = "unemployment_dataset_usa",
    password: str | None = None,
) -> pd.DataFrame:
    """
    Connexion PostgreSQL → chargement de la table en DataFrame (RAM) → fermeture.

    Retourne un DataFrame long avec colonnes: date, series_id, value

    - Si password=None, demande le mot de passe via getpass.
    - Pour automatiser (sans prompt), passe password="..."
    """
    if password is None:
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
    Filtre un DataFrame long (date, series_id, value) en gardant uniquement les series_id demandés.
    """
    required = {"date", "series_id", "value"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Colonnes attendues {required}, colonnes trouvées {list(df.columns)}"
        )

    return df[df["series_id"].isin(series_list)].copy()


def long_to_wide(df_long: pd.DataFrame, aggfunc: str = "first") -> pd.DataFrame:
    """
    Transforme long → wide (index=date, colonnes=series_id, valeurs=value).
    pivot_table est robuste si doublons (date, series_id).
    """
    required = {"date", "series_id", "value"}
    if not required.issubset(df_long.columns):
        raise ValueError(
            f"Colonnes attendues {required}, colonnes trouvées {list(df_long.columns)}"
        )

    df_wide = (
        df_long.pivot_table(
            index="date",
            columns="series_id",
            values="value",
            aggfunc=aggfunc
        )
        .sort_index()
    )

    return df_wide


def summarize_wide(df_wide: pd.DataFrame, top_n: int = 10) -> None:
    """
    Petit résumé utile en debug / logs.
    """
    print("\n=== RÉSUMÉ DATASET (WIDE) ===")
    print("Shape :", df_wide.shape)
    print("Colonnes :", list(df_wide.columns))

    print(f"\nNaN par colonne (top {top_n}) :")
    print(df_wide.isna().sum().sort_values(ascending=False).head(top_n))

    print("\nAperçu :")
    print(df_wide.head())


def project_root_from_script(script_file: str, parents_up: int = 2) -> Path:
    """
    Calcule la racine projet à partir d'un script.
    Ex: si le script est dans 2_data_processing/feature_engineering/, parents_up=2
    remonte vers la racine du projet.
    """
    return Path(script_file).resolve().parents[parents_up]


def get_features_output_dir(script_file: str) -> Path:
    """
    Dossier de sortie standard: <project_root>/1_data/processed/features
    (créé automatiquement si absent)
    """
    project_root = project_root_from_script(script_file, parents_up=2)
    output_dir = project_root / "1_data" / "processed" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_wide_to_csv(df_wide: pd.DataFrame, script_file: str, filename: str) -> Path:
    """
    Sauvegarde df_wide en CSV dans <project_root>/1_data/processed/features/<filename>
    Retourne le chemin complet.
    """
    output_dir = get_features_output_dir(script_file)
    output_path = output_dir / filename
    df_wide.to_csv(output_path, index=True)  # index=date conservé
    return output_path