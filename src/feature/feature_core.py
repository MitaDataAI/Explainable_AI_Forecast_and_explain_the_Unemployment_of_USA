# feature_core.py
# Code commun réutilisable pour :
# - charger la table PostgreSQL en format LONG (date, series_id, value)
# - sélectionner des séries
# - résumer des données LONG
# - sauvegarder / relire des fichiers LONG (CSV + Parquet)
# ⚠️ Aucun support WIDE volontairement (choix d'architecture)

from __future__ import annotations

import os
import getpass
from pathlib import Path
from typing import Iterable, Optional

import psycopg
import pandas as pd


# -------------------------------------------------------------------
# 1) Chargement depuis PostgreSQL (LONG)
# -------------------------------------------------------------------
def load_from_postgres(
    host: str = "127.0.0.1",
    port: int = 5432,
    dbname: str = "unemployment_usa",
    user: str = "postgres",
    table: str = "macro.observations_monthly",  # accepte schema.table
    password: Optional[str] = None,
) -> pd.DataFrame:
    """
    Connexion PostgreSQL → chargement de la table en DataFrame → fermeture.

    Retourne un DataFrame LONG avec colonnes :
        date | series_id | value

    - Si password=None :
        1) essaie la variable d’environnement PGPASSWORD
        2) sinon demande via getpass (mode dev)
    """
    if password is None:
        password = os.getenv("PGPASSWORD")
    if password is None:
        password = getpass.getpass(f"Mot de passe PostgreSQL ({user}): ")

    conn = psycopg.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
    )

    try:
        query = f"SELECT date, series_id, value FROM {table};"
        df = pd.read_sql(query, conn, parse_dates=["date"])
    finally:
        conn.close()

    return df


# -------------------------------------------------------------------
# 2) Sélection des séries (LONG)
# -------------------------------------------------------------------
def select_series_long(df: pd.DataFrame, series_list: Iterable[str]) -> pd.DataFrame:
    """
    Filtre un DataFrame LONG en gardant uniquement les series_id demandés.
    """
    required = {"date", "series_id", "value"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Colonnes attendues {required}, colonnes trouvées {list(df.columns)}"
        )

    series_set = set(series_list)
    return df[df["series_id"].isin(series_set)].copy()


# -------------------------------------------------------------------
# 3) Résumé / contrôle qualité (LONG)
# -------------------------------------------------------------------
def summarize_long(df_long: pd.DataFrame, top_n: int = 10) -> None:
    """
    Résumé rapide pour debug / logs sur données LONG.
    """
    required = {"date", "series_id", "value"}
    if not required.issubset(df_long.columns):
        raise ValueError(
            f"Colonnes attendues {required}, colonnes trouvées {list(df_long.columns)}"
        )

    # sécurité : forcer date en datetime si ce n'est pas le cas
    if not pd.api.types.is_datetime64_any_dtype(df_long["date"]):
        df_long = df_long.copy()
        df_long["date"] = pd.to_datetime(df_long["date"], errors="coerce")

    print("\n=== RÉSUMÉ DATASET (LONG) ===")
    print("Shape :", df_long.shape)

    print("Dates :", df_long["date"].min(), "→", df_long["date"].max())
    print("Nb séries :", df_long["series_id"].nunique())

    print(f"\nNb observations par série (top {top_n}) :")
    print(df_long["series_id"].value_counts().head(top_n))

    print(f"\nValeurs manquantes sur value (top {top_n} séries) :")
    print(df_long[df_long["value"].isna()]["series_id"].value_counts().head(top_n))

    print("\nAperçu :")
    print(df_long.head())


# -------------------------------------------------------------------
# 4) Gestion des chemins projet
# -------------------------------------------------------------------
def project_root_from_script(script_file: str, parents_up: int = 2) -> Path:
    """
    Calcule la racine projet à partir d'un script.
    Ex :
      script = 2_data_processing/feature_engineering/build_*.py
      parents_up = 2 → racine projet
    """
    return Path(script_file).resolve().parents[parents_up]


def get_features_output_dir(script_file: str) -> Path:
    """
    Dossier standard de sortie :
      <project_root>/1_data/processed/features
    """
    project_root = project_root_from_script(script_file, parents_up=2)
    output_dir = project_root / "1_data" / "processed" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# -------------------------------------------------------------------
# 5) Sauvegarde / lecture CSV (LONG)
# -------------------------------------------------------------------
def save_long_to_csv(df_long: pd.DataFrame, script_file: str, filename: str) -> Path:
    """
    Sauvegarde un DataFrame LONG en CSV.
    """
    output_dir = get_features_output_dir(script_file)
    output_path = output_dir / filename
    df_long.to_csv(output_path, index=False)
    return output_path


def read_long_from_csv(script_file: str, filename: str) -> pd.DataFrame:
    """
    Relit un CSV LONG depuis le dossier features.
    Utile pour chaîner :
      RAW → STATIONARY sans recharger PostgreSQL.
    """
    path = get_features_output_dir(script_file) / filename
    return pd.read_csv(path, parse_dates=["date"])


# -------------------------------------------------------------------
# 6) Sauvegarde / lecture Parquet (LONG)
# -------------------------------------------------------------------
def save_long_to_parquet(df_long: pd.DataFrame, script_file: str, filename: str) -> Path:
    """
    Sauvegarde un DataFrame LONG en Parquet.
    Recommandé pour Feast (Arrow/Dask friendly).
    """
    output_dir = get_features_output_dir(script_file)
    output_path = output_dir / filename
    df_long.to_parquet(output_path, index=False)
    return output_path


def read_long_from_parquet(script_file: str, filename: str) -> pd.DataFrame:
    """
    Relit un Parquet LONG depuis le dossier features.
    """
    path = get_features_output_dir(script_file) / filename
    df = pd.read_parquet(path)

    # Par sécurité, on force date en datetime si nécessaire
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df