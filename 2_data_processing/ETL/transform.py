# ETL/transform.py

import pandas as pd
import sys
from pathlib import Path

# Ajout du dossier racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import (
    FRED_MD_INPUT,
    FRED_MD_RAW_CSV,
    FRED_MD_TCODE_CSV,
    PROCESSED_DIR,
)


def prepare_fred_md_tables():
    """
    Lit le CSV FRED-MD brut et fabrique deux DataFrames :
    - long_df : (date, series_id, value) au format long
    - transform_df : (series_id, transform_code)
    """
    if not FRED_MD_INPUT.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {FRED_MD_INPUT}.\n"
            f"Assure-toi que le fichier FRED-MD est bien présent dans 1_data/raw."
        )

    print(f"Lecture du fichier {FRED_MD_INPUT} ...")
    # On force l'encodage en latin-1 pour bien lire le fichier brut si accents
    df = pd.read_csv(FRED_MD_INPUT, encoding="latin1").dropna(how="all")

    if "sasdate" not in df.columns:
        raise ValueError("La colonne 'sasdate' est introuvable dans le CSV FRED-MD.")

    # Renommer la colonne sasdate -> date
    df = df.rename(columns={"sasdate": "date"})

    # Première ligne = codes de transformation (tcode)
    tcode_row = df.iloc[0].copy()

    # Les lignes suivantes = données
    data = df.iloc[1:].copy()

    # Convertir les dates en type date
    data["date"] = pd.to_datetime(data["date"]).dt.date

    # Passage au format long : (date, series_id, value)
    long_df = data.melt(
        id_vars=["date"],
        var_name="series_id",
        value_name="value",
    )

    # Table des codes de transformation : (series_id, transform_code)
    tcode_series = tcode_row.drop("date")
    transform_df = tcode_series.reset_index()
    transform_df.columns = ["series_id", "transform_code"]
    transform_df["transform_code"] = transform_df["transform_code"].astype(int)

    return long_df, transform_df


def save_outputs(long_df: pd.DataFrame, transform_df: pd.DataFrame):
    """Sauvegarde les DataFrames en CSV dans le dossier cleaned/."""
    print(f"Sauvegarde dans {PROCESSED_DIR} ...")

    # On écrit en UTF-8 pour une compatibilité propre avec PostgreSQL
    long_df.to_csv(FRED_MD_RAW_CSV, index=False, encoding="utf-8")
    transform_df.to_csv(FRED_MD_TCODE_CSV, index=False, encoding="utf-8")

    print("Fichiers enregistrés :")
    print(f" - {FRED_MD_RAW_CSV}")
    print(f" - {FRED_MD_TCODE_CSV}")


if __name__ == "__main__":
    long_df, transform_df = prepare_fred_md_tables()
    print(f"Lignes fred_md_raw : {len(long_df)}")
    print(f"Lignes fred_md_transform : {len(transform_df)}")

    save_outputs(long_df, transform_df)