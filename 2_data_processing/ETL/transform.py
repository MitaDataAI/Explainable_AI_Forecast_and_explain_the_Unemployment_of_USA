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
    NBER_USREC_INPUT,
    NBER_USREC_CSV,
)

# ------------------------------------------------------------
# Normalisation MINIMALE des series_id
# ------------------------------------------------------------
def normalize_series_id(s: str) -> str:
    """
    Normalisation minimale :
    - suppression de '&'
    - suppression des espaces
    - mise en majuscules
    """
    if pd.isna(s):
        return s

    return (
        str(s)
        .replace("&", "")
        .replace(" ", "")
        .upper()
    )


# ------------------------------------------------------------
# 1) Préparation FRED-MD
# ------------------------------------------------------------
def prepare_fred_md_tables():
    """
    Lit le CSV FRED-MD brut et fabrique deux DataFrames :
    - long_df : (date, series_id, value)
    - transform_df : (series_id, transform_code)
    """
    if not FRED_MD_INPUT.exists():
        raise FileNotFoundError(f"Fichier introuvable : {FRED_MD_INPUT}")

    print(f"Lecture du fichier {FRED_MD_INPUT} ...")

    df = pd.read_csv(FRED_MD_INPUT, encoding="latin1").dropna(how="all")

    if "sasdate" not in df.columns:
        raise ValueError("La colonne 'sasdate' est introuvable dans FRED-MD")

    df = df.rename(columns={"sasdate": "date"})

    # Ligne des tcodes
    tcode_row = df.iloc[0].copy()
    data = df.iloc[1:].copy()

    # Dates
    data["date"] = pd.to_datetime(data["date"]).dt.date

    # Format long
    long_df = data.melt(
        id_vars=["date"],
        var_name="series_id",
        value_name="value",
    )

    # 🔧 NORMALISATION series_id (LONG)
    long_df["series_id"] = long_df["series_id"].apply(normalize_series_id)

    # Table des tcodes
    tcode_series = tcode_row.drop("date")
    transform_df = tcode_series.reset_index()
    transform_df.columns = ["series_id", "transform_code"]
    transform_df["transform_code"] = transform_df["transform_code"].astype(int)

    # 🔧 NORMALISATION series_id (TCODES)
    transform_df["series_id"] = transform_df["series_id"].apply(normalize_series_id)

    return long_df, transform_df


def save_outputs(long_df, transform_df):
    """Sauvegarde les fichiers FRED-MD."""
    print(f"Sauvegarde FRED-MD dans {PROCESSED_DIR} ...")

    long_df.to_csv(FRED_MD_RAW_CSV, index=False, encoding="utf-8")
    transform_df.to_csv(FRED_MD_TCODE_CSV, index=False, encoding="utf-8")

    print(f" - {FRED_MD_RAW_CSV}")
    print(f" - {FRED_MD_TCODE_CSV}")


# ------------------------------------------------------------
# 2) Préparation NBER USREC
# ------------------------------------------------------------
def prepare_nber_usrec():
    """
    Lit le fichier brut USREC (observation_date, USREC)
    et renvoie un DataFrame AU FORMAT LONG :
    (date, series_id, value) avec series_id = 'USREC'.
    """
    primary_path = Path(NBER_USREC_INPUT)
    fallback_path = PROJECT_ROOT / "1_data" / "raw" / "2025-11-NBER-USREC.csv"

    if primary_path.exists():
        usrec_path = primary_path
    elif fallback_path.exists():
        usrec_path = fallback_path
    else:
        raise FileNotFoundError(
            f"USREC introuvable : {primary_path} ni {fallback_path}"
        )

    print(f"Lecture du fichier USREC {usrec_path} ...")

    df = pd.read_csv(usrec_path)

    expected = {"observation_date", "USREC"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Colonnes attendues : {expected}, trouvé : {set(df.columns)}")

    df = df.rename(columns={"observation_date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date

    df = df.rename(columns={"USREC": "value"})
    df["value"] = df["value"].astype("int8")

    df["series_id"] = "USREC"

    df = df[["date", "series_id", "value"]].sort_values("date").reset_index(drop=True)

    return df


def save_nber_usrec(df):
    """Sauvegarde du fichier USREC propre au format long (date, series_id, value)."""
    print(f"Sauvegarde NBER USREC dans {PROCESSED_DIR} ...")
    df.to_csv(NBER_USREC_CSV, index=False, encoding="utf-8")
    print(f" - {NBER_USREC_CSV}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":

    # --- FRED-MD ---
    long_df, transform_df = prepare_fred_md_tables()
    print(f"Lignes fred_md_raw : {len(long_df)}")
    print(f"Lignes fred_md_transform : {len(transform_df)}")

    save_outputs(long_df, transform_df)

    # --- USREC ---
    usrec_df = prepare_nber_usrec()
    print(f"Lignes nber_usrec : {len(usrec_df)}")

    save_nber_usrec(usrec_df)