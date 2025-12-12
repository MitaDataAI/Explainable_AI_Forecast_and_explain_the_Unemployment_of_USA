from pathlib import Path

# ----------------------------------------------------
# Répertoire racine du projet (là où se trouve config.py)
# ----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------
# Dossiers du projet
# ----------------------------------------------------
RAW_DIR = PROJECT_ROOT / "1_data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "1_data" / "processed" / "cleaned"

# Création automatique des dossiers s'ils n'existent pas
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------
# Fichier d’entrée FRED-MD (nom adaptable)
# ----------------------------------------------------
FRED_MD_INPUT = RAW_DIR / "2025-11-MD.csv"  # Modifie selon la version téléchargée

# ----------------------------------------------------
# Sorties FRED-MD
# ----------------------------------------------------
FRED_MD_RAW_CSV = PROCESSED_DIR / "fred_md_raw.csv"
FRED_MD_TCODE_CSV = PROCESSED_DIR / "fred_md_transform.csv"

# ----------------------------------------------------
# NBER USREC
# - Fichier brut téléchargé depuis FRED (ex: observation_date,USREC)
# - Fichier nettoyé exporté par ETL/transform.py
# ----------------------------------------------------
# ⚠️ Ici on met le VRAI nom de ton fichier brut
NBER_USREC_INPUT = RAW_DIR / "2025-11-NBER-USREC.csv"   # Input brut
NBER_USREC_CSV   = PROCESSED_DIR / "nber_usrec.csv"     # Output propre (format long : date,series_id,value)