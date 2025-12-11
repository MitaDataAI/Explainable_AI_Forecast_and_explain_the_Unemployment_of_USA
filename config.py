from pathlib import Path

# Répertoire racine du projet (là où se trouve config.py)
PROJECT_ROOT = Path(__file__).resolve().parent

# -----------------------
# Dossiers
# -----------------------
RAW_DIR = PROJECT_ROOT / "1_data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "1_data" / "processed" / "cleaned"

# -----------------------
# Fichier d’entrée FRED-MD
# -----------------------
FRED_MD_INPUT = RAW_DIR / "2025-11-MD.csv"  # adapte le nom si besoin

# -----------------------
# Fichiers de sortie
# -----------------------
FRED_MD_RAW_CSV = PROCESSED_DIR / "fred_md_raw.csv"
FRED_MD_TCODE_CSV = PROCESSED_DIR / "fred_md_transform.csv"