from pathlib import Path
from feast import FileSource

# --------------------------------------------------
# Trouver la racine projet en remontant jusqu'à trouver "configs"
# --------------------------------------------------
REPO_DIR = Path(__file__).resolve().parent  # .../feature_repo

PROJECT_ROOT = REPO_DIR
while not (PROJECT_ROOT / "configs").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

if not (PROJECT_ROOT / "configs").exists():
    raise RuntimeError(
        f"Impossible de trouver la racine projet (dossier 'configs' introuvable) depuis {REPO_DIR}"
    )

# --------------------------------------------------
# Parquet stationnarisé (format LONG)
# --------------------------------------------------
STATIONARY_LONG_PARQUET = (
    PROJECT_ROOT
    / "1_data"
    / "processed"
    / "features"
    / "unemployment_features_stationary_long.parquet"
)

if not STATIONARY_LONG_PARQUET.exists():
    raise FileNotFoundError(
        f"Parquet introuvable : {STATIONARY_LONG_PARQUET}\n"
        "➡️ Exécute 2_build_features_stationary.py pour le générer."
    )

# --------------------------------------------------
# Feast FileSource
# --------------------------------------------------
stationary_long_source = FileSource(
    path=str(STATIONARY_LONG_PARQUET),
    timestamp_field="date",
)

# --------------------------------------------------
# Parquet BRUT (format LONG)
# --------------------------------------------------
RAW_LONG_PARQUET = (
    PROJECT_ROOT
    / "1_data"
    / "processed"
    / "features"
    / "unemployment_features_raw_long.parquet"
)

if not RAW_LONG_PARQUET.exists():
    raise FileNotFoundError(
        f"Parquet introuvable : {RAW_LONG_PARQUET}\n"
        "➡️ Génère le parquet brut long (voir script ETL/features)."
    )

raw_long_source = FileSource(
    path=str(RAW_LONG_PARQUET),
    timestamp_field="date",
)