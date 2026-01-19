# configs/ETL.py

from dataclasses import dataclass
from pathlib import Path
from .settings import ProjectSettings

P = ProjectSettings()


@dataclass(frozen=True)
class ETLConfig:
    """
    Configuration du pipeline ETL FRED (FRED-only).

    Principe :
    - Aucun calcul
    - Aucune logique métier
    - Uniquement des règles, chemins et noms d’artefacts
    """

    # ======================
    # API FRED
    # ======================
    FRED_BASE_URL: str = "https://api.stlouisfed.org/fred"
    FRED_SERIES_URL: str = FRED_BASE_URL + "/series"
    FRED_OBS_URL: str = FRED_BASE_URL + "/series/observations"

    # Série utilisée pour le health-check
    HEALTHCHECK_SERIES_ID: str = "UNRATE"

    # Timeout API (secondes)
    API_TIMEOUT: int = 10

    # ======================
    # Extraction rules
    # ======================
    # Bornage temporel des séries (côté API)
    OBS_START: str = "1959-01-01"

    # ======================
    # Sources (INPUTS)
    # ======================
    # Liste des variables FRED-MD (header-only ou liste)
    FRED_MD_COLUMNS_PATH: Path = P.RAW_DIR / "fred_md_columns.csv"

    # ======================
    # Mapping rules
    # ======================
    # Statuts autorisés pour l'extraction
    ALLOWED_STATUS: frozenset[str] = frozenset({
        "DIRECT",
        "STRIP_X",
        "MANUAL",
    })

    # ======================
    # Outputs / Artefacts
    # ======================

    # --- Extraction (RAW) ---
    OUT_DIR_RAW: Path = P.RAW_DIR

    # Dataset brut déjà mensuel (fourni ou généré upstream)
    OUT_CSV_RAW: str = "dataset_fred.csv"

    # Log des échecs d’extraction (si utilisé)
    OUT_FAILURES: str = "fred_md_extraction_failures.csv"

    # --- Mapping ---
    MAPPING_PATH: Path = P.PROCESSED_DIR / "fred_md_to_fred_mapping.csv"

    # --- Transform / Cleaned ---
    CLEANED_DIR: Path = P.PROCESSED_DIR / "cleaned"

    # Dataset final canonique (long mensuel)
    OUT_DATASET_LONG: str = "dataset_fred_long.csv"

    # ======================
    # Execution behaviour
    # ======================
    # Throttling API
    SLEEP_EVERY: int = 20
    SLEEP_SECONDS: float = 0.2