# configs/settings.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectSettings:
    """
    Paramètres globaux du projet.
    -> Aucune logique métier ici.
    -> Uniquement des conventions de chemins et dossiers partagés.
    """

    # Racine des données
    DATA_DIR: Path = Path("1_data")

    # Données
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"

    # Notebooks
    NOTEBOOK_DIR: Path = Path("3_notebook")

    # Scripts & pipelines
    SCRIPTS_DIR: Path = Path("3_scripts")
    ETL_DIR: Path = Path("2_data_processing") / "ETL"

    # Tests
    TESTS_DIR: Path = Path("4_tests")

    # Artifacts (modèles, métriques, logs…)
    ARTIFACTS_DIR: Path = Path("artifacts")