# 2_data_processing/ETL/5_load.py

import sys
from pathlib import Path
import getpass

import psycopg2

# Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from configs.ETL import ETLConfig

CFG = ETLConfig()

# ----------------------------
# Paramètres de connexion
# ----------------------------
DB_NAME = "unemployment_usa"
DB_USER = "postgres"
DB_HOST = "localhost"
DB_PORT = 5432


def _to_project_path(path_value, default_relative: str | None = None) -> Path:
    """
    Convertit un path config en Path absolu basé sur PROJECT_ROOT si nécessaire.
    """
    if path_value is None:
        if default_relative is None:
            raise ValueError("Path config manquant et aucun défaut fourni.")
        return PROJECT_ROOT / default_relative

    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


# Paths robustes
CLEANED_DIR = _to_project_path(
    getattr(CFG, "CLEANED_DIR", None),
    "1_data/cleaned",
)

OUT_DATASET_LONG = getattr(CFG, "OUT_DATASET_LONG", None)
if OUT_DATASET_LONG is None:
    OUT_DATASET_LONG = getattr(CFG, "OUT_LONG_CSV", "dataset_fred_long.csv")

LONG_CSV_PATH = CLEANED_DIR / OUT_DATASET_LONG


def copy_csv_to_table(cur, csv_path: Path, table_name: str, columns: str):
    """
    Charge un CSV via COPY FROM STDIN.
    columns exemple: "(date, series_id, value)"
    """
    print(f"Chargement de {csv_path} -> {table_name} ...")

    with csv_path.open("r", encoding="utf-8") as f:
        cur.copy_expert(
            f"COPY {table_name} {columns} FROM STDIN WITH CSV HEADER;",
            f,
        )

    print(f"✅ Table {table_name} remplie.")


def main():
    long_csv = LONG_CSV_PATH

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"CLEANED_DIR: {CLEANED_DIR}")
    print(f"LONG_CSV_PATH: {long_csv}")
    print(f"LONG_CSV exists: {long_csv.exists()}")

    if not long_csv.exists():
        raise FileNotFoundError(
            f"CSV introuvable: {long_csv}\n"
            "Astuce: lance d'abord `python 2_data_processing/ETL/5_transform.py`."
        )

    db_password = getpass.getpass("Mot de passe PostgreSQL : ")

    print("Connexion à PostgreSQL...")
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=db_password,
        host=DB_HOST,
        port=DB_PORT,
    )
    conn.autocommit = False

    try:
        cur = conn.cursor()

        # Vérif rapide du schéma
        cur.execute("CREATE SCHEMA IF NOT EXISTS macro;")

        # 1) Retirer la FK si elle existe
        print("Désactivation temporaire de la contrainte FK (si existante)...")
        cur.execute(
            """
            DO $$
            BEGIN
              IF EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'observations_monthly_series_id_fkey'
                  AND conrelid = 'macro.observations_monthly'::regclass
              ) THEN
                ALTER TABLE macro.observations_monthly
                  DROP CONSTRAINT observations_monthly_series_id_fkey;
              END IF;
            END $$;
            """
        )

        # 2) Purge rejouable
        print("Vidage des tables macro.observations_monthly et macro.series...")
        cur.execute("TRUNCATE TABLE macro.observations_monthly;")
        cur.execute("TRUNCATE TABLE macro.series;")

        # 3) Import des observations
        copy_csv_to_table(
            cur,
            long_csv,
            "macro.observations_monthly",
            "(date, series_id, value)",
        )

        # 4) Recréer macro.series à partir des observations
        print("Construction de macro.series (DISTINCT series_id)...")
        cur.execute(
            """
            INSERT INTO macro.series(series_id)
            SELECT DISTINCT series_id
            FROM macro.observations_monthly
            WHERE series_id IS NOT NULL
            ON CONFLICT (series_id) DO NOTHING;
            """
        )

        # 5) Remettre la FK
        print("Réactivation de la contrainte FK (si absente)...")
        cur.execute(
            """
            DO $$
            BEGIN
              IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'observations_monthly_series_id_fkey'
                  AND conrelid = 'macro.observations_monthly'::regclass
              ) THEN
                ALTER TABLE macro.observations_monthly
                  ADD CONSTRAINT observations_monthly_series_id_fkey
                  FOREIGN KEY (series_id) REFERENCES macro.series(series_id);
              END IF;
            END $$;
            """
        )

        conn.commit()
        print("✅ Données chargées avec succès dans PostgreSQL (FRED-only).")

        # 6) Vérifications rapides
        cur.execute("SELECT COUNT(*) FROM macro.observations_monthly;")
        n_obs = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM macro.series;")
        n_series = cur.fetchone()[0]

        cur.execute(
            """
            SELECT MIN(date), MAX(date)
            FROM macro.observations_monthly;
            """
        )
        min_date, max_date = cur.fetchone()

        print(f"✅ observations_monthly: {n_obs} lignes")
        print(f"✅ series: {n_series} séries")
        print(f"✅ période chargée: {min_date} -> {max_date}")

    except Exception as e:
        conn.rollback()
        print("❌ Erreur pendant le chargement — rollback effectué.")
        print(e)
        raise

    finally:
        conn.close()
        print("Connexion PostgreSQL fermée.")


if __name__ == "__main__":
    main()