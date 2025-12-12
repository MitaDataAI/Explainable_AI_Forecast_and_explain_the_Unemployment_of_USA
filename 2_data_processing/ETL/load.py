import sys
from pathlib import Path
import getpass
import psycopg2

# Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import (
    FRED_MD_RAW_CSV,
    FRED_MD_TCODE_CSV,
    NBER_USREC_CSV,      # fichier nber_usrec.csv
)

# Paramètres de connexion
DB_NAME = "unemployment_usa"
DB_USER = "postgres"
DB_PASSWORD = getpass.getpass("Mot de passe PostgreSQL : ")
DB_HOST = "localhost"
DB_PORT = 5432


def copy_csv_to_table(cursor, csv_path: Path, table_name: str, columns: str):
    print(f"Chargement de {csv_path} dans la table {table_name} ...")
    with csv_path.open("r", encoding="utf-8") as f:
        cursor.copy_expert(
            f"COPY {table_name} {columns} FROM STDIN WITH CSV HEADER;",
            f,
        )
    print(f"Table {table_name} remplie.")


def main():

    # Vérification fichiers
    if not FRED_MD_RAW_CSV.exists():
        raise FileNotFoundError(f"Fichier introuvable : {FRED_MD_RAW_CSV}")
    if not FRED_MD_TCODE_CSV.exists():
        raise FileNotFoundError(f"Fichier introuvable : {FRED_MD_TCODE_CSV}")
    if not NBER_USREC_CSV.exists():
        raise FileNotFoundError(f"Fichier introuvable : {NBER_USREC_CSV}")

    print("Connexion à PostgreSQL...")
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )
    conn.autocommit = False

    try:
        cur = conn.cursor()

        print("Vidage des tables fred_md_raw, fred_md_transform et nber_usrec...")
        cur.execute("TRUNCATE TABLE fred_md_raw;")
        cur.execute("TRUNCATE TABLE fred_md_transform;")
        cur.execute("TRUNCATE TABLE nber_usrec;")   # <-- CORRIGÉ ICI

        # ---- LOAD FRED-MD RAW ----
        copy_csv_to_table(
            cur,
            FRED_MD_RAW_CSV,
            "fred_md_raw",
            "(date, series_id, value)",
        )

        # ---- LOAD FRED-MD TRANSFORM ----
        copy_csv_to_table(
            cur,
            FRED_MD_TCODE_CSV,
            "fred_md_transform",
            "(series_id, transform_code)",
        )

        # ---- LOAD NBER USREC (FORMAT LONG) ----
        copy_csv_to_table(
            cur,
            NBER_USREC_CSV,
            "nber_usrec",
            "(date, series_id, value)",
        )

        conn.commit()
        print("Données chargées avec succès dans PostgreSQL.")

    except Exception as e:
        conn.rollback()
        print("Erreur pendant le chargement — rollback effectué.")
        print(e)
        raise

    finally:
        conn.close()
        print("Connexion PostgreSQL fermée.")


if __name__ == "__main__":
    main()