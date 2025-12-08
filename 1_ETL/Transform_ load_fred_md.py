# ETL/Transform_load_fred_md.py
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from getpass import getpass

# --------------------
# 1. Config chemins
# --------------------

# Dossier racine du projet = dossier parent de ETL
BASE_DIR = Path(__file__).resolve().parents[1]

# Chemin absolu vers le CSV
CSV_PATH = BASE_DIR / "data" / "raw" / "fred_md_current.csv"

DB_USER = "postgres"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "postgres"

# --------------------
# 2. Connexion PostgreSQL (avec demande de mot de passe)
# --------------------

def get_engine():
    print("🔐 Entrez votre mot de passe PostgreSQL :")
    db_password = getpass()   # mot de passe masqué

    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{db_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return engine

# --------------------
# 3. Transform : préparer les DataFrames à partir du CSV
# --------------------

def prepare_fred_md_tables():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {CSV_PATH}\n"
            f"Assure-toi d'avoir enregistré current.csv ici sous ce nom."
        )

    print(f"📖 Lecture du fichier {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH).dropna(how="all")

    # Dans FRED-MD, la colonne de dates s'appelle 'sasdate'
    if "sasdate" not in df.columns:
        raise ValueError("La colonne 'sasdate' est introuvable dans le CSV FRED-MD.")

    # 1) Ligne des codes de transformation (tcode) = première ligne de données
    tcode_row = df.iloc[0].copy()

    # 2) Données = toutes les lignes suivantes
    data = df.iloc[1:].copy()

    # 3) Conversion de la date
    data["sasdate"] = pd.to_datetime(data["sasdate"]).dt.date

    # 4) Passage du format wide -> long
    long_df = data.melt(
        id_vars=["sasdate"],
        var_name="series_id",
        value_name="value",
    ).rename(columns={"sasdate": "date"})

    # 5) Construction de la table des codes de transformation
    tcode_series = tcode_row.drop("sasdate")   # on enlève la colonne date
    transform_df = tcode_series.reset_index()
    transform_df.columns = ["series_id", "transform_code"]
    transform_df["transform_code"] = transform_df["transform_code"].astype(int)

    return long_df, transform_df

# --------------------
# 4. Load : insérer dans PostgreSQL
# --------------------

def load_into_postgres(engine, long_df: pd.DataFrame, transform_df: pd.DataFrame):
    with engine.begin() as conn:
        print("🧹 TRUNCATE des tables cibles...")
        conn.execute(text("TRUNCATE fred_md_raw;"))
        conn.execute(text("TRUNCATE fred_md_transform;"))

        print("⬆️ Insertion dans fred_md_raw ...")
        long_df.to_sql("fred_md_raw", conn, if_exists="append", index=False)

        print("⬆️ Insertion dans fred_md_transform ...")
        transform_df.to_sql("fred_md_transform", conn, if_exists="append", index=False)

    print("✅ Données chargées dans PostgreSQL.")

# --------------------
# 5. Main
# --------------------

def main():
    engine = get_engine()
    long_df, transform_df = prepare_fred_md_tables()

    print(f"🔎 Lignes fred_md_raw : {len(long_df)}")
    print(f"🔎 Lignes fred_md_transform : {len(transform_df)}")

    load_into_postgres(engine, long_df, transform_df)

if __name__ == "__main__":
    main()