# 2_data_processing/ETL/5_transform.py
# Transform pour données déjà mensuelles → sortie UNIQUE en format long

import sys
from pathlib import Path

# Ajout du dossier racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from configs.ETL import ETLConfig

CFG = ETLConfig()


def normalize_series_id(s: str) -> str:
    """
    Normalisation minimale des identifiants de séries :
    - suppression de '&'
    - suppression des espaces
    - mise en majuscules
    """
    if pd.isna(s):
        return s
    return str(s).replace("&", "").replace(" ", "").upper()


def main():
    # -------------------------------------------------
    # 1) Lecture du dataset mensuel brut
    # -------------------------------------------------
    raw_path = CFG.OUT_DIR_RAW / CFG.OUT_CSV_RAW
    if not raw_path.exists():
        raise FileNotFoundError(f"Fichier brut introuvable: {raw_path}")

    print(f"Lecture: {raw_path}")
    df = pd.read_csv(raw_path)

    # Robustesse : première colonne = date
    first_col = df.columns[0]
    if str(first_col).lower() != "date":
        df = df.rename(columns={first_col: "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = (
        df.dropna(subset=["date"])
          .sort_values("date")
          .set_index("date")
    )

    # Conversion numérique sécurisée
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalisation des noms de séries
    df.columns = [normalize_series_id(c) for c in df.columns]

    # Bornage temporel
    df = df[df.index >= pd.Timestamp(CFG.OBS_START)]

    # -------------------------------------------------
    # 2) Transformation DIRECTE en format long
    # -------------------------------------------------
    long_df = (
        df.reset_index()
          .melt(
              id_vars=["date"],
              var_name="series_id",
              value_name="value"
          )
          .sort_values(["series_id", "date"])
          .reset_index(drop=True)
    )

    # -------------------------------------------------
    # 3) Sauvegarde UNIQUE
    # -------------------------------------------------
    cleaned_dir = CFG.CLEANED_DIR
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    out_path = cleaned_dir / "dataset_fred_long.csv"
    long_df.to_csv(out_path, index=False)

    # -------------------------------------------------
    # 4) Logs qualité
    # -------------------------------------------------
    n_total = len(long_df)
    n_missing = int(long_df["value"].isna().sum())
    miss_share = (n_missing / n_total) if n_total else 0.0

    print(f"✅ Dataset long sauvegardé : {out_path}")
    print(f"✅ Période : {long_df['date'].min().date()} → {long_df['date'].max().date()}")
    print(f"ℹ️ Missing values : {n_missing}/{n_total} = {miss_share:.2%}")

    last_month = long_df["date"].max()
    last_month_missing = long_df[long_df["date"] == last_month]["value"].isna().sum()
    print(f"ℹ️ Dernier mois ({last_month.date()}) : {last_month_missing} valeurs manquantes")


if __name__ == "__main__":
    main()