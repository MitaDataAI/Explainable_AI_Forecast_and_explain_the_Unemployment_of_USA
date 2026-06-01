# 2_data_processing/ETL/5_transform.py
# Transform pour données déjà mensuelles -> sortie UNIQUE en format long

import sys
from pathlib import Path

import pandas as pd

# Ajout du dossier racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from configs.ETL import ETLConfig

CFG = ETLConfig()


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


def drop_transform_rows(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Supprime les lignes techniques du type 'Transform:' présentes dans certains
    fichiers FRED-MD/FRED historiques.
    """
    if date_col not in df.columns:
        return df

    mask_transform = (
        df[date_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("TRANSFORM:")
    )

    n_transform = int(mask_transform.sum())
    if n_transform > 0:
        print(f"⚠️ Lignes techniques 'Transform:' supprimées: {n_transform}")
        df = df.loc[~mask_transform].copy()

    return df


# Paths robustes
OUT_DIR_RAW = _to_project_path(
    getattr(CFG, "OUT_DIR_RAW", None),
    "1_data/raw",
)

CLEANED_DIR = _to_project_path(
    getattr(CFG, "CLEANED_DIR", None),
    "1_data/cleaned",
)

OUT_CSV_RAW = getattr(CFG, "OUT_CSV_RAW", "fred_raw_dataset.csv")
OUT_LONG_CSV = getattr(CFG, "OUT_DATASET_LONG", "dataset_fred_long.csv")


def main():
    # -------------------------------------------------
    # 1) Lecture du dataset mensuel brut
    # -------------------------------------------------
    raw_path = OUT_DIR_RAW / OUT_CSV_RAW

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"RAW_PATH: {raw_path}")
    print(f"RAW exists: {raw_path.exists()}")
    print(f"CLEANED_DIR: {CLEANED_DIR}")

    if not raw_path.exists():
        raise FileNotFoundError(f"Fichier brut introuvable: {raw_path}")

    print(f"Lecture: {raw_path}")
    df = pd.read_csv(raw_path)

    if df.empty:
        raise ValueError(f"Le fichier brut est vide: {raw_path}")

    # Robustesse : première colonne = date
    first_col = df.columns[0]
    if str(first_col).lower() != "date":
        df = df.rename(columns={first_col: "date"})

    if "date" not in df.columns:
        raise ValueError("Aucune colonne 'date' détectée dans le fichier brut.")

    # Supprimer explicitement la ligne technique "Transform:"
    df = drop_transform_rows(df, date_col="date")

    # Parsing robuste des dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    n_bad_dates = int(df["date"].isna().sum())
    if n_bad_dates > 0:
        print(f"⚠️ Dates invalides supprimées: {n_bad_dates}")

    df = (
        df.dropna(subset=["date"])
          .sort_values("date")
          .set_index("date")
    )

    if df.empty:
        raise ValueError("Aucune donnée valide après parsing de la colonne date.")

    # Conversion numérique sécurisée
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalisation des noms de séries
    df.columns = [normalize_series_id(c) for c in df.columns]

    # Colonnes dupliquées éventuelles après normalisation
    col_index = pd.Index(df.columns)
    if col_index.duplicated().any():
        duplicated_cols = col_index[col_index.duplicated()].unique().tolist()
        print(f"⚠️ Colonnes dupliquées après normalisation: {duplicated_cols}")
        df = df.T.groupby(level=0).first().T

    # Bornage temporel
    obs_start = pd.Timestamp(getattr(CFG, "OBS_START", "1900-01-01"))
    df = df[df.index >= obs_start].copy()

    if df.empty:
        raise ValueError(
            f"Aucune donnée restante après bornage à partir de OBS_START={obs_start.date()}."
        )

    # Normalisation de l'index
    df.index = pd.DatetimeIndex(df.index).normalize()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Vérification légère : on s'attend à des dates mensuelles
    non_month_start = int((df.index.day != 1).sum())
    if non_month_start > 0:
        print(f"⚠️ {non_month_start} dates ne tombent pas le 1er du mois.")

    # Petit contrôle qualité : first_nonnull par série
    first_nonnull = {}
    for col in df.columns:
        idx = df[col].first_valid_index()
        first_nonnull[col] = None if idx is None else pd.Timestamp(idx).date()

    n_1959 = sum(v == pd.Timestamp("1959-01-01").date() for v in first_nonnull.values() if v is not None)
    print(f"ℹ️ Séries dont le first_nonnull est 1959-01-01 : {n_1959}/{len(first_nonnull)}")

    # -------------------------------------------------
    # 2) Transformation DIRECTE en format long
    # -------------------------------------------------
    long_df = (
        df.reset_index()
          .melt(
              id_vars=["date"],
              var_name="series_id",
              value_name="value",
          )
          .sort_values(["series_id", "date"])
          .reset_index(drop=True)
    )

    if long_df.empty:
        raise ValueError("Le dataset long est vide après transformation.")

    # -------------------------------------------------
    # 3) Sauvegarde UNIQUE
    # -------------------------------------------------
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    out_path = CLEANED_DIR / OUT_LONG_CSV
    long_df.to_csv(out_path, index=False)

    # -------------------------------------------------
    # 4) Logs qualité
    # -------------------------------------------------
    n_total = len(long_df)
    n_missing = int(long_df["value"].isna().sum())
    miss_share = (n_missing / n_total) if n_total else 0.0

    min_date = long_df["date"].min()
    max_date = long_df["date"].max()

    print(f"✅ Dataset long sauvegardé : {out_path}")
    print(f"✅ Période : {min_date.date()} -> {max_date.date()}")
    print(f"✅ Dimensions : {long_df.shape[0]} lignes x {long_df.shape[1]} colonnes")
    print(f"ℹ️ Missing values : {n_missing}/{n_total} = {miss_share:.2%}")

    last_month = max_date
    last_month_missing = int(
        long_df.loc[long_df["date"] == last_month, "value"].isna().sum()
    )
    print(f"ℹ️ Dernier mois ({last_month.date()}) : {last_month_missing} valeurs manquantes")

    print("\nAperçu :")
    print(long_df.head())


if __name__ == "__main__":
    main()