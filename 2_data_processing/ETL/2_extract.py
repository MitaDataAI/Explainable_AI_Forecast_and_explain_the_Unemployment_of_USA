# 2_data_processing/ETL/3_extract.py

import sys
from pathlib import Path
import os
import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from configs.ETL import ETLConfig

CFG = ETLConfig()


def _to_project_path(path_value, default_relative: str | None = None) -> Path:
    if path_value is None:
        return PROJECT_ROOT / default_relative
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


SPECIAL_SOURCE_PATH = _to_project_path(
    getattr(CFG, "SPECIAL_SOURCE_PATH", None),
    "1_data/raw/dataset_fred.csv",
)

OUT_DIR_RAW = _to_project_path(
    getattr(CFG, "OUT_DIR_RAW", None),
    "1_data/raw",
)

OUT_CSV_RAW = getattr(CFG, "OUT_CSV_RAW", "fred_raw_dataset.csv")


def canonical_name(col: str) -> str:
    c = str(col).strip()

    mapping = {
        "sasdate": "date",
        "S&P 500": "SP500",
        "OILPRICEx": "OILPRICEX",
    }
    return mapping.get(c, c)


def first_valid(s: pd.Series):
    s = s.dropna()
    return s.iloc[0] if not s.empty else None


def last_valid(s: pd.Series):
    s = s.dropna()
    return s.iloc[-1] if not s.empty else None


def to_monthly_start(s: pd.Series, rule="first_valid") -> pd.Series:
    if s.empty:
        return s

    s = s.dropna().sort_index()
    if s.empty:
        return s

    s.index = pd.to_datetime(s.index).normalize()
    grouped = s.groupby(s.index.to_period("M"))

    if rule == "last_valid":
        monthly = grouped.apply(last_valid)
    else:
        monthly = grouped.apply(first_valid)

    monthly.index = monthly.index.to_timestamp()
    monthly.index.name = "date"
    return monthly


def fetch_usrec_from_fred() -> pd.Series:
    """
    Récupère uniquement USREC depuis l'API FRED
    et retourne une Series mensuelle indexée par date.
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY manquante dans les variables d'environnement.")

    timeout = getattr(CFG, "API_TIMEOUT", 30)
    url = "https://api.stlouisfed.org/fred/series/observations"

    params = {
        "series_id": "USREC",
        "api_key": api_key,
        "file_type": "json",
    }

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()

    data = r.json()
    observations = data.get("observations", [])
    if not observations:
        raise ValueError("Aucune observation reçue pour USREC depuis FRED.")

    df = pd.DataFrame(observations)

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("Réponse FRED invalide pour USREC.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.set_index("date")

    usrec = to_monthly_start(df["value"], rule="first_valid").rename("USREC")
    return usrec


def debug_series_info(name: str, s: pd.Series):
    s_non_null = s.dropna()
    print(f"\n[DEBUG {name}]")
    print(f"Présente: {'oui' if name == s.name else 'série fournie'}")
    print(f"Nb total lignes: {len(s)}")
    print(f"Nb non-null: {len(s_non_null)}")

    if not s_non_null.empty:
        print(f"First non-null date: {s_non_null.index.min().date()}")
        print(f"Last non-null date : {s_non_null.index.max().date()}")
        print("Tail non-null:")
        print(s_non_null.tail())
    else:
        print("Série présente mais entièrement vide.")


def main():
    print(f"Dataset source: {SPECIAL_SOURCE_PATH}")

    if not SPECIAL_SOURCE_PATH.exists():
        raise FileNotFoundError(f"dataset_fred.csv introuvable: {SPECIAL_SOURCE_PATH}")

    raw = pd.read_csv(SPECIAL_SOURCE_PATH)

    # Enlève la ligne Transform:
    df = raw.iloc[1:].copy()

    # Renomme uniquement les colonnes concernées
    df = df.rename(columns={c: canonical_name(c) for c in df.columns})

    if "date" not in df.columns:
        raise ValueError("Colonne 'date' absente après renommage.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.set_index("date")

    # Conversion numérique
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Supprime USREC s'il existe déjà dans le CSV source pour éviter conflit/doublon
    if "USREC" in df.columns:
        print("USREC détecté dans le CSV source -> suppression avant rechargement API.")
        df = df.drop(columns=["USREC"])

    # Agrégation mensuelle du dataset source
    out = []
    for col in df.columns:
        rule = "last_valid" if col in {"SP500", "OILPRICEX"} else "first_valid"
        s = to_monthly_start(df[col], rule=rule).rename(col)
        out.append(s)

    df_monthly = pd.concat(out, axis=1).sort_index() if out else pd.DataFrame()

    # Fenêtre cible du dataset principal
    if df_monthly.empty:
        raise ValueError("df_monthly est vide, impossible d'aligner USREC sur la fenêtre temporelle.")

    target_start = df_monthly.index.min()
    target_end = df_monthly.index.max()

    print(f"\nFenêtre cible du dataset principal : {target_start.date()} -> {target_end.date()}")

    # Ajout de USREC depuis l'API FRED
    print("\nRécupération de USREC depuis l'API FRED...")
    usrec = fetch_usrec_from_fred()

    # Vérification 1 : USREC brut après fetch
    debug_series_info("USREC", usrec)

    if usrec.dropna().empty:
        raise ValueError("USREC récupéré depuis FRED mais vide après conversion/agrégation.")

    # Alignement de USREC sur la fenêtre du dataset principal
    usrec = usrec.loc[(usrec.index >= target_start) & (usrec.index <= target_end)]

    print("\n[DEBUG USREC APRES ALIGNEMENT]")
    debug_series_info("USREC", usrec)

    if usrec.dropna().empty:
        raise ValueError("USREC vide après alignement sur la fenêtre temporelle du dataset principal.")

    if df_monthly.empty:
        df_monthly = usrec.to_frame()
    else:
        df_monthly = df_monthly.join(usrec, how="left")

    # Vérification 2 : USREC juste après jointure
    print("\n[DEBUG APRES JOINTURE]")
    print("USREC dans df_monthly ?", "USREC" in df_monthly.columns)
    if "USREC" in df_monthly.columns:
        debug_series_info("USREC", df_monthly["USREC"])
    else:
        raise ValueError("USREC absent de df_monthly juste après la jointure.")

    # Nettoyage index
    df_monthly.index = pd.to_datetime(df_monthly.index).normalize()
    df_monthly = df_monthly[~df_monthly.index.duplicated(keep="first")]
    df_monthly = df_monthly.sort_index()

    # Vérification 3 : USREC juste avant sauvegarde
    print("\n[DEBUG AVANT SAVE]")
    print("USREC dans df_monthly ?", "USREC" in df_monthly.columns)

    if "USREC" not in df_monthly.columns:
        raise ValueError("USREC absent du dataset final avant sauvegarde.")

    if df_monthly["USREC"].dropna().empty:
        raise ValueError("USREC présent dans le dataset final mais entièrement vide avant sauvegarde.")

    debug_series_info("USREC", df_monthly["USREC"])

    OUT_DIR_RAW.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR_RAW / OUT_CSV_RAW
    df_monthly.to_csv(out_path, index_label="date")

    print("\n✅ DATASET FINAL")
    print(f"Shape: {df_monthly.shape}")
    print(f"Start: {df_monthly.index.min().date()}")
    print(f"End: {df_monthly.index.max().date()}")
    print(f"Saved: {out_path}")

    print("\n[CHECK]")
    for col in ["SP500", "OILPRICEX", "USREC"]:
        if col in df_monthly.columns:
            s = df_monthly[col].dropna()
            if not s.empty:
                print(
                    f"{col}: {len(s)} non-null | "
                    f"first={s.index.min().date()} | "
                    f"last={s.index.max().date()}"
                )
            else:
                print(f"{col}: présente mais vide")
        else:
            print(f"{col}: ABSENTE")


if __name__ == "__main__":
    main()