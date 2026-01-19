# 2_data_processing/ETL/3_extract.py

import sys
from pathlib import Path

# Ajoute la racine du projet au PYTHONPATH (robuste Windows)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import os
import time
import requests
import pandas as pd

from configs.ETL import ETLConfig

CFG = ETLConfig()


def fetch_fred_series(series_id: str, api_key: str, timeout: int | None = None) -> pd.Series:
    """
    Télécharge une série FRED (observations) et renvoie une pd.Series indexée par date.
    Extraction bornée à partir de CFG.OBS_START (bornage côté API).
    """
    timeout = timeout or CFG.API_TIMEOUT

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": CFG.OBS_START,
    }
    r = requests.get(CFG.FRED_OBS_URL, params=params, timeout=timeout)
    r.raise_for_status()

    obs = r.json().get("observations", [])
    if not obs:
        raise RuntimeError(f"Aucune observation reçue pour {series_id}")

    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # '.' -> NaN

    return df.set_index("date")["value"].sort_index()


def main():
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY manquante (variable d'environnement).")

    # Mapping (PROCESSED)
    mapping_path = CFG.MAPPING_PATH
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping introuvable: {mapping_path}")

    mapping = pd.read_csv(mapping_path)

    # Statuts utilisables (depuis config si dispo, sinon default)
    allowed_status = getattr(CFG, "ALLOWED_STATUS", frozenset({"DIRECT", "STRIP_X", "MANUAL"}))
    allowed_status = set(allowed_status)

    usable = mapping[mapping["status"].isin(allowed_status)].copy()
    usable = usable.dropna(subset=["fred_series_id"])

    print(f"Séries à télécharger: {len(usable)} ({', '.join(sorted(allowed_status))})")
    print(f"Observation start (bornage): {CFG.OBS_START}")

    series_list: list[pd.Series] = []
    failures: list[tuple[str, str, str]] = []

    # 1) Télécharger toutes les séries résolues (brut, multi-fréquences)
    for i, row in enumerate(usable.itertuples(index=False), start=1):
        out_name = str(row.fred_md_name).strip()   # nom colonne final (FRED-MD)
        sid = str(row.fred_series_id).strip()      # id FRED à télécharger

        try:
            s = fetch_fred_series(sid, api_key).rename(out_name)
            series_list.append(s)
        except Exception as e:
            failures.append((out_name, sid, str(e)))
            print(f"[WARN] Échec {out_name} (download {sid}) -> {e}")

        if CFG.SLEEP_EVERY and i % CFG.SLEEP_EVERY == 0:
            time.sleep(CFG.SLEEP_SECONDS)

    if not series_list:
        raise RuntimeError("Aucune série téléchargée. Vérifie le mapping et la clé API.")

    df = pd.concat(series_list, axis=1).sort_index()

    # 2) Ajouter USREC en BRUT (sans mensualiser ici)
    try:
        usrec = fetch_fred_series("USREC", api_key).rename("USREC")
        df = df.join(usrec, how="outer")
        print("✅ USREC ajouté (brut)")
    except Exception as e:
        print(f"[WARN] Impossible d'ajouter USREC : {e}")

    # 3) Sécurité : s'assurer que l'index ne descend pas sous OBS_START
    df = df[df.index >= pd.Timestamp(CFG.OBS_START)]

    # 4) Sauvegarde RAW uniquement (pilotée par config)
    out_dir = CFG.OUT_DIR_RAW
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv_raw = out_dir / CFG.OUT_CSV_RAW
    df.to_csv(out_csv_raw)

    print(f"\n✅ Dataset brut (multi-fréquences): {df.shape[0]} dates x {df.shape[1]} variables")
    print(f"✅ Début index: {df.index.min().date()} | Fin index: {df.index.max().date()}")
    print(f"✅ Sauvegardé: {out_csv_raw}")

    # 5) Sauvegarder les échecs
    if failures:
        fail_path = out_dir / CFG.OUT_FAILURES
        pd.DataFrame(failures, columns=["fred_md_name", "fred_series_id", "error"]).to_csv(
            fail_path, index=False
        )
        print(f"⚠️ Échecs: {len(failures)} (détails: {fail_path})")


if __name__ == "__main__":
    main()