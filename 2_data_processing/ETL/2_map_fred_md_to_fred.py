# 2_data_processing/ETL/2_map_fred_md_to_fred.py

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

# -------------------------------------------------------------------
# MAPPINGS MANUELS (prioritaires, traçables)
# -------------------------------------------------------------------
MANUAL_MAP = {
    "S&P 500": "SP500",
}

# -------------------------------------------------------------------
# UTILS
# -------------------------------------------------------------------
def series_exists(series_id: str, api_key: str, timeout: int | None = None) -> bool:
    """Retourne True si series_id existe dans FRED."""
    timeout = timeout or CFG.API_TIMEOUT
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    r = requests.get(CFG.FRED_SERIES_URL, params=params, timeout=timeout)
    if r.status_code != 200:
        return False
    j = r.json()
    return "seriess" in j and len(j["seriess"]) > 0


def resolve_fred_id(fred_md_name: str, api_key: str) -> tuple[str | None, str]:
    """
    Essaie de résoudre un nom FRED-MD vers un ID FRED.
    Retourne (resolved_id, status)

    status ∈ {MANUAL, DIRECT, STRIP_X, NOT_FOUND, SKIP}
    """
    name = str(fred_md_name).strip()

    # 0) Mapping manuel (prioritaire)
    if name in MANUAL_MAP:
        return MANUAL_MAP[name], "MANUAL"

    # Colonnes techniques
    if name.lower() in {"sasdate", "date"}:
        return None, "SKIP"

    # 1) Test direct
    if series_exists(name, api_key):
        return name, "DIRECT"

    # 2) Fallback: enlever suffixe 'x' (ex: RETAILx -> RETAIL)
    if name.endswith("x"):
        candidate = name[:-1]
        if candidate and series_exists(candidate, api_key):
            return candidate, "STRIP_X"

    return None, "NOT_FOUND"


def read_fred_md_names(path: Path) -> list[str]:
    """
    Lit les noms FRED-MD depuis un CSV.

    Formats supportés:
    A) CSV header-only wide: 0 lignes, les séries sont les colonnes (ton cas)
    B) CSV liste: une colonne 'fred_md_name' (ou similaire) contenant les séries
    C) CSV liste: une seule colonne quelconque contenant les séries
    """
    # Lire juste le header (rapide, et marche si le fichier est "header-only")
    df0 = pd.read_csv(path, nrows=0)

    # A) header-only wide (beaucoup de colonnes)
    if len(df0.columns) > 1:
        return [c.strip() for c in df0.columns if str(c).strip()]

    # Sinon, on lit le contenu (format liste)
    df = pd.read_csv(path)

    # B) colonne explicite
    for col in ("fred_md_name", "name", "series", "series_name", "variable"):
        if col in df.columns:
            return df[col].dropna().astype(str).str.strip().tolist()

    # C) une seule colonne
    if len(df.columns) == 1:
        col = df.columns[0]
        return df[col].dropna().astype(str).str.strip().tolist()

    raise ValueError(
        f"Format inattendu pour {path}. "
        "Utilise soit: (A) header-only (colonnes=series), soit (B) une colonne 'fred_md_name', soit (C) une seule colonne."
    )


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY manquante (variable d'environnement).")

    # ✅ Source des noms (dans la config)
    csv_path = CFG.FRED_MD_COLUMNS_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {csv_path}")

    series_names = read_fred_md_names(csv_path)

    rows = []
    for i, name in enumerate(series_names, start=1):
        resolved, status = resolve_fred_id(name, api_key)
        rows.append(
            {
                "fred_md_name": name,
                "fred_series_id": resolved,
                "status": status,
            }
        )

        # Rate limit gentil pour l'API (piloté par config)
        if CFG.SLEEP_EVERY and i % CFG.SLEEP_EVERY == 0:
            time.sleep(CFG.SLEEP_SECONDS)

    out = pd.DataFrame(rows)

    # ----------------------------------------------------------------
    # RÉSUMÉS
    # ----------------------------------------------------------------
    print("Résumé disponibilité FRED (depuis ta source de noms):")
    print(out["status"].value_counts(dropna=False))

    print("\nExemples STRIP_X (FRED-MD -> FRED):")
    ex = out[out["status"] == "STRIP_X"].head(10)
    if ex.empty:
        print("  (aucun)")
    else:
        for _, r in ex.iterrows():
            print(f"  {r['fred_md_name']} -> {r['fred_series_id']}")

    print("\nExemples MANUAL:")
    exm = out[out["status"] == "MANUAL"]
    if exm.empty:
        print("  (aucun)")
    else:
        for _, r in exm.iterrows():
            print(f"  {r['fred_md_name']} -> {r['fred_series_id']}")

    print("\nExemples NOT_FOUND (à investiguer):")
    ex2 = out[out["status"] == "NOT_FOUND"].head(10)
    if ex2.empty:
        print("  (aucun)")
    else:
        for _, r in ex2.iterrows():
            print(f"  {r['fred_md_name']}")

    # ----------------------------------------------------------------
    # SAUVEGARDE (pilotée par config)
    # ----------------------------------------------------------------
    out_path = CFG.MAPPING_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\nMapping sauvegardé dans: {out_path}")


if __name__ == "__main__":
    main()