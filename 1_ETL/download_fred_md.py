# ETL/download_fred_md.py

import pandas as pd
from pathlib import Path
import requests
from io import StringIO

# URL directe vers FRED-MD (current.csv)
FRED_MD_URL = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv"

# Emplacement où sauvegarder le CSV
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "fred_md_current.csv"

# En-têtes pour tenter de simuler un navigateur
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.stlouisfed.org/research/economists/mccracken/fred-databases",
    "Accept": "text/csv,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def download_fred_md():
    print("📥 Tentative de téléchargement de FRED-MD (current.csv)...")

    try:
        # Essai de téléchargement avec en-têtes
        resp = requests.get(FRED_MD_URL, headers=HEADERS, timeout=10)
        print(f"Status code : {resp.status_code}")

        # Si blocage (403, 404, etc.)
        if resp.status_code != 200:
            print("❌ Téléchargement automatique impossible (403 ou erreur serveur).")
            print("➡️ Télécharge manuellement le fichier ici :")
            print("   https://www.stlouisfed.org/research/economists/mccracken/fred-databases")
            print(f"➡️ Puis enregistre-le sous : {CSV_PATH}")
            return

        # Lire le CSV depuis le texte renvoyé
        df = pd.read_csv(StringIO(resp.text))

        # Sauvegarder localement
        df.to_csv(CSV_PATH, index=False)
        print(f"✅ Fichier téléchargé et sauvegardé dans : {CSV_PATH}")

    except Exception as e:
        # N'importe quelle autre erreur réseau
        print("❌ Erreur lors du téléchargement automatique.")
        print("Reason:", e)
        print("\n➡️ Télécharge manuellement le fichier ici :")
        print("   https://www.stlouisfed.org/research/economists/mccracken/fred-databases")
        print(f"➡️ Enregistre le fichier sous : {CSV_PATH}")

if __name__ == "__main__":
    download_fred_md()