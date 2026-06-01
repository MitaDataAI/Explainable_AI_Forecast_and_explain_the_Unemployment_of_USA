import sys
from pathlib import Path
import os
import requests

# Ajoute la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from configs.ETL import ETLConfig

CFG = ETLConfig()


def check_fred_api(api_key: str | None = None, timeout: int | None = None) -> bool:
    """
    Check simple de l'API FRED :
    - clé API présente
    - endpoint accessible
    - réponse valide
    """

    api_key = api_key or os.getenv("FRED_API_KEY")
    timeout = timeout or CFG.API_TIMEOUT

    # 1. Vérifier la clé
    if not api_key:
        print("[ERROR] API key manquante. Mets-la dans FRED_API_KEY.")
        return False

    # 2. Paramètres (série stable)
    params = {
        "series_id": "UNRATE",  # série fiable pour test
        "api_key": api_key,
        "file_type": "json",
    }

    try:
        # 3. Appel API
        r = requests.get(CFG.FRED_SERIES_URL, params=params, timeout=timeout)

        # 4. Vérifier HTTP
        if r.status_code != 200:
            print(f"[ERROR] HTTP {r.status_code}")
            print("Détail :", r.text[:300])
            return False

        # 5. Vérifier JSON
        try:
            data = r.json()
        except Exception:
            print("[ERROR] Réponse non JSON.")
            return False

        # 6. Vérifier structure FRED
        if "seriess" not in data or len(data["seriess"]) == 0:
            print("[ERROR] Réponse invalide FRED (champ 'seriess' absent ou vide).")
            return False

        # 7. Succès
        print("[OK] API FRED accessible et clé valide.")
        return True

    except requests.exceptions.Timeout:
        print("[ERROR] Timeout API FRED.")
        return False

    except requests.exceptions.ConnectionError:
        print("[ERROR] Impossible de se connecter à FRED.")
        return False

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Erreur API FRED : {e}")
        return False


if __name__ == "__main__":
    ok = check_fred_api()

    if ok:
        print(">>> FRED READY")
    else:
        print(">>> FRED FAILED")