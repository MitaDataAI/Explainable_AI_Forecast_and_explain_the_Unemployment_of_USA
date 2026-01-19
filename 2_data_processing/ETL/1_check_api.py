import sys
from pathlib import Path
import os
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from configs.ETL import ETLConfig

CFG = ETLConfig()


def check_fred_api(api_key: str | None = None, timeout: int | None = None) -> bool:
    """
    Pre-flight check:
    - API FRED accessible
    - API key valide
    """
    api_key = api_key or os.getenv("FRED_API_KEY")
    timeout = timeout or CFG.API_TIMEOUT

    if not api_key:
        print("[ERROR] API key manquante. Mets-la dans la variable d'env FRED_API_KEY.")
        return False

    params = {
        "series_id": CFG.HEALTHCHECK_SERIES_ID,
        "api_key": api_key,
        "file_type": "json",
    }

    try:
        r = requests.get(CFG.FRED_SERIES_URL, params=params, timeout=timeout)

        if r.status_code != 200:
            detail = ""
            try:
                j = r.json()
                detail = j.get("error_message") or j.get("message") or str(j)
            except Exception:
                detail = r.text[:300]

            print(f"[ERROR] HTTP {r.status_code} – {detail}")
            return False

        data = r.json()
        if "seriess" not in data:
            print("[ERROR] Réponse inattendue : champ 'seriess' absent.")
            return False

        print("[OK] API FRED accessible et clé valide.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Problème réseau/API : {e}")
        return False


if __name__ == "__main__":
    check_fred_api()