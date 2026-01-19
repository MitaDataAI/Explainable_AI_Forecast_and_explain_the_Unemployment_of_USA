import os
import requests

SEARCH_URL = "https://api.stlouisfed.org/fred/series/search"
SERIES_URL = "https://api.stlouisfed.org/fred/series"
OBS_URL = "https://api.stlouisfed.org/fred/series/observations"


def _api_key() -> str:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY manquante.")
    return api_key


def fred_search(query: str, limit: int = 10):
    params = {
        "search_text": query,  # doit être NON vide
        "api_key": _api_key(),
        "file_type": "json",
        "limit": limit,
        "offset": 0,
        "order_by": "popularity",
        "sort_order": "desc",
    }
    r = requests.get(SEARCH_URL, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def fred_series_meta(series_id: str):
    params = {"series_id": series_id, "api_key": _api_key(), "file_type": "json"}
    r = requests.get(SERIES_URL, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def fred_tail_observations(series_id: str, n_last: int = 5):
    params = {
        "series_id": series_id,
        "api_key": _api_key(),
        "file_type": "json",
        "sort_order": "desc",
        "limit": n_last,
        "offset": 0,
    }
    r = requests.get(OBS_URL, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def print_search_results(query: str, limit: int = 10):
    data = fred_search(query, limit=limit)
    results = data.get("seriess", [])
    print(f"\nRésultats pour '{query}' (top {len(results)} par popularité):")
    for s in results:
        print(
            f"- id={s.get('id')} | title={s.get('title')} | "
            f"freq={s.get('frequency')} | units={s.get('units')}"
        )
    return results


def try_candidate(series_id: str, n_last: int = 5) -> bool:
    """Teste si la série existe + affiche quelques obs. Retourne True si OK."""
    try:
        meta = fred_series_meta(series_id).get("seriess", [])
        if not meta:
            return False

        m = meta[0]
        print(
            f"[OK] {series_id} existe: title={m.get('title')} | "
            f"freq={m.get('frequency')} | units={m.get('units')}"
        )

        tail = fred_tail_observations(series_id, n_last=n_last).get("observations", [])
        print("Dernières obs (date, value):")
        for o in tail:
            print(o["date"], o["value"])
        return True

    except requests.HTTPError as e:
        # Parfois 400/404 si id invalide
        print(f"[FAIL] {series_id} -> {e}")
        return False


if __name__ == "__main__":
    # ✅ Tes variables NOT_FOUND (depuis ton audit)
    NOT_FOUND = [
        "HWI",
        "HWIURATIO",
        "CLAIMSx",
        "AMDMNOx",
        "CONSPI",
        "S&P 500",
        "S&P div yield",
        "S&P PE ratio",
        "COMPAPFFx",
    ]

    # Candidats "évidents" à tester en plus (tu peux enrichir)
    MANUAL_CANDIDATES = {
        "S&P 500": ["SP500"],  # déjà validé
        # Les suivants seront suggérés par la recherche; ici on ne met rien de sûr
        "S&P div yield": [],
        "S&P PE ratio": [],
        "CLAIMSx": [],
        "HWI": [],
        "HWIURATIO": [],
        "CONSPI": [],
        "AMDMNOx": [],
        "COMPAPFFx": [],
    }

    for q in NOT_FOUND:
        print("\n" + "=" * 80)
        print(f"QUERY: {q}")

        # 1) Affiche les meilleurs résultats de recherche
        results = print_search_results(q, limit=10)

        # 2) Essayer un candidat manuel s'il existe (ex: SP500)
        for cand in MANUAL_CANDIDATES.get(q, []):
            print(f"\n---\nTest direct candidate series_id='{cand}'")
            if try_candidate(cand, n_last=5):
                break

        # 3) Sinon, essaie automatiquement les 3 premiers IDs de la recherche
        if results:
            top_ids = [r.get("id") for r in results[:3] if r.get("id")]
            for sid in top_ids:
                print(f"\n---\nAuto-test candidate from search: '{sid}'")
                if try_candidate(sid, n_last=3):
                    break
        else:
            print("[INFO] Aucun résultat de recherche pour cette requête.")