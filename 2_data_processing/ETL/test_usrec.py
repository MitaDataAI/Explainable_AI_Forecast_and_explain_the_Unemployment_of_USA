import os
import requests
import pandas as pd


def fetch_fred_series(series_id: str, timeout: int = 20) -> pd.Series:
    """
    Télécharge une série FRED via l'API et retourne une pd.Series indexée par date.
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY manquante (variable d'environnement).")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()

    obs = r.json().get("observations", [])
    if not obs:
        raise RuntimeError(f"Aucune observation reçue pour {series_id}.")

    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # '.' -> NaN

    return df.set_index("date")["value"].rename(series_id).sort_index()


def test_usrec():
    usrec = fetch_fred_series("USREC")

    print("Dernières observations USREC :")
    print(usrec.tail(12))

    print("\nValeurs uniques (hors NaN) :")
    print(usrec.dropna().unique())

    print("\nType de la série :")
    print(usrec.dtype)


if __name__ == "__main__":
    test_usrec()