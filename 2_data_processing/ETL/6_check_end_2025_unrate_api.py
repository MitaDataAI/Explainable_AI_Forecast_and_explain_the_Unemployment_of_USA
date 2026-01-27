import os
import requests
import pandas as pd

API_KEY = os.getenv("FRED_API_KEY")
assert API_KEY is not None, "FRED_API_KEY not set"

url = "https://api.stlouisfed.org/fred/series/observations"
params = {
    "series_id": "UNRATE",
    "api_key": API_KEY,
    "file_type": "json",
    "observation_start": "2025-10-01",
}

r = requests.get(url, params=params)
r.raise_for_status()

data = r.json()["observations"]

df = (
    pd.DataFrame(data)[["date", "value"]]
    .assign(
        date=lambda d: pd.to_datetime(d["date"]),
        value=lambda d: pd.to_numeric(d["value"], errors="coerce"),
    )
    .sort_values("date")
)

# 🔧 Modification ici : NaN -> "."
df["value"] = df["value"].apply(lambda x: "." if pd.isna(x) else x)

print(df)