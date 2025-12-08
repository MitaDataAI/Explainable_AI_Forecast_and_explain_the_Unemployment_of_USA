from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

now = datetime.utcnow()

rows = []
# EXEMPLE : 3 séries économiques FRED
for series_id in ["UNRATE", "CPIAUCSL", "FEDFUNDS"]:
    for i in range(1, 6):
        rows.append(
            {
                "series_id": series_id,
                "event_timestamp": now - timedelta(days=i),
                "value": 10.0 + i,
                "lag_1": 9.0 + i,
                "rolling_mean_12": 8.5 + i,
            }
        )

df = pd.DataFrame(rows)
df.to_parquet("data/fred_features.parquet", index=False)

print("✅ Données FRED dummy créées :")
print(df.head())