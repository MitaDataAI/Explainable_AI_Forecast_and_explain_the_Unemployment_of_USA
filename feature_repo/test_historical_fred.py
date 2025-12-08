import os
import sys
import pandas as pd
from feast import FeatureStore

# Ajoute le dossier courant au PYTHONPATH
sys.path.append(os.path.dirname(__file__))

def main():
    # Charge les données FRED du fichier Parquet
    df = pd.read_parquet("data/fred_features.parquet")

    # L'entity_df doit contenir series_id + event_timestamp
    entity_df = df[["series_id", "event_timestamp"]]

    store = FeatureStore(repo_path=".")

    # Demande des features historiques
    historical_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "fred_features:value",
            "fred_features:lag_1",
            "fred_features:rolling_mean_12",
        ],
    ).to_df()

    print("✅ Historical FRED features récupérées :")
    print(historical_df.head())

if __name__ == "__main__":
    main()