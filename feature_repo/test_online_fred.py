import os
import sys
from feast import FeatureStore

# Ajoute le dossier courant au PYTHONPATH
sys.path.append(os.path.dirname(__file__))

def main():
    store = FeatureStore(repo_path=".")

    # On demande les features online pour une série FRED : UNRATE
    feature_vector = store.get_online_features(
        features=[
            "fred_features:value",
            "fred_features:lag_1",
            "fred_features:rolling_mean_12",
        ],
        entity_rows=[{"series_id": "UNRATE"}],  # clé d'entité correcte
    ).to_dict()

    # On simplifie la structure (Feast renvoie {col: [val]})
    feature_vector = {k: v[0] for k, v in feature_vector.items()}

    print("✅ Features online pour UNRATE :")
    print(feature_vector)

if __name__ == "__main__":
    main()