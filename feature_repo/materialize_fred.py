import os
import sys
from datetime import datetime, timedelta
from feast import FeatureStore

# Ajoute le dossier courant au PYTHONPATH
sys.path.append(os.path.dirname(__file__))

def main():
    store = FeatureStore(repo_path=".")

    end = datetime.utcnow()
    start = end - timedelta(days=30)

    store.materialize(start_date=start, end_date=end)

    print(f"✅ Materialize effectué : {start} → {end}")

if __name__ == "__main__":
    main()