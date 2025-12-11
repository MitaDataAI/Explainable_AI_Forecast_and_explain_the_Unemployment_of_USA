import os
import sys
sys.path.append(os.path.dirname(__file__))

from feast import FeatureStore
from fred_features import fred_features
from entities import fred_series

def main():
    store = FeatureStore(repo_path=".")
    store.apply([fred_series, fred_features])
    print("✅ Feast FRED repo appliqué.")

if __name__ == "__main__":
    main()