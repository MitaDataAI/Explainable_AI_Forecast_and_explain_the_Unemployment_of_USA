# test_feast.py
from __future__ import annotations

from datetime import datetime
import pandas as pd
from feast import FeatureStore


def main() -> None:
    # 1) Charger le FeatureStore depuis le repo courant
    store = FeatureStore(repo_path=".")

    print("\n=== Feast repo loaded ===")
    print("Project:", store.project)

    # 2) Lister les FeatureViews et leurs features (pour ne pas deviner les noms)
    fvs = store.list_feature_views()
    if not fvs:
        raise RuntimeError("Aucune FeatureView trouvée. Vérifie que `feast apply` a bien été fait.")

    print("\n=== Feature Views ===")
    for fv in fvs:
        print(f"- {fv.name} | entities={fv.entities} | features={[f.name for f in fv.features]}")

    # On prend la première FV, ou la tienne explicitement
    fv_name = "stationary_value"
    fv = next((x for x in fvs if x.name == fv_name), None)
    if fv is None:
        raise RuntimeError(f"FeatureView '{fv_name}' introuvable. Found: {[x.name for x in fvs]}")

    # 3) Construire la liste des features "fv_name:feature_name"
    # Exemple: "stationary_value:value"
    feature_refs = [f"{fv.name}:{feat.name}" for feat in fv.features]
    if not feature_refs:
        raise RuntimeError(f"La FeatureView '{fv.name}' n'a aucune feature déclarée.")

    print("\n=== Features to fetch ===")
    for fr in feature_refs:
        print("  ", fr)

    # 4) Construire un entity_df minimal
    # IMPORTANT: doit contenir join_keys + event_timestamp
    # Ici: series_id + event_timestamp
    entity_df = pd.DataFrame(
        {
            "series_id": ["UNRATE", "CPIAUCSL"],  # adapte si besoin
            "event_timestamp": [datetime(2020, 1, 1), datetime(2020, 1, 1)],
        }
    )

    print("\n=== Entity DF ===")
    print(entity_df)

    # 5) Appel historical features
    print("\n=== Fetching historical features... ===")
    df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
    ).to_df()

    print("\n=== Result ===")
    print(df.head(20))
    print("\nColumns:", list(df.columns))
    print("\n✅ SUCCESS: Feast returned historical features.")


if __name__ == "__main__":
    main()