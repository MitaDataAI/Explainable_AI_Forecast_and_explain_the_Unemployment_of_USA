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

    # 2) Lister les FeatureViews
    fvs = store.list_feature_views()
    print("\n=== Feature Views ===")
    for fv in fvs:
        print(f"- {fv.name} | features={[f.name for f in fv.features]}")

    # 3) Tester explicitement les 2 FeatureViews
    feature_refs = ["raw_value:value", "stationary_value:value"]

    print("\n=== Features to fetch ===")
    for fr in feature_refs:
        print("  ", fr)

    # 4) Entity DF minimal (join keys + timestamp)
    entity_df = pd.DataFrame(
        {
            "series_id": ["UNRATE", "UNRATE", "UNRATE"],
            "date": [
                datetime(2019, 12, 1),
                datetime(2020, 4, 1),
                datetime(2020, 5, 1),
            ],
        }
    )

    print("\n=== Entity DF ===")
    print(entity_df)

    # 5) Fetch historical features
    # ⚠️ IMPORTANT : full_feature_names=True pour éviter la collision sur "value"
    df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
        full_feature_names=True,   # ✅ FIX ICI
    ).to_df()

    print("\n=== Result ===")
    print(df)
    print("\nColumns:", list(df.columns))

    # 6) Sanity check brut vs stationnaire
    raw_col = [c for c in df.columns if c.startswith("raw_value__")]
    sta_col = [c for c in df.columns if c.startswith("stationary_value__")]

    if raw_col and sta_col:
        raw_col = raw_col[0]
        sta_col = sta_col[0]

        print("\n=== Sanity check ===")
        print("raw_col:", raw_col, "| stationary_col:", sta_col)
        print(df[["date", raw_col, sta_col]])

        print("\nraw describe:\n", df[raw_col].describe())
        print("\nstationary describe:\n", df[sta_col].describe())
        print(
            "\nabs(raw - stationary) describe:\n",
            (df[raw_col] - df[sta_col]).abs().describe(),
        )

    print("\n✅ SUCCESS: Feast returned raw & stationary features.")


if __name__ == "__main__":
    main()