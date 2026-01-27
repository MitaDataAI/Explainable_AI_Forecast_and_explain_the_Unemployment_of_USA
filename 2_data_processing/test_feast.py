import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path=".")

entity_df = pd.DataFrame(
    {
        "series_id": ["UNRATE", "CPIAUCSL"],
        "date": pd.to_datetime(["2019-01-01", "2019-01-01"]),
    }
)

df = store.get_historical_features(
    entity_df=entity_df,
    features=["stationary_value:value"],
).to_df()

print(df.head(10))