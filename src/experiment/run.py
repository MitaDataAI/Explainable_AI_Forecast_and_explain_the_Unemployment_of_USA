from mlforecast import MLForecast
import pandas as pd

from .pipeline.load_features import load_features_from_feast
from .pipeline.run_backtesting import run_backtesting
from .registry.model_registry import MLFORECAST_MODELS


# --------------------------------------------------
# 0. FULL VERBOSE MODE (ne rien cacher)
# --------------------------------------------------
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


# --------------------------------------------------
# 1. Entity DF (doit matcher Feast)
#   - entity key  : series_id
#   - time column : date
# --------------------------------------------------
dates = pd.date_range(
    start="1959-01-01",
    end="2025-09-01",
    freq="MS"   # mensuel
)

entity_df = pd.DataFrame({
    "series_id": ["UNRATE"] * len(dates),
    "date": dates,
})

FEATURE_REFS = ["stationary_value:value"]


# --------------------------------------------------
# 2. Load features from Feast
# --------------------------------------------------
ts = load_features_from_feast(
    entity_df=entity_df,
    feature_refs=FEATURE_REFS
)

print("\n=== RAW FEATURES FROM FEAST ===")
print(ts.to_string(index=False))


# --------------------------------------------------
# 3. Adapter au format MLForecast
#   MLForecast attend: unique_id, ds, y
# --------------------------------------------------
ts = ts.rename(columns={
    "series_id": "unique_id",
    "date": "ds",
    "value": "y"
})

print("\n=== DATA AFTER RENAMING (MLForecast FORMAT) ===")
print(ts.to_string(index=False))


# --------------------------------------------------
# 4. Build MLForecast
# --------------------------------------------------
mlf = MLForecast(
    models={name: model() for name, model in MLFORECAST_MODELS.items()},
    freq="MS",
    lags=[1, 12],
)

print("\n=== MLFORECAST MODELS ===")
for name in MLFORECAST_MODELS:
    print(f"- {name}")


# --------------------------------------------------
# 5. Run backtesting
# --------------------------------------------------
bkt_df = run_backtesting(mlf, ts)

print("\n=== BACKTESTING RESULTS (FULL) ===")
print(bkt_df.to_string(index=False))