from feast import FeatureView, Field
from feast.types import Float32
from datetime import timedelta

from entities import fred_series
from data_sources import fred_source

fred_features = FeatureView(
    name="fred_features",
    entities=[fred_series],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="value", dtype=Float32),
        Field(name="lag_1", dtype=Float32),
        Field(name="rolling_mean_12", dtype=Float32),
    ],
    online=True,
    source=fred_source,
)