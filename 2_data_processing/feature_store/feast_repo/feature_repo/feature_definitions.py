from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Float32

from entities import series
from data_sources import stationary_long_source

stationary_value = FeatureView(
    name="stationary_value",
    entities=[series],
    ttl=timedelta(days=365 * 100),
    schema=[
        Field(name="value", dtype=Float32),
    ],
    source=stationary_long_source,
    online=False,
)