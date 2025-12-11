from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Float64, String

from .entities import fred_series
from .fred_sources import fred_md_source

macro_features = FeatureView(
    name="macro_features",
    entities=[fred_series],
    ttl=timedelta(days=0),
    schema=[
        Field(name="series_id", dtype=String),
        Field(name="value", dtype=Float64),
    ],
    source=fred_md_source,
    online=False,  # PRECISEMENT : offline only
)