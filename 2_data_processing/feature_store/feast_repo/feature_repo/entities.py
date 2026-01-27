# entities.py
from feast import Entity, ValueType

series = Entity(
    name="series_id",
    join_keys=["series_id"],
    value_type=ValueType.STRING,
    description="Unique identifier for each macroeconomic time series (LONG format)",
)