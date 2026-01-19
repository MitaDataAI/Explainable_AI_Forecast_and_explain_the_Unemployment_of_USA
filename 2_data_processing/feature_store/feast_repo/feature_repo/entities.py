from feast import Entity, ValueType

series = Entity(
    name="series_id",
    join_keys=["series_id"],   # important
    value_type=ValueType.STRING,
    description="Unique identifier for each time series",
)