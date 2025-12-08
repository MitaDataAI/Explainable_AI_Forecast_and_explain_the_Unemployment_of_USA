from feast import Entity
from feast.types import ValueType

fred_series = Entity(
    name="fred_series",
    join_keys=["series_id"],
    value_type=ValueType.STRING,
    description="Identifiant de série FRED",
)