from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Int64, Float32

from entities import customer
from data_sources import customer_features_source

customer_features_30d = FeatureView(
    name="customer_features_30d",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="recency_days", dtype=Int64),
        Field(name="nb_transactions_30d", dtype=Int64),
        Field(name="avg_basket_30d", dtype=Float32),
        Field(name="churn_score_raw", dtype=Float32),
    ],
    online=True,
    source=customer_features_source,
)