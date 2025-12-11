from feast import FileSource

fred_source = FileSource(
    name="fred_source",
    path="data/fred_features.parquet",
    timestamp_field="event_timestamp",
)