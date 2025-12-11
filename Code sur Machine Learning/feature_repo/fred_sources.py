from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource

fred_md_source = PostgreSQLSource(
    name="fred_md_source",
    query="""
        SELECT
            date AS event_timestamp,
            series_id,
            value
        FROM fred_md_raw
    """,
    timestamp_field="event_timestamp",
)