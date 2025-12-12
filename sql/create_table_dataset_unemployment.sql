CREATE TABLE unemployment_dataset_USA (
    date        DATE NOT NULL,
    series_id   TEXT NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    CONSTRAINT unemployment_dataset_usa_pkey PRIMARY KEY (date, series_id)
);
