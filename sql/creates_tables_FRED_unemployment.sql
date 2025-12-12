\c unemployment_usa

DROP TABLE IF EXISTS unemployment_dataset_usa;

CREATE TABLE unemployment_dataset_usa (
    date        DATE                NOT NULL,
    series_id   TEXT                NOT NULL,
    value       DOUBLE PRECISION    NOT NULL,
    CONSTRAINT unemployment_dataset_usa_pkey PRIMARY KEY (date, series_id)
);

-- 1) Insérer FRED-MD (en filtrant les valeurs manquantes)
INSERT INTO unemployment_dataset_usa (date, series_id, value)
SELECT date,
       series_id,
       value::DOUBLE PRECISION
FROM fred_md_raw
WHERE value IS NOT NULL
  AND value::TEXT <> 'NaN'
  AND value::TEXT <> ''
ORDER BY date, series_id;

-- 2) Insérer NBER USREC
INSERT INTO unemployment_dataset_usa (date, series_id, value)
SELECT date, series_id, value
FROM nber_usrec
ORDER BY date, series_id;

-- 3) Index pour avoir un ordre naturel (date, series_id)
CREATE INDEX IF NOT EXISTS idx_unemployment_dataset_usa_date_series
    ON unemployment_dataset_usa(date, series_id);