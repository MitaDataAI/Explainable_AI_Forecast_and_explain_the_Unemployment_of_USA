-- On se connecte à la base
\c unemployment_usa

-- 1) Table des données brutes (format long)
CREATE TABLE IF NOT EXISTS fred_md_raw (
    date        DATE NOT NULL,
    series_id   TEXT NOT NULL,
    value       DOUBLE PRECISION
);

-- 2) Table des codes de transformation
CREATE TABLE IF NOT EXISTS fred_md_transform (
    series_id       TEXT NOT NULL,
    transform_code  INTEGER NOT NULL
);

-- 3) Index (fortement recommandés)
CREATE INDEX IF NOT EXISTS idx_fred_md_raw_date
    ON fred_md_raw(date);

CREATE INDEX IF NOT EXISTS idx_fred_md_raw_series
    ON fred_md_raw(series_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_fred_md_transform_series
    ON fred_md_transform(series_id);

-- Vérification des tables créées
\dt