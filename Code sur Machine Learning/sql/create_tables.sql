-- Table des valeurs FRED-MD en format long
CREATE TABLE IF NOT EXISTS fred_md_raw (
    date       DATE,
    series_id  TEXT,
    value      DOUBLE PRECISION
);

-- Table des codes de transformation FRED-MD
CREATE TABLE IF NOT EXISTS fred_md_transform (
    series_id      TEXT,
    transform_code INTEGER
);