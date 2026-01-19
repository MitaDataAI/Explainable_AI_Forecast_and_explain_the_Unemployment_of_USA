-- Schema
CREATE SCHEMA IF NOT EXISTS macro;

-- 1) Dictionnaire des séries
CREATE TABLE IF NOT EXISTS macro.series (
    series_id TEXT PRIMARY KEY,
    source TEXT DEFAULT 'FRED',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2) Observations mensuelles (format long)
CREATE TABLE IF NOT EXISTS macro.observations_monthly (
    date DATE NOT NULL,
    series_id TEXT NOT NULL,
    value DOUBLE PRECISION NULL,
    PRIMARY KEY (date, series_id),
    FOREIGN KEY (series_id) REFERENCES macro.series(series_id)
);

-- Index pour requêtes temporelles / par série
CREATE INDEX IF NOT EXISTS idx_obs_monthly_series ON macro.observations_monthly(series_id);
CREATE INDEX IF NOT EXISTS idx_obs_monthly_date ON macro.observations_monthly(date);