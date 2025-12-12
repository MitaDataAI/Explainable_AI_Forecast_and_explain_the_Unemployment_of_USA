-- Connexion à la base
\c unemployment_usa

------------------------------------------------------------
-- 1) Suppression de la table si elle existe
------------------------------------------------------------
DROP TABLE IF EXISTS unemployment_dataset_usa;

------------------------------------------------------------
-- 2) Création de la table finale en format long
------------------------------------------------------------
CREATE TABLE unemployment_dataset_usa (
    date        DATE             NOT NULL,
    series_id   TEXT             NOT NULL,
    value       DOUBLE PRECISION,     -- On accepte les NULL ici
    CONSTRAINT unemployment_dataset_usa_pkey PRIMARY KEY (date, series_id)
);

------------------------------------------------------------
-- 3) Insertion des séries FRED-MD
--    (certaines contiennent des NULL réels dans les dernières dates,
--     ce qui est normal pour FRED-MD, donc on les accepte)
------------------------------------------------------------
INSERT INTO unemployment_dataset_usa (date, series_id, value)
SELECT 
    date,
    series_id,
    value::DOUBLE PRECISION
FROM fred_md_raw
ORDER BY date, series_id;

------------------------------------------------------------
-- 4) Insertion de la série de récession NBER USREC
------------------------------------------------------------
INSERT INTO unemployment_dataset_usa (date, series_id, value)
SELECT 
    date,
    series_id,
    value::DOUBLE PRECISION
FROM nber_usrec
ORDER BY date, series_id;

------------------------------------------------------------
-- 5) (Optionnel mais recommandé) Index pour accélérer l’accès
------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_unemployment_dataset_usa_date_series
    ON unemployment_dataset_usa(date, series_id);