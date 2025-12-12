DROP TABLE IF EXISTS fred_usrec;

CREATE TABLE nber_usrec (
    date        DATE        NOT NULL,
    series_id   TEXT        NOT NULL,
    value       SMALLINT    NOT NULL,
    CONSTRAINT fred_usrec_pkey PRIMARY KEY (date, series_id)
);