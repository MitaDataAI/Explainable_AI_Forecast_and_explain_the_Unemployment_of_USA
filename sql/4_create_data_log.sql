CREATE TABLE IF NOT EXISTS macro.data_log (
    id SERIAL PRIMARY KEY,
    run_timestamp TIMESTAMP DEFAULT NOW(),

    -- contexte
    series_id TEXT,
    stage TEXT,              -- data_status / refresh / validation
    action TEXT,             -- check / run / skip

    -- statut global
    status TEXT,

    -- local DB
    local_exists BOOLEAN,
    local_total_obs INT,
    local_first_date DATE,
    local_last_date DATE,
    local_n_nulls INT,
    local_null_dates TEXT,

    -- remote FRED
    remote_exists BOOLEAN,
    remote_total_obs INT,
    remote_first_date DATE,
    remote_last_date DATE,

    -- diff
    missing_periods INT,
    missing_dates TEXT,

    -- validation
    validation_status TEXT,

    -- refresh
    rows_added INT
);