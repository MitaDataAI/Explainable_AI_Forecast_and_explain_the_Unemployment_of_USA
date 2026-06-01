import sys
from pathlib import Path
import os
import getpass
import time
from io import StringIO
import argparse

import psycopg2
import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from configs.ETL import ETLConfig

CFG = ETLConfig()

DB_NAME = "unemployment_usa"
DB_USER = "postgres"
DB_HOST = "localhost"
DB_PORT = 5432

# Valeurs par défaut, surchargeables en CLI
DEFAULT_REFRESH_END_DATE = "2026-02-01"
DEFAULT_REFRESH_LOOKBACK_MONTHS = 24

# Séries à imputer par interpolation linéaire simple après refresh
POST_REFRESH_INTERPOLATION_SERIES = {
    "CPIAUCSL",
    "UNRATE",
}

# Séries traitées dans un bloc spécial, mais refreshées via API FRED
SPECIAL_SOURCE_SERIES_MAP = {
    "SP500": "SP500",
    "OILPRICEX": "WTISPLC",
}

DATASET_FRED_CANDIDATE_PATHS = [
    PROJECT_ROOT / "1_data" / "raw" / "dataset_fred.csv",
    PROJECT_ROOT / "data" / "raw" / "dataset_fred.csv",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Refresh FRED monthly observations without deleting existing rows.")
    parser.add_argument(
        "--refresh-end-date",
        default=os.getenv("REFRESH_END_DATE", DEFAULT_REFRESH_END_DATE),
        help="Date maximale incluse pour le refresh, ex: 2026-01-01",
    )
    parser.add_argument(
        "--lookback-months",
        type=int,
        default=int(os.getenv("REFRESH_LOOKBACK_MONTHS", DEFAULT_REFRESH_LOOKBACK_MONTHS)),
        help="Nombre de mois à regarder en arrière pour chercher des dates manquantes",
    )
    parser.add_argument(
        "--impute-after-refresh",
        action="store_true",
        help="Impute les trous mensuels isolés après le refresh",
    )
    return parser.parse_args()


def format_date(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(pd.Timestamp(value).normalize().date())


def month_start(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.to_period("M").to_timestamp()


def load_fred_mapping() -> dict[str, str]:
    """
    Charge le mapping local -> FRED.

    Compatible avec un fichier :
    raw_name, normalized_name, canonical_name, status

    On utilise canonical_name comme :
    - nom local
    - identifiant distant FRED
    """
    candidate_paths = []

    cfg_mapping_path = getattr(CFG, "MAPPING_PATH", None)
    if cfg_mapping_path is not None:
        candidate_paths.append(Path(cfg_mapping_path))

    candidate_paths.extend([
        PROJECT_ROOT / "1_data" / "processed" / "fred_md_to_fred_mapping.csv",
        PROJECT_ROOT / "data" / "processed" / "fred_md_to_fred_mapping.csv",
        PROJECT_ROOT / "artifacts" / "processed" / "fred_md_to_fred_mapping.csv",
        PROJECT_ROOT / "artifacts" / "data_intake" / "fred_md_to_fred_mapping.csv",
    ])

    mapping_path = None
    for path in candidate_paths:
        if path.exists():
            mapping_path = path
            break

    if mapping_path is None:
        checked = "\n".join(str(p) for p in candidate_paths)
        raise FileNotFoundError(
            "Mapping file not found. Checked these paths:\n"
            f"{checked}"
        )

    print(f"Using mapping file: {mapping_path}")

    df = pd.read_csv(mapping_path)

    if df.empty:
        raise ValueError("Mapping file is empty.")

    df.columns = [str(c).strip() for c in df.columns]
    print("Columns found in mapping:", df.columns.tolist())

    expected_cols = {"raw_name", "normalized_name", "canonical_name", "status"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(
            "Structure de mapping non reconnue.\n"
            f"Colonnes disponibles: {df.columns.tolist()}\n"
            f"Colonnes attendues: {sorted(expected_cols)}"
        )

    df["canonical_name"] = df["canonical_name"].astype(str).str.strip()
    df["status"] = df["status"].astype(str).str.strip()

    df["canonical_name"] = df["canonical_name"].replace({
        "": pd.NA,
        "nan": pd.NA,
        "None": pd.NA,
    })

    before = len(df)

    df["status_norm"] = df["status"].str.upper()
    allowed_status = set(
        s.upper() for s in getattr(CFG, "ALLOWED_STATUS", {"DIRECT", "STRIP_X", "MANUAL"})
    )

    matched = int(df["status_norm"].isin(allowed_status).sum())

    print("Unique status values found:", sorted(df["status"].dropna().unique().tolist()))
    print("Allowed status values:", sorted(allowed_status))
    print(f"Status matches found: {matched}")

    if matched > 0:
        df = df[df["status_norm"].isin(allowed_status)].copy()
        print(f"Filtered by status: {before} -> {len(df)} rows")
    else:
        print("[WARNING] No status matched allowed values. Skipping status filter.")

    df = df.dropna(subset=["canonical_name"]).copy()

    if df.empty:
        raise ValueError("Mapping file empty after cleaning canonical_name.")

    duplicated_local = df[df.duplicated(subset=["canonical_name"], keep=False)]
    if not duplicated_local.empty:
        print("[WARNING] Duplicate canonical_name found. Keeping first occurrence.")
        print(duplicated_local[["canonical_name", "status"]].head(10).to_string(index=False))
        df = df.drop_duplicates(subset=["canonical_name"], keep="first").copy()

    mapping = dict(zip(df["canonical_name"], df["canonical_name"]))

    print(f"Loaded mapping entries: {len(mapping)}")

    for key in ["SP500", "OILPRICEX", "UNRATE", "USREC"]:
        if key in mapping:
            print(f"[CHECK] {key} -> {mapping[key]}")
        else:
            print(f"[CHECK] {key} -> NOT FOUND")

    return mapping


def fetch_fred_observations(
    remote_series_id: str,
    api_key: str,
    observation_start: str | None = None,
    observation_end: str | None = None,
    timeout: int | None = None,
) -> pd.DataFrame:
    """
    Download observations from FRED between observation_start and observation_end.
    """
    timeout = timeout or getattr(CFG, "API_TIMEOUT", 30)

    params = {
        "series_id": remote_series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
    }
    if observation_start is not None:
        params["observation_start"] = observation_start
    if observation_end is not None:
        params["observation_end"] = observation_end

    r = requests.get(CFG.FRED_OBS_URL, params=params, timeout=timeout)
    r.raise_for_status()

    obs = r.json().get("observations", [])
    if not obs:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(obs)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if observation_end is not None:
        df = df[df["date"] <= pd.Timestamp(observation_end)].copy()

    return df


def load_dataset_fred_source() -> pd.DataFrame:
    dataset_path = None
    for path in DATASET_FRED_CANDIDATE_PATHS:
        if path.exists():
            dataset_path = path
            break

    if dataset_path is None:
        checked = "\n".join(str(p) for p in DATASET_FRED_CANDIDATE_PATHS)
        raise FileNotFoundError(
            "dataset_fred.csv not found. Checked these paths:\n"
            f"{checked}"
        )

    print(f"Using dataset source file: {dataset_path}")

    raw = pd.read_csv(dataset_path)

    if raw.empty:
        raise ValueError("dataset_fred.csv is empty")

    df = raw.copy()

    if "date" not in df.columns:
        raise ValueError("Column 'date' not found in dataset_fred.csv")

    first_date = pd.to_datetime(df["date"].iloc[0], errors="coerce")
    if pd.isna(first_date):
        df = df.iloc[1:].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return df


def ensure_series_exists(cur, series_id: str) -> None:
    cur.execute(
        """
        INSERT INTO macro.series(series_id)
        VALUES (%s)
        ON CONFLICT (series_id) DO NOTHING;
        """,
        (series_id,),
    )


def get_local_last_non_null_date(cur, series_id: str) -> str | None:
    cur.execute(
        """
        SELECT MAX(date)
        FROM macro.observations_monthly
        WHERE series_id = %s
          AND value IS NOT NULL;
        """,
        (series_id,),
    )
    value = cur.fetchone()[0]
    return format_date(value)


def get_local_total_obs(cur, series_id: str) -> int:
    cur.execute(
        """
        SELECT COUNT(*)
        FROM macro.observations_monthly
        WHERE series_id = %s;
        """,
        (series_id,),
    )
    return int(cur.fetchone()[0])


def get_local_first_date(cur, series_id: str) -> str | None:
    cur.execute(
        """
        SELECT MIN(date)
        FROM macro.observations_monthly
        WHERE series_id = %s;
        """,
        (series_id,),
    )
    value = cur.fetchone()[0]
    return format_date(value)


def get_local_last_date_any(cur, series_id: str) -> str | None:
    cur.execute(
        """
        SELECT MAX(date)
        FROM macro.observations_monthly
        WHERE series_id = %s;
        """,
        (series_id,),
    )
    value = cur.fetchone()[0]
    return format_date(value)


def get_local_null_info(cur, series_id: str, end_date: str | None = None) -> tuple[int, str | None]:
    if end_date is None:
        cur.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE value IS NULL) AS n_nulls,
                STRING_AGG(TO_CHAR(date, 'YYYY-MM-DD'), ', ' ORDER BY date)
                    FILTER (WHERE value IS NULL) AS null_dates
            FROM macro.observations_monthly
            WHERE series_id = %s;
            """,
            (series_id,),
        )
    else:
        cur.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE value IS NULL) AS n_nulls,
                STRING_AGG(TO_CHAR(date, 'YYYY-MM-DD'), ', ' ORDER BY date)
                    FILTER (WHERE value IS NULL) AS null_dates
            FROM macro.observations_monthly
            WHERE series_id = %s
              AND date <= %s;
            """,
            (series_id, end_date),
        )

    row = cur.fetchone()
    return int(row[0] or 0), row[1]


def get_existing_dates_in_range(cur, series_id: str, start_date: str, end_date: str) -> set[str]:
    cur.execute(
        """
        SELECT TO_CHAR(date, 'YYYY-MM-DD')
        FROM macro.observations_monthly
        WHERE series_id = %s
          AND date >= %s
          AND date <= %s;
        """,
        (series_id, start_date, end_date),
    )
    return {row[0] for row in cur.fetchall()}


def copy_dataframe_to_observations(cur, df_long: pd.DataFrame) -> int:
    """
    Bulk insert rows into macro.observations_monthly.
    """
    if df_long.empty:
        return 0

    buffer = StringIO()
    df_long.to_csv(buffer, index=False)
    buffer.seek(0)

    cur.copy_expert(
        """
        COPY macro.observations_monthly (date, series_id, value)
        FROM STDIN WITH CSV HEADER
        """,
        buffer,
    )

    return len(df_long)


def insert_refresh_log(
    cur,
    *,
    series_id: str,
    status: str,
    local_exists: bool | None,
    remote_exists: bool | None,
    local_total_obs: int | None,
    remote_total_obs: int | None,
    local_first_date: str | None,
    local_last_date: str | None,
    remote_first_date: str | None,
    remote_last_date: str | None,
    local_n_nulls: int | None,
    local_null_dates: str | None,
    missing_periods: int | None,
    missing_dates: str | None,
    validation_status: str | None,
    rows_added: int | None,
) -> None:
    cur.execute(
        """
        INSERT INTO macro.data_log (
            series_id,
            stage,
            action,
            status,
            local_exists,
            remote_exists,
            local_total_obs,
            remote_total_obs,
            local_first_date,
            local_last_date,
            remote_first_date,
            remote_last_date,
            local_n_nulls,
            local_null_dates,
            missing_periods,
            missing_dates,
            validation_status,
            rows_added
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """,
        (
            series_id,
            "fred_refresh",
            "run",
            status,
            local_exists,
            remote_exists,
            local_total_obs,
            remote_total_obs,
            local_first_date,
            local_last_date,
            remote_first_date,
            remote_last_date,
            local_n_nulls,
            local_null_dates,
            missing_periods,
            missing_dates,
            validation_status,
            rows_added,
        ),
    )


def build_missing_dates_from_df(df_remote_new: pd.DataFrame) -> str | None:
    if df_remote_new.empty:
        return None
    dates = (
        pd.to_datetime(df_remote_new["date"], errors="coerce")
        .dropna()
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )
    return ", ".join(dates) if dates else None


def compute_refresh_start(local_last_non_null_date: str | None, lookback_months: int) -> str | None:
    """
    Regarde une fenêtre glissante pour chercher les dates absentes,
    sans effacer les lignes déjà présentes.
    """
    if local_last_non_null_date is None:
        return getattr(CFG, "OBS_START", "1900-01-01")

    ts = pd.Timestamp(local_last_non_null_date).to_period("M").to_timestamp()
    ts = ts - pd.offsets.MonthBegin(lookback_months - 1)

    obs_start = pd.Timestamp(getattr(CFG, "OBS_START", "1900-01-01"))
    ts = max(ts, obs_start)

    return str(ts.date())


def _first_valid(x: pd.Series):
    x = x.dropna()
    return x.iloc[0] if not x.empty else None


def _last_valid(x: pd.Series):
    x = x.dropna()
    return x.iloc[-1] if not x.empty else None


def choose_monthly_rule(series_id: str, remote_series_id: str) -> str:
    series_id_u = str(series_id).upper()
    remote_id_u = str(remote_series_id).upper()

    if series_id_u in {"SP500", "OILPRICEX", "VIXCLSX"} or remote_id_u in {"SP500", "OILPRICE", "VIXCLS"}:
        return "last_valid"

    return "first_valid"


def to_monthly_start(df_remote: pd.DataFrame, rule: str = "first_valid") -> pd.DataFrame:
    """
    Convertit un DataFrame FRED (date, value) en fréquence mensuelle début de mois.
    """
    if df_remote.empty:
        return pd.DataFrame(columns=["date", "value"])

    df = df_remote.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    if df.empty:
        return pd.DataFrame(columns=["date", "value"])

    s = df.set_index("date")["value"]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    idx = pd.DatetimeIndex(s.index)

    if len(idx) > 0 and idx.day.nunique() == 1 and idx.day[0] == 1:
        out = s.reset_index()
        out.columns = ["date", "value"]
        return out

    grouped = s.groupby(s.index.to_period("M"))

    if rule == "first_valid":
        monthly = grouped.apply(_first_valid)
    elif rule == "last_valid":
        monthly = grouped.apply(_last_valid)
    elif rule == "mean":
        monthly = grouped.mean()
    else:
        raise ValueError(f"Unknown monthly rule: {rule}")

    monthly.index = monthly.index.to_timestamp(how="start")
    monthly.index.name = "date"

    out = monthly.reset_index()
    out.columns = ["date", "value"]
    out = out.sort_values("date").reset_index(drop=True)

    return out


def prepare_long_dataframe(df_values: pd.DataFrame, series_id: str) -> pd.DataFrame:
    df_long = df_values.copy()
    df_long["series_id"] = series_id
    df_long = df_long[["date", "series_id", "value"]]
    df_long["date"] = pd.to_datetime(df_long["date"]).dt.strftime("%Y-%m-%d")
    return df_long


def keep_only_missing_dates(cur, series_id: str, df_values: pd.DataFrame) -> pd.DataFrame:
    """
    Garde uniquement les dates absentes en base.
    Aucune suppression, aucune modification de l'existant.
    """
    if df_values.empty:
        return df_values.copy()

    start_date = format_date(df_values["date"].min())
    end_date = format_date(df_values["date"].max())

    existing_dates = get_existing_dates_in_range(cur, series_id, start_date, end_date)

    df = df_values.copy()
    df["date_str"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df[~df["date_str"].isin(existing_dates)].copy()
    df = df.drop(columns=["date_str"])

    return df.reset_index(drop=True)


def refresh_one_series(
    cur,
    series_id: str,
    remote_series_id: str,
    api_key: str,
    refresh_end_date: str,
    lookback_months: int,
) -> dict:
    """
    Refresh one series without deletion:
    insert only missing dates from the recent window.
    """
    local_last_non_null_date = get_local_last_non_null_date(cur, series_id)
    refresh_end_ts = pd.Timestamp(refresh_end_date)

    observation_start = compute_refresh_start(local_last_non_null_date, lookback_months)
    if observation_start is None:
        observation_start = getattr(CFG, "OBS_START", "1900-01-01")

    if pd.Timestamp(observation_start) > refresh_end_ts:
        return {
            "series_id": series_id,
            "remote_series_id": remote_series_id,
            "status": "UP_TO_DATE",
            "rows_added": 0,
            "rows_deleted": 0,
            "missing_dates": None,
            "missing_periods": 0,
            "validation_status": "OK",
            "remote_first_date": None,
            "remote_last_date": None,
            "remote_total_obs": 0,
            "local_last_non_null_date": local_last_non_null_date,
            "observation_start": observation_start,
            "monthly_rule": None,
            "source_type": "FRED",
        }

    df_remote_raw = fetch_fred_observations(
        remote_series_id=remote_series_id,
        api_key=api_key,
        observation_start=observation_start,
        observation_end=refresh_end_date,
    )

    if df_remote_raw.empty:
        return {
            "series_id": series_id,
            "remote_series_id": remote_series_id,
            "status": "UP_TO_DATE" if local_last_non_null_date is not None else "REMOTE_EMPTY",
            "rows_added": 0,
            "rows_deleted": 0,
            "missing_dates": None,
            "missing_periods": 0,
            "validation_status": "OK" if local_last_non_null_date is not None else None,
            "remote_first_date": None,
            "remote_last_date": None,
            "remote_total_obs": 0,
            "local_last_non_null_date": local_last_non_null_date,
            "observation_start": observation_start,
            "monthly_rule": None,
            "source_type": "FRED",
        }

    monthly_rule = choose_monthly_rule(series_id, remote_series_id)
    df_remote = to_monthly_start(df_remote_raw, rule=monthly_rule)
    df_remote = df_remote[df_remote["date"] <= refresh_end_ts].copy()

    if df_remote.empty:
        return {
            "series_id": series_id,
            "remote_series_id": remote_series_id,
            "status": "UP_TO_DATE",
            "rows_added": 0,
            "rows_deleted": 0,
            "missing_dates": None,
            "missing_periods": 0,
            "validation_status": "OK",
            "remote_first_date": None,
            "remote_last_date": None,
            "remote_total_obs": 0,
            "local_last_non_null_date": local_last_non_null_date,
            "observation_start": observation_start,
            "monthly_rule": monthly_rule,
            "source_type": "FRED",
        }

    remote_first_date = format_date(df_remote["date"].min())
    remote_last_date = format_date(df_remote["date"].max())
    remote_total_obs = len(df_remote)

    df_missing_only = keep_only_missing_dates(cur, series_id, df_remote)

    if df_missing_only.empty:
        return {
            "series_id": series_id,
            "remote_series_id": remote_series_id,
            "status": "UP_TO_DATE",
            "rows_added": 0,
            "rows_deleted": 0,
            "missing_dates": None,
            "missing_periods": 0,
            "validation_status": "OK",
            "remote_first_date": remote_first_date,
            "remote_last_date": remote_last_date,
            "remote_total_obs": remote_total_obs,
            "local_last_non_null_date": local_last_non_null_date,
            "observation_start": observation_start,
            "monthly_rule": monthly_rule,
            "source_type": "FRED",
        }

    df_long = prepare_long_dataframe(df_missing_only, series_id)

    ensure_series_exists(cur, series_id)
    rows_added = copy_dataframe_to_observations(cur, df_long)

    missing_dates = build_missing_dates_from_df(df_missing_only)
    validation_status = "OK" if rows_added == len(df_long) else "PARTIAL"

    return {
        "series_id": series_id,
        "remote_series_id": remote_series_id,
        "status": "SUCCESS" if rows_added > 0 else "NO_ROWS_INSERTED",
        "rows_added": rows_added,
        "rows_deleted": 0,
        "missing_dates": missing_dates,
        "missing_periods": len(df_long),
        "validation_status": validation_status,
        "remote_first_date": remote_first_date,
        "remote_last_date": remote_last_date,
        "remote_total_obs": remote_total_obs,
        "local_last_non_null_date": local_last_non_null_date,
        "observation_start": observation_start,
        "monthly_rule": monthly_rule,
        "source_type": "FRED",
    }


def refresh_one_special_source_series(
    cur,
    series_id: str,
    source_column: str,
    df_source: pd.DataFrame,
    refresh_end_date: str,
    lookback_months: int,
) -> dict:
    """
    Refresh one special series via API FRED without deletion:
    insert only missing dates.
    """
    local_last_non_null_date = get_local_last_non_null_date(cur, series_id)
    refresh_end_ts = pd.Timestamp(refresh_end_date)

    observation_start = compute_refresh_start(local_last_non_null_date, lookback_months)
    if observation_start is None:
        observation_start = getattr(CFG, "OBS_START", "1900-01-01")

    if pd.Timestamp(observation_start) > refresh_end_ts:
        return {
            "series_id": series_id,
            "remote_series_id": source_column,
            "status": "UP_TO_DATE",
            "rows_added": 0,
            "rows_deleted": 0,
            "missing_dates": None,
            "missing_periods": 0,
            "validation_status": "OK",
            "remote_first_date": None,
            "remote_last_date": None,
            "remote_total_obs": 0,
            "local_last_non_null_date": local_last_non_null_date,
            "observation_start": observation_start,
            "monthly_rule": None,
            "source_type": "FRED_SPECIAL",
        }

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY is missing.")

    df_remote_raw = fetch_fred_observations(
        remote_series_id=source_column,
        api_key=api_key,
        observation_start=observation_start,
        observation_end=refresh_end_date,
    )

    if df_remote_raw.empty:
        return {
            "series_id": series_id,
            "remote_series_id": source_column,
            "status": "UP_TO_DATE" if local_last_non_null_date is not None else "REMOTE_EMPTY",
            "rows_added": 0,
            "rows_deleted": 0,
            "missing_dates": None,
            "missing_periods": 0,
            "validation_status": "OK" if local_last_non_null_date is not None else None,
            "remote_first_date": None,
            "remote_last_date": None,
            "remote_total_obs": 0,
            "local_last_non_null_date": local_last_non_null_date,
            "observation_start": observation_start,
            "monthly_rule": None,
            "source_type": "FRED_SPECIAL",
        }

    monthly_rule = choose_monthly_rule(series_id, source_column)
    df_values = to_monthly_start(df_remote_raw, rule=monthly_rule)
    df_values = df_values[df_values["date"] <= refresh_end_ts].copy()

    if df_values.empty:
        return {
            "series_id": series_id,
            "remote_series_id": source_column,
            "status": "UP_TO_DATE",
            "rows_added": 0,
            "rows_deleted": 0,
            "missing_dates": None,
            "missing_periods": 0,
            "validation_status": "OK",
            "remote_first_date": None,
            "remote_last_date": None,
            "remote_total_obs": 0,
            "local_last_non_null_date": local_last_non_null_date,
            "observation_start": observation_start,
            "monthly_rule": monthly_rule,
            "source_type": "FRED_SPECIAL",
        }

    remote_first_date = format_date(df_values["date"].min())
    remote_last_date = format_date(df_values["date"].max())
    remote_total_obs = len(df_values)

    df_missing_only = keep_only_missing_dates(cur, series_id, df_values)

    if df_missing_only.empty:
        return {
            "series_id": series_id,
            "remote_series_id": source_column,
            "status": "UP_TO_DATE",
            "rows_added": 0,
            "rows_deleted": 0,
            "missing_dates": None,
            "missing_periods": 0,
            "validation_status": "OK",
            "remote_first_date": remote_first_date,
            "remote_last_date": remote_last_date,
            "remote_total_obs": remote_total_obs,
            "local_last_non_null_date": local_last_non_null_date,
            "observation_start": observation_start,
            "monthly_rule": monthly_rule,
            "source_type": "FRED_SPECIAL",
        }

    df_long = prepare_long_dataframe(df_missing_only, series_id)

    ensure_series_exists(cur, series_id)
    rows_added = copy_dataframe_to_observations(cur, df_long)

    missing_dates = build_missing_dates_from_df(df_missing_only)
    validation_status = "OK" if rows_added == len(df_long) else "PARTIAL"

    return {
        "series_id": series_id,
        "remote_series_id": source_column,
        "status": "SUCCESS" if rows_added > 0 else "NO_ROWS_INSERTED",
        "rows_added": rows_added,
        "rows_deleted": 0,
        "missing_dates": missing_dates,
        "missing_periods": len(df_long),
        "validation_status": validation_status,
        "remote_first_date": remote_first_date,
        "remote_last_date": remote_last_date,
        "remote_total_obs": remote_total_obs,
        "local_last_non_null_date": local_last_non_null_date,
        "observation_start": observation_start,
        "monthly_rule": monthly_rule,
        "source_type": "FRED_SPECIAL",
    }


def interpolate_isolated_monthly_nulls(cur, series_id: str, end_date: str) -> list[dict]:
    """
    Remplit uniquement les trous isolés:
    prev month non-null + current month null + next month non-null.
    """
    cur.execute(
        """
        WITH ordered AS (
            SELECT
                date,
                value,
                LAG(date)  OVER (ORDER BY date) AS prev_date,
                LAG(value) OVER (ORDER BY date) AS prev_value,
                LEAD(date)  OVER (ORDER BY date) AS next_date,
                LEAD(value) OVER (ORDER BY date) AS next_value
            FROM macro.observations_monthly
            WHERE series_id = %s
              AND date <= %s
        ),
        targets AS (
            SELECT
                date,
                ((prev_value + next_value) / 2.0) AS interp_value
            FROM ordered
            WHERE value IS NULL
              AND prev_value IS NOT NULL
              AND next_value IS NOT NULL
              AND prev_date = (date - INTERVAL '1 month')::date
              AND next_date = (date + INTERVAL '1 month')::date
        )
        UPDATE macro.observations_monthly t
        SET value = targets.interp_value
        FROM targets
        WHERE t.series_id = %s
          AND t.date = targets.date
        RETURNING t.date, t.value;
        """,
        (series_id, end_date, series_id),
    )

    rows = cur.fetchall()
    return [
        {"series_id": series_id, "date": format_date(r[0]), "value": float(r[1])}
        for r in rows
    ]


def main():
    args = parse_args()

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY is missing.")

    db_password = getpass.getpass("PostgreSQL password: ")
    fred_mapping = load_fred_mapping()

    if "USREC" not in fred_mapping:
        print("[WARNING] USREC absent du mapping -> ajout manuel")
        fred_mapping["USREC"] = "USREC"

    print("\n[CHECK MAPPING FINAL]")
    for key in ["SP500", "OILPRICEX", "UNRATE", "USREC"]:
        if key in fred_mapping:
            print(f"{key} -> {fred_mapping[key]}")
        else:
            print(f"{key} -> NOT FOUND")

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=db_password,
        host=DB_HOST,
        port=DB_PORT,
    )
    conn.autocommit = False

    refresh_results = []
    interpolation_results = []

    try:
        cur = conn.cursor()

        all_series = sorted(
            (series_id, remote_series_id)
            for series_id, remote_series_id in fred_mapping.items()
            if series_id not in SPECIAL_SOURCE_SERIES_MAP
        )

        print(f"Number of FRED series in mapping: {len(all_series)}")
        print(f"Number of special source series: {len(SPECIAL_SOURCE_SERIES_MAP)}")
        print(f"Refresh capped at: {args.refresh_end_date}")
        print(f"Lookback window (months): {args.lookback_months}")
        print(f"Impute after refresh: {args.impute_after_refresh}")
        print("\nFRED DATA REFRESH")
        print("=" * 60)

        sleep_every = getattr(CFG, "SLEEP_EVERY", 0)
        sleep_seconds = getattr(CFG, "SLEEP_SECONDS", 0)

        for i, (series_id, remote_series_id) in enumerate(all_series, start=1):
            print(f"\nSeries: {series_id} ({i}/{len(all_series)})")
            print("-" * 60)
            print(f"remote_series_id: {remote_series_id}")

            local_first_before = get_local_first_date(cur, series_id)
            local_last_before = get_local_last_date_any(cur, series_id)
            local_last_non_null_before = get_local_last_non_null_date(cur, series_id)
            local_total_before = get_local_total_obs(cur, series_id)
            local_n_nulls, local_null_dates = get_local_null_info(cur, series_id, end_date=args.refresh_end_date)

            try:
                result = refresh_one_series(
                    cur=cur,
                    series_id=series_id,
                    remote_series_id=remote_series_id,
                    api_key=api_key,
                    refresh_end_date=args.refresh_end_date,
                    lookback_months=args.lookback_months,
                )
            except Exception as e:
                result = {
                    "series_id": series_id,
                    "remote_series_id": remote_series_id,
                    "status": "FAILED",
                    "rows_added": 0,
                    "rows_deleted": 0,
                    "missing_dates": None,
                    "missing_periods": None,
                    "validation_status": str(e),
                    "remote_first_date": None,
                    "remote_last_date": None,
                    "remote_total_obs": None,
                    "local_last_non_null_date": local_last_non_null_before,
                    "observation_start": None,
                    "monthly_rule": None,
                    "source_type": "FRED",
                }

            refresh_results.append(result)

            insert_refresh_log(
                cur,
                series_id=series_id,
                status=result["status"],
                local_exists=(local_total_before > 0),
                remote_exists=(result["status"] != "FAILED"),
                local_total_obs=local_total_before,
                remote_total_obs=result.get("remote_total_obs"),
                local_first_date=local_first_before,
                local_last_date=local_last_before,
                remote_first_date=result.get("remote_first_date"),
                remote_last_date=result.get("remote_last_date"),
                local_n_nulls=local_n_nulls,
                local_null_dates=local_null_dates,
                missing_periods=result.get("missing_periods"),
                missing_dates=result.get("missing_dates"),
                validation_status=result.get("validation_status"),
                rows_added=result.get("rows_added"),
            )

            print(f"local_first_date_before: {local_first_before}")
            print(f"local_last_date_any_before: {local_last_before}")
            print(f"local_last_non_null_before: {local_last_non_null_before}")
            print(f"observation_start_used: {result.get('observation_start')}")
            print(f"monthly_rule: {result.get('monthly_rule')}")
            print(f"source_type: {result.get('source_type')}")
            print(f"status: {result['status']}")
            print(f"rows_deleted: {result.get('rows_deleted')}")
            print(f"rows_added: {result['rows_added']}")
            print(f"validation_status: {result['validation_status']}")
            print(f"remote_last_date: {result.get('remote_last_date')}")

            if sleep_every and i % sleep_every == 0:
                time.sleep(sleep_seconds)

        print("\nSPECIAL SOURCE REFRESH")
        print("=" * 60)

        df_source = None

        for series_id, source_column in SPECIAL_SOURCE_SERIES_MAP.items():
            print(f"\nSeries: {series_id}")
            print("-" * 60)
            print(f"source_column: {source_column}")

            local_first_before = get_local_first_date(cur, series_id)
            local_last_before = get_local_last_date_any(cur, series_id)
            local_last_non_null_before = get_local_last_non_null_date(cur, series_id)
            local_total_before = get_local_total_obs(cur, series_id)
            local_n_nulls, local_null_dates = get_local_null_info(cur, series_id, end_date=args.refresh_end_date)

            try:
                result = refresh_one_special_source_series(
                    cur=cur,
                    series_id=series_id,
                    source_column=source_column,
                    df_source=df_source,
                    refresh_end_date=args.refresh_end_date,
                    lookback_months=args.lookback_months,
                )
            except Exception as e:
                result = {
                    "series_id": series_id,
                    "remote_series_id": source_column,
                    "status": "FAILED",
                    "rows_added": 0,
                    "rows_deleted": 0,
                    "missing_dates": None,
                    "missing_periods": None,
                    "validation_status": str(e),
                    "remote_first_date": None,
                    "remote_last_date": None,
                    "remote_total_obs": None,
                    "local_last_non_null_date": local_last_non_null_before,
                    "observation_start": None,
                    "monthly_rule": None,
                    "source_type": "FRED_SPECIAL",
                }

            refresh_results.append(result)

            insert_refresh_log(
                cur,
                series_id=series_id,
                status=result["status"],
                local_exists=(local_total_before > 0),
                remote_exists=(result["status"] != "FAILED"),
                local_total_obs=local_total_before,
                remote_total_obs=result.get("remote_total_obs"),
                local_first_date=local_first_before,
                local_last_date=local_last_before,
                remote_first_date=result.get("remote_first_date"),
                remote_last_date=result.get("remote_last_date"),
                local_n_nulls=local_n_nulls,
                local_null_dates=local_null_dates,
                missing_periods=result.get("missing_periods"),
                missing_dates=result.get("missing_dates"),
                validation_status=result.get("validation_status"),
                rows_added=result.get("rows_added"),
            )

            print(f"local_first_date_before: {local_first_before}")
            print(f"local_last_date_any_before: {local_last_before}")
            print(f"local_last_non_null_before: {local_last_non_null_before}")
            print(f"observation_start_used: {result.get('observation_start')}")
            print(f"monthly_rule: {result.get('monthly_rule')}")
            print(f"source_type: {result.get('source_type')}")
            print(f"status: {result['status']}")
            print(f"rows_deleted: {result.get('rows_deleted')}")
            print(f"rows_added: {result['rows_added']}")
            print(f"validation_status: {result['validation_status']}")
            print(f"remote_last_date: {result.get('remote_last_date')}")

        if args.impute_after_refresh:
            print("\nPOST-REFRESH INTERPOLATION")
            print("=" * 60)

            for series_id in sorted(POST_REFRESH_INTERPOLATION_SERIES):
                updates = interpolate_isolated_monthly_nulls(
                    cur=cur,
                    series_id=series_id,
                    end_date=args.refresh_end_date,
                )

                if updates:
                    print(f"{series_id}: {len(updates)} value(s) interpolated")
                    for row in updates:
                        print(f"  {row['date']} -> {row['value']}")
                    interpolation_results.extend(updates)
                else:
                    print(f"{series_id}: no isolated null to interpolate")

        conn.commit()
        print("\nFRED data refresh logged into macro.data_log")

    except Exception as e:
        conn.rollback()
        print("Error during data refresh. Rollback executed.")
        print(e)
        raise

    finally:
        conn.close()
        print("PostgreSQL connection closed.")

    out_dir = PROJECT_ROOT / "artifacts" / "data_intake"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "fred_refresh_results.csv"
    pd.DataFrame(refresh_results).to_csv(out_path, index=False)
    print(f"Refresh results saved to: {out_path}")

    if interpolation_results:
        interp_path = out_dir / "fred_refresh_interpolations.csv"
        pd.DataFrame(interpolation_results).to_csv(interp_path, index=False)
        print(f"Interpolation results saved to: {interp_path}")


if __name__ == "__main__":
    main()