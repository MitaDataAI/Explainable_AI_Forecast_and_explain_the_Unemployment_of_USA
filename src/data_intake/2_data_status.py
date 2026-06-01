import sys
from pathlib import Path
import os
import getpass
import time

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


def format_date(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(pd.Timestamp(value).normalize().date())


def build_missing_dates(local_last_date, remote_last_date) -> list[str]:
    if local_last_date is None or remote_last_date is None:
        return []

    local_last_date = pd.Timestamp(local_last_date).normalize()
    remote_last_date = pd.Timestamp(remote_last_date).normalize()

    if remote_last_date <= local_last_date:
        return []

    missing_idx = pd.date_range(
        start=local_last_date + pd.offsets.MonthBegin(1),
        end=remote_last_date,
        freq="MS",
    )
    return [str(d.date()) for d in missing_idx]


def get_all_series_ids(cur) -> list[str]:
    cur.execute(
        """
        SELECT series_id
        FROM macro.series
        ORDER BY series_id;
        """
    )
    rows = cur.fetchall()
    return [row[0] for row in rows]


def load_fred_mapping() -> dict[str, str]:
    """
    Charge le mapping local -> FRED.

    Compatible avec un fichier de type :
    raw_name, normalized_name, canonical_name, status

    Ici, on utilise canonical_name comme :
    - nom local
    - identifiant remote FRED

    Le filtre sur status n'est appliqué que s'il trouve réellement des correspondances.
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

    for key in ["SP500", "OILPRICEX", "UNRATE"]:
        if key in mapping:
            print(f"[CHECK] {key} -> {mapping[key]}")
        else:
            print(f"[CHECK] {key} -> NOT FOUND")

    return mapping


def get_fred_series_status(
    remote_series_id: str,
    api_key: str,
    timeout: int | None = None,
) -> dict:
    timeout = timeout or CFG.API_TIMEOUT

    params = {
        "series_id": remote_series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
        "observation_start": CFG.OBS_START,
    }

    try:
        r = requests.get(CFG.FRED_OBS_URL, params=params, timeout=timeout)

        if r.status_code != 200:
            detail = ""
            try:
                j = r.json()
                detail = j.get("error_message") or j.get("message") or str(j)
            except Exception:
                detail = r.text[:300]

            return {
                "remote_exists": False,
                "remote_total_obs": 0,
                "remote_first_date": None,
                "remote_last_date": None,
                "remote_error": f"HTTP {r.status_code}: {detail}",
            }

        payload = r.json()
        obs = payload.get("observations", [])

        if not obs:
            return {
                "remote_exists": False,
                "remote_total_obs": 0,
                "remote_first_date": None,
                "remote_last_date": None,
                "remote_error": None,
            }

        df = pd.DataFrame(obs)

        if "date" not in df.columns:
            return {
                "remote_exists": False,
                "remote_total_obs": 0,
                "remote_first_date": None,
                "remote_last_date": None,
                "remote_error": "FRED response missing 'date' column",
            }

        df = df[["date"]].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        if df.empty:
            return {
                "remote_exists": False,
                "remote_total_obs": 0,
                "remote_first_date": None,
                "remote_last_date": None,
                "remote_error": None,
            }

        return {
            "remote_exists": True,
            "remote_total_obs": int(len(df)),
            "remote_first_date": df["date"].min().normalize(),
            "remote_last_date": df["date"].max().normalize(),
            "remote_error": None,
        }

    except requests.exceptions.RequestException as e:
        return {
            "remote_exists": False,
            "remote_total_obs": 0,
            "remote_first_date": None,
            "remote_last_date": None,
            "remote_error": str(e),
        }


def get_local_series_status(cur, series_id: str) -> dict:
    cur.execute(
        """
        SELECT
            COUNT(*) AS total_obs,
            MIN(date) AS first_date,
            MAX(date) AS last_date,
            COUNT(*) FILTER (WHERE value IS NULL) AS n_nulls
        FROM macro.observations_monthly
        WHERE series_id = %s;
        """,
        (series_id,),
    )

    row = cur.fetchone()

    if row is None:
        return {
            "series_id": series_id,
            "local_exists": False,
            "local_total_obs": 0,
            "local_first_date": None,
            "local_last_date": None,
            "local_n_nulls": 0,
            "local_null_dates": [],
        }

    total_obs, first_date, last_date, n_nulls = row
    total_obs = int(total_obs or 0)
    n_nulls = int(n_nulls or 0)

    local_null_dates = []

    if total_obs > 0 and n_nulls > 0:
        cur.execute(
            """
            SELECT date
            FROM macro.observations_monthly
            WHERE series_id = %s
              AND value IS NULL
            ORDER BY date;
            """,
            (series_id,),
        )

        null_rows = cur.fetchall()
        local_null_dates = [str(pd.Timestamp(r[0]).date()) for r in null_rows]

    local_exists = total_obs > 0

    return {
        "series_id": series_id,
        "local_exists": local_exists,
        "local_total_obs": total_obs,
        "local_first_date": first_date,
        "local_last_date": last_date,
        "local_n_nulls": n_nulls,
        "local_null_dates": local_null_dates,
    }


def compute_data_status(local_status: dict, remote_status: dict, remote_series_id: str | None) -> dict:
    series_id = local_status["series_id"]
    local_last = local_status["local_last_date"]
    remote_last = remote_status["remote_last_date"]

    if remote_series_id is None:
        status = "REMOTE_ID_MISSING"
        missing_periods = None
        missing_dates = []

    elif remote_status.get("remote_error"):
        status = "REMOTE_ERROR"
        missing_periods = None
        missing_dates = []

    elif not remote_status["remote_exists"]:
        status = "REMOTE_EMPTY"
        missing_periods = None
        missing_dates = []

    elif not local_status["local_exists"]:
        status = "LOCAL_MISSING"
        missing_periods = remote_status["remote_total_obs"]
        missing_dates = []

    elif local_last is None:
        status = "LOCAL_CORRUPTED"
        missing_periods = None
        missing_dates = []

    else:
        local_last = pd.Timestamp(local_last).normalize()
        remote_last = pd.Timestamp(remote_last).normalize()

        if remote_last > local_last:
            status = "UPDATE_NEEDED"
            missing_dates = build_missing_dates(local_last, remote_last)
            missing_periods = len(missing_dates)
        elif remote_last == local_last:
            status = "UP_TO_DATE"
            missing_periods = 0
            missing_dates = []
        else:
            status = "LOCAL_AHEAD_OF_REMOTE"
            missing_periods = 0
            missing_dates = []

    return {
        "series_id": series_id,
        "remote_series_id": remote_series_id,
        "status": status,
        "local_exists": local_status["local_exists"],
        "remote_exists": remote_status["remote_exists"],
        "local_total_obs": local_status["local_total_obs"],
        "remote_total_obs": remote_status["remote_total_obs"],
        "local_first_date": format_date(local_status["local_first_date"]),
        "local_last_date": format_date(local_status["local_last_date"]),
        "remote_first_date": format_date(remote_status["remote_first_date"]),
        "remote_last_date": format_date(remote_status["remote_last_date"]),
        "local_n_nulls": local_status["local_n_nulls"],
        "local_null_dates": ", ".join(local_status["local_null_dates"]) if local_status["local_null_dates"] else None,
        "missing_periods": missing_periods,
        "missing_dates": ", ".join(missing_dates) if missing_dates else None,
        "remote_error": remote_status.get("remote_error"),
    }


def insert_data_log(cur, result: dict) -> None:
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
            result["series_id"],
            "fred_data_status",
            "check",
            result["status"],
            result["local_exists"],
            result["remote_exists"],
            result["local_total_obs"],
            result["remote_total_obs"],
            result["local_first_date"],
            result["local_last_date"],
            result["remote_first_date"],
            result["remote_last_date"],
            result["local_n_nulls"],
            result["local_null_dates"],
            result["missing_periods"],
            result["missing_dates"],
            None,
            None,
        ),
    )


def check_data_status(cur, series_id: str, remote_series_id: str | None, api_key: str) -> dict:
    local_status = get_local_series_status(cur, series_id)

    if remote_series_id is None:
        remote_status = {
            "remote_exists": False,
            "remote_total_obs": 0,
            "remote_first_date": None,
            "remote_last_date": None,
            "remote_error": None,
        }
    else:
        remote_status = get_fred_series_status(remote_series_id, api_key)

    return compute_data_status(local_status, remote_status, remote_series_id)


def print_result(result: dict) -> None:
    ordered_keys = [
        "series_id",
        "remote_series_id",
        "status",
        "local_exists",
        "remote_exists",
        "local_total_obs",
        "remote_total_obs",
        "local_first_date",
        "local_last_date",
        "remote_first_date",
        "remote_last_date",
        "local_n_nulls",
        "local_null_dates",
        "missing_periods",
        "missing_dates",
        "remote_error",
    ]

    for key in ordered_keys:
        print(f"{key}: {result.get(key)}")


def main():
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY is missing.")

    db_password = getpass.getpass("PostgreSQL password: ")
    results = []
    fred_mapping = load_fred_mapping()

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=db_password,
        host=DB_HOST,
        port=DB_PORT,
    )
    conn.autocommit = False

    try:
        cur = conn.cursor()

        series_to_check = get_all_series_ids(cur)
        print(f"Number of features to check: {len(series_to_check)}")
        print("\nFRED DATA STATUS")
        print("=" * 60)

        sleep_every = getattr(CFG, "SLEEP_EVERY", 0)
        sleep_seconds = getattr(CFG, "SLEEP_SECONDS", 0)

        for i, series_id in enumerate(series_to_check, start=1):
            print(f"\nSeries: {series_id} ({i}/{len(series_to_check)})")
            print("-" * 60)

            remote_series_id = fred_mapping.get(series_id)
            result = check_data_status(cur, series_id, remote_series_id, api_key)
            results.append(result)

            print_result(result)
            insert_data_log(cur, result)

            if result["status"] == "UP_TO_DATE":
                print("No update needed.")
            elif result["status"] == "UPDATE_NEEDED":
                print(f"Update required: {result['missing_periods']} missing period(s).")
            elif result["status"] == "LOCAL_MISSING":
                print("Series not found locally. Initial load required.")
            elif result["status"] == "REMOTE_ID_MISSING":
                print("Remote FRED id missing in mapping.")
            elif result["status"] == "REMOTE_ERROR":
                print("Remote FRED request failed.")
            elif result["status"] == "REMOTE_EMPTY":
                print("Series not found on remote source.")
            else:
                print(f"Status detected: {result['status']}")

            if sleep_every and i % sleep_every == 0:
                time.sleep(sleep_seconds)

        conn.commit()
        print("\nFRED data status logged into macro.data_log")

    except Exception as e:
        conn.rollback()
        print("Error during data status logging. Rollback executed.")
        print(e)
        raise

    finally:
        conn.close()
        print("PostgreSQL connection closed.")

    out_dir = PROJECT_ROOT / "artifacts" / "data_intake"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "fred_data_status.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)

    print(f"Status saved to: {out_path}")


if __name__ == "__main__":
    main()