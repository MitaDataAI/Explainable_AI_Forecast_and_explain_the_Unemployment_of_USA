# src/data_intake/4_data_validation.py

import sys
from pathlib import Path
import os
import getpass

import pandas as pd
import psycopg2
import pointblank as pb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

DB_NAME = "unemployment_usa"
DB_USER = "postgres"
DB_HOST = "localhost"
DB_PORT = 5432

VALIDATION_SERIES_IDS = [
    "BUSLOANS", "CPIAUCSL", "DPCERA3M086SBEA", "INDPRO",
    "M2SL", "OILPRICEX", "RPI", "SP500", "TB3MS", "UNRATE", "USREC",
]

OPTIONAL_SERIES_IDS = {"USREC"}

# borne max demandée
REFRESH_END_DATE = os.getenv("REFRESH_END_DATE", "2026-01-01")


# =========================
# EFFECTIVE END DATE
# =========================
def get_effective_validation_end_date(cur, requested_end_date: str) -> str:
    """
    La date effective de validation est le minimum des dernières dates
    disponibles parmi les séries suivies, sous la borne requested_end_date.
    """
    cur.execute(
        """
        SELECT MIN(last_date)::date
        FROM (
            SELECT
                series_id,
                MAX(date) AS last_date
            FROM macro.observations_monthly
            WHERE series_id = ANY(%s)
              AND date <= %s
            GROUP BY series_id
        ) t;
        """,
        (VALIDATION_SERIES_IDS, requested_end_date),
    )
    value = cur.fetchone()[0]

    if value is None:
        raise ValueError("Impossible de déterminer la date effective de validation.")

    return str(pd.Timestamp(value).date())


# =========================
# FETCH
# =========================
def fetch_validation_dataframe(cur, end_date: str) -> pd.DataFrame:
    cur.execute(
        """
        SELECT date, series_id, value
        FROM macro.observations_monthly
        WHERE series_id = ANY(%s)
          AND date <= %s
        ORDER BY series_id, date;
        """,
        (VALIDATION_SERIES_IDS, end_date),
    )
    rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=["date", "series_id", "value"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["series_id"] = df["series_id"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


# =========================
# LOG
# =========================
def insert_validation_log(cur, status: str, validation_status: str) -> None:
    cur.execute(
        """
        INSERT INTO macro.data_log (
            series_id,
            stage,
            action,
            status,
            validation_status
        )
        VALUES (%s, %s, %s, %s, %s);
        """,
        (
            None,
            "fred_data_validation",
            "run",
            status,
            validation_status,
        ),
    )


# =========================
# REPORT
# =========================
def save_pointblank_report(validation, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(validation.get_json_report())


# =========================
# DEBUG NULL
# =========================
def print_null_summary(df: pd.DataFrame, title: str) -> None:
    nulls = df[df["value"].isna()].copy()

    print(f"\n{title}")
    print("-" * len(title))

    if nulls.empty:
        print("No NULL values found.")
    else:
        print(nulls[["date", "series_id", "value"]].to_string(index=False))


# =========================
# IMPUTATION
# =========================
def impute_linear_in_db(cur, end_date: str) -> list[dict]:
    """
    Imputation linéaire pour les NaN internes par série.
    Ne remplit pas les bords de série.
    """
    cur.execute(
        """
        WITH ordered AS (
            SELECT
                series_id,
                date,
                value,
                LAG(date) OVER (PARTITION BY series_id ORDER BY date) AS prev_date,
                LAG(value) OVER (PARTITION BY series_id ORDER BY date) AS prev_value,
                LEAD(date) OVER (PARTITION BY series_id ORDER BY date) AS next_date,
                LEAD(value) OVER (PARTITION BY series_id ORDER BY date) AS next_value
            FROM macro.observations_monthly
            WHERE series_id = ANY(%s)
              AND date <= %s
        ),
        targets AS (
            SELECT
                series_id,
                date,
                (
                    prev_value
                    + (next_value - prev_value)
                      * (
                          (
                              (EXTRACT(YEAR FROM age(date, prev_date)) * 12)
                              + EXTRACT(MONTH FROM age(date, prev_date))
                            )
                          /
                          NULLIF(
                              (
                                  (EXTRACT(YEAR FROM age(next_date, prev_date)) * 12)
                                  + EXTRACT(MONTH FROM age(next_date, prev_date))
                              ),
                              0
                          )
                        )
                  )::double precision AS interp_value
            FROM ordered
            WHERE value IS NULL
              AND prev_value IS NOT NULL
              AND next_value IS NOT NULL
        )
        UPDATE macro.observations_monthly t
        SET value = targets.interp_value
        FROM targets
        WHERE t.series_id = targets.series_id
          AND t.date = targets.date
        RETURNING t.series_id, t.date, t.value;
        """,
        (VALIDATION_SERIES_IDS, end_date),
    )

    rows = cur.fetchall()
    return [
        {
            "series_id": r[0],
            "date": str(pd.Timestamp(r[1]).date()),
            "value": float(r[2]),
        }
        for r in rows
    ]


# =========================
# BUSINESS LOGIC
# =========================
def check_monthly_frequency(df: pd.DataFrame):
    fail_issues = []
    warning_issues = []

    for sid in VALIDATION_SERIES_IDS:
        sub = df[df["series_id"] == sid].sort_values("date").copy()

        if sub.empty:
            if sid in OPTIONAL_SERIES_IDS:
                warning_issues.append(f"{sid}: missing series (skipped)")
            else:
                fail_issues.append(f"{sid}: missing series")
            continue

        bad_day = sub[sub["date"].dt.day != 1]
        if not bad_day.empty:
            fail_issues.append(f"{sid}: non-month-start dates detected")

        full_range = pd.date_range(
            start=sub["date"].min(),
            end=sub["date"].max(),
            freq="MS",
        )
        missing = full_range.difference(pd.DatetimeIndex(sub["date"]))

        if len(missing) > 0:
            preview = ", ".join(pd.Series(missing).dt.strftime("%Y-%m-%d").head(10).tolist())
            fail_issues.append(f"{sid}: {len(missing)} missing months ({preview})")

    return fail_issues, warning_issues


# =========================
# MAIN
# =========================
def main():
    db_password = getpass.getpass("PostgreSQL password: ")

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

        print(f"\nValidation requested end date: {REFRESH_END_DATE}")

        effective_end_date = get_effective_validation_end_date(cur, REFRESH_END_DATE)
        print(f"Validation effective end date: {effective_end_date}")

        df_validation = fetch_validation_dataframe(cur, effective_end_date)

        if df_validation.empty:
            raise ValueError("No data found for validation.")

        print(f"\nRows loaded: {len(df_validation)}")
        print_null_summary(df_validation, "NULL BEFORE IMPUTATION")

        updates = impute_linear_in_db(cur, effective_end_date)

        print(f"\nImputed values: {len(updates)}")
        if updates:
            print(pd.DataFrame(updates).to_string(index=False))

        df_validation = fetch_validation_dataframe(cur, effective_end_date)
        print_null_summary(df_validation, "NULL AFTER IMPUTATION")

        present_series = sorted(df_validation["series_id"].dropna().unique().tolist())
        required_for_pointblank = [
            sid for sid in VALIDATION_SERIES_IDS if sid not in OPTIONAL_SERIES_IDS or sid in present_series
        ]

        validation = (
            pb.Validate(
                data=df_validation,
                tbl_name="macro.observations_monthly",
                label="FRED Validation",
                thresholds=pb.Thresholds(warning=0.1, error=0, critical=0.1),
            )
            .col_vals_not_null(columns=["date", "series_id", "value"])
            .col_vals_ge(columns="value", value=0)
            .col_vals_in_set(columns="series_id", set=required_for_pointblank)
            .rows_distinct(columns_subset=["date", "series_id"])
            .interrogate()
        )

        monthly_fail_issues, monthly_warning_issues = check_monthly_frequency(df_validation)

        if monthly_fail_issues:
            print("\nBusiness logic FAIL issues:")
            for issue in monthly_fail_issues:
                print("-", issue)

        if monthly_warning_issues:
            print("\nBusiness logic WARNING issues:")
            for issue in monthly_warning_issues:
                print("-", issue)

        main_status = "PASS" if validation.all_passed() else "FAIL"
        business_status = "PASS" if not monthly_fail_issues else "FAIL"

        global_status = "PASS"
        if main_status == "FAIL" or business_status == "FAIL":
            global_status = "FAIL"

        report_path = PROJECT_ROOT / "artifacts" / "data_intake" / "fred_data_validation_pointblank_report.json"
        save_pointblank_report(validation, report_path)

        insert_validation_log(
            cur,
            status=global_status,
            validation_status=(
                f"pointblank={main_status}; "
                f"business={business_status}; "
                f"imputed={len(updates)}; "
                f"warnings={len(monthly_warning_issues)}; "
                f"requested_end_date={REFRESH_END_DATE}; "
                f"effective_end_date={effective_end_date}; "
                f"report={report_path}"
            ),
        )

        if global_status == "FAIL":
            raise ValueError("Data validation failed.")

        conn.commit()
        print("\nVALIDATION SUCCESS")
        print(f"Pointblank report saved to: {report_path}")

    except Exception as e:
        conn.rollback()
        print("\nVALIDATION FAILED -> rollback")
        print(e)
        raise

    finally:
        conn.close()
        print("PostgreSQL connection closed.")


if __name__ == "__main__":
    main()