from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
from feast import FeatureStore

# =========================================================
# 1) Utils: trouver la racine projet + repo Feast
# =========================================================
def find_project_root(start: Path, marker: str = "2_data_processing") -> Path:
    """Walk up parents until we find a folder named `marker`."""
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(
        f"Cannot find project root: marker '{marker}' not found from {start}"
    )

def get_feast_repo_path(project_root: Path) -> Path:
    """Return the expected Feast repo path."""
    feast_repo = (
        project_root
        / "2_data_processing"
        / "feature_store"
        / "feast_repo"
        / "feature_repo"
    )
    if not feast_repo.exists():
        raise FileNotFoundError(f"Feast repo path not found: {feast_repo}")
    return feast_repo

# =========================================================
# 2) Utils: build entity_df
# =========================================================
def build_entity_df(
    series_ids: list[str],
    start: str = "1960-01-01",
    end: str = "2025-08-01",
    freq: str = "MS",
) -> pd.DataFrame:
    """Create the entity dataframe (series_id, date) for Feast."""
    dates = pd.date_range(start=start, end=end, freq=freq)
    entity_df = pd.MultiIndex.from_product(
        [series_ids, dates],
        names=["series_id", "date"],
    ).to_frame(index=False)
    return entity_df

# =========================================================
# 3) Utils: load features from Feast and pivot to wide
# =========================================================
def load_wide_from_feast(
    feature_ref: str,
    series_ids: list[str],
    start: str = "1960-01-01",
    end: str = "2025-08-01",
    freq: str = "MS",
    project_marker: str = "2_data_processing",
    expected_value_col: str | None = None,
) -> pd.DataFrame:
    """
    Load a single feature (e.g., "raw_value:value" or "stationary_value:value")
    from Feast, then pivot to wide dataframe (index=date, columns=series_id).

    Parameters
    ----------
    feature_ref : str
        Feast feature reference, e.g. "raw_value:value" or "stationary_value:value"
    expected_value_col : str | None
        If you know the exact returned column name (e.g. "raw_value__value"),
        you can pass it. Otherwise it auto-detects the returned value column.
    """
    # Locate Feast repo
    project_root = find_project_root(Path.cwd(), marker=project_marker)
    feast_repo_path = get_feast_repo_path(project_root)

    # Init Feast store
    store = FeatureStore(repo_path=str(feast_repo_path))

    # Build entity df
    entity_df = build_entity_df(series_ids, start=start, end=end, freq=freq)

    # Retrieve features
    df_long = store.get_historical_features(
        entity_df=entity_df,
        features=[feature_ref],
    ).to_df()

    # Identify the value column produced by Feast
    if expected_value_col is not None:
        value_col = expected_value_col
        if value_col not in df_long.columns:
            raise KeyError(
                f"expected_value_col='{value_col}' not found. "
                f"Available columns: {list(df_long.columns)}"
            )
    else:
        # ✅ Case A: Feast returns directly "value"
        if "value" in df_long.columns and "series_id" in df_long.columns and "date" in df_long.columns:
            value_col = "value"
        else:
            # ✅ Case B (standard): "<feature_view>__<feature_name>"
            candidates = [c for c in df_long.columns if "__" in c and c not in ("series_id", "date")]
            if len(candidates) == 0:
                raise KeyError(
                    "No Feast value column found. "
                    f"Available columns: {list(df_long.columns)}"
                )

            if len(candidates) > 1:
                view, feat = feature_ref.split(":")
                preferred = f"{view}__{feat}"
                value_col = preferred if preferred in candidates else candidates[0]
            else:
                value_col = candidates[0]

    # LONG -> WIDE
    df_wide = (
        df_long
        .rename(columns={value_col: "value"})
        .pivot(index="date", columns="series_id", values="value")
        .sort_index()
    )

    # Ensure datetime index
    df_wide.index = pd.to_datetime(df_wide.index)
    return df_wide


# =========================================================
# 4) Utils: Build dataset (Target UNRATE + Exogenous variables)
# =========================================================
def build_unrate_exog_dataset(
    df_stationary: pd.DataFrame,
    target_id: str = "UNRATE",
    value_col: str = "value",
    series_col: str = "series_id",
    date_col: str = "date",
    unique_id: str = "UNRATE",
    dropna: bool = True,
    align_to_month_start: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Target variable: stationary UNRATE
    Exogenous variables: other macro series at the same timestamp

    Input df_stationary must be LONG format with columns:
      - series_id, date, value

    Returns:
      - df_model: wide dataset [date, y, <exog...>]
      - ts_lr: MLForecast long format [unique_id, ds, y, <exog...>]
      - exog_cols: list of exogenous column names
    """
    required = {series_col, date_col, value_col}
    missing = required - set(df_stationary.columns)
    if missing:
        raise ValueError(f"df_stationary is missing columns: {sorted(missing)}")

    df = df_stationary[[series_col, date_col, value_col]].copy()

    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Optional: enforce month-start timestamps (MS)
    if align_to_month_start:
        df[date_col] = (
            df[date_col]
            .dt.to_period("M")
            .dt.to_timestamp(how="start")
            .dt.normalize()
        )

    # -------------------------
    # Target y
    # -------------------------
    df_y = (
        df[df[series_col] == target_id]
        .sort_values(date_col)
        .rename(columns={value_col: "y"})
        .reset_index(drop=True)
    )

    # -------------------------
    # Exogenous X (all other series)
    # -------------------------
    df_x_long = df[df[series_col] != target_id].copy()

    # Keep only dates present in y
    df_x_long = df_x_long[df_x_long[date_col].isin(df_y[date_col])]

    df_x = (
        df_x_long
        .pivot_table(index=date_col, columns=series_col, values=value_col, aggfunc="last")
        .reset_index()
    )

    # Merge y + X
    df_model = (
        df_y[[date_col, "y"]]
        .merge(df_x, on=date_col, how="left")
    )

    if dropna:
        df_model = df_model.dropna()

    # -------------------------
    # MLForecast format
    # -------------------------
    ts_lr = df_model.rename(columns={date_col: "ds"}).copy()
    ts_lr["unique_id"] = unique_id

    exog_cols = [c for c in ts_lr.columns if c not in ["unique_id", "ds", "y"]]
    ts_lr = ts_lr[["unique_id", "ds", "y"] + exog_cols].copy()

    # Ensure ds is month-start normalized
    if align_to_month_start:
        ts_lr["ds"] = (
            pd.to_datetime(ts_lr["ds"])
            .dt.to_period("M")
            .dt.to_timestamp(how="start")
            .dt.normalize()
        )

    return df_model, ts_lr, exog_cols