# =========================================================
# Standard library
# =========================================================
from pathlib import Path
from typing import List, Tuple

# =========================================================
# Data
# =========================================================
import pandas as pd

# =========================================================
# Feature Store (Feast)
# =========================================================
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


def load_wide_from_feast(
    feature_ref: str,
    series_ids: list[str],
    start: str = "1960-01-01",
    end: str = "2025-08-01",
    freq: str = "MS",
    project_marker: str = "2_data_processing",
    expected_value_col: str | None = None,
    *,
    ensure_ms: bool = True,
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
    ensure_ms : bool
        If True, force index to Month Start ("MS") timestamps.
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

    # Basic sanity checks
    required = {"series_id", "date"}
    missing = required - set(df_long.columns)
    if missing:
        raise KeyError(f"Missing columns from Feast result: {missing}. Got: {list(df_long.columns)}")

    # Identify value column produced by Feast
    if expected_value_col is not None:
        value_col = expected_value_col
        if value_col not in df_long.columns:
            raise KeyError(
                f"expected_value_col='{value_col}' not found. "
                f"Available columns: {list(df_long.columns)}"
            )
    else:
        # Case A: Feast returns directly "value"
        if "value" in df_long.columns:
            value_col = "value"
        else:
            # Case B: "<feature_view>__<feature_name>"
            candidates = [c for c in df_long.columns if "__" in c and c not in ("series_id", "date")]
            if not candidates:
                raise KeyError(
                    "No Feast value column found. "
                    f"Available columns: {list(df_long.columns)}"
                )

            if len(candidates) > 1 and ":" in feature_ref:
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

    # Ensure datetime index + (option) MS alignment
    idx = pd.to_datetime(df_wide.index)
    if ensure_ms:
        idx = idx.to_period("M").to_timestamp(how="start").normalize()
    df_wide.index = idx
    df_wide.index.name = "date"

    return df_wide


def make_ts_from_wide(
    df_wide: pd.DataFrame,
    target_col: str = "UNRATE",
    lags: int | list[int] = 12,
    *,
    unique_id_value: str | None = None,
    date_name: str = "ds",
    target_name: str = "y",
    drop_original_exog: bool = True,
    dropna: bool = True,
    include_target_lags: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a MLForecast-style dataframe from a wide dataframe:
    [unique_id, ds, y, exog...] then apply exogenous lags.
    """
    if target_col not in df_wide.columns:
        raise KeyError(
            f"target_col='{target_col}' not found in columns: {list(df_wide.columns)}"
        )

    if isinstance(lags, int):
        lags = [lags]

    df = df_wide.sort_index().copy()

    if unique_id_value is None:
        unique_id_value = target_col

    # -----------------------------
    # Include or exclude target lags
    # -----------------------------
    if include_target_lags:
        exog_cols = list(df.columns)
    else:
        exog_cols = [c for c in df.columns if c != target_col]

    ts_df = df.reset_index().rename(columns={"date": date_name})
    ts_df[target_name] = ts_df[target_col]
    ts_df["unique_id"] = unique_id_value

    ts_df = ts_df[["unique_id", date_name, target_name] + exog_cols].copy()
    ts_df = ts_df.sort_values(["unique_id", date_name]).copy()

    exog_cols_lagged = []
    for c in exog_cols:
        for lag in lags:
            new_c = f"{c}_lag{lag}"
            ts_df[new_c] = ts_df.groupby("unique_id")[c].shift(lag)
            exog_cols_lagged.append(new_c)

    if drop_original_exog:
        ts_df = ts_df.drop(columns=exog_cols, errors="ignore")

    if dropna:
        ts_df = ts_df.dropna(
            subset=[target_name] + exog_cols_lagged
        ).reset_index(drop=True)

    return ts_df, exog_cols_lagged
