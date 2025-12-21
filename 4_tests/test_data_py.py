# 4_tests/test_data_py.py

from pathlib import Path
import pandas as pd
import pytest

from explainable_ai_forecast.experiments.data import ensure_ms_index, load_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "1_data" / "processed" / "features" / "unemployment_features_stationary.csv"


def test_load_features_file_exists():
    assert DATA_PATH.exists(), f"Fichier introuvable: {DATA_PATH}"


def test_load_features_returns_ms_index_sorted():
    df, info = load_features(str(DATA_PATH), date_col="date")

    # info utile
    assert info.n_rows == df.shape[0]
    assert info.n_cols == df.shape[1]

    # index datetime
    assert isinstance(df.index, pd.DatetimeIndex)

    # tri croissant
    assert df.index.is_monotonic_increasing

    # début de mois (day=1)
    assert (df.index.day == 1).all()

    # freq MS si pandas arrive à l'inférer après asfreq
    assert df.index.freqstr == "MS"


def test_ensure_ms_index_raises_on_duplicate_dates():
    # deux lignes sur le même mois
    raw = pd.DataFrame(
        {
            "date": ["2020-01-15", "2020-01-20"],
            "x": [1, 2],
        }
    )
    with pytest.raises(ValueError, match="doublons"):
        ensure_ms_index(raw, date_col="date")


def test_ensure_ms_index_raises_on_unparseable_date():
    raw = pd.DataFrame(
        {
            "date": ["2020-01-01", "NOT_A_DATE"],
            "x": [1, 2],
        }
    )
    with pytest.raises(ValueError, match="non parseables"):
        ensure_ms_index(raw, date_col="date")


def test_ensure_ms_index_inserts_missing_months_as_nan():
    raw = pd.DataFrame(
        {
            "date": ["2020-01-15", "2020-03-10"],  # février manquant
            "x": [10.0, 30.0],
        }
    )
    df = ensure_ms_index(raw, date_col="date")

    # Après asfreq MS : Jan, Feb, Mar
    assert df.shape[0] == 3
    assert df.index[0] == pd.Timestamp("2020-01-01")
    assert df.index[1] == pd.Timestamp("2020-02-01")
    assert df.index[2] == pd.Timestamp("2020-03-01")

    # Février doit être NaN
    assert pd.isna(df.loc["2020-02-01", "x"])