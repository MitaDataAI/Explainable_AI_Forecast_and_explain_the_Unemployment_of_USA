import pandas as pd
import pytest

from explainable_ai_forecast.experiments.features import make_supervised


def test_make_supervised_shift_and_lag():
    df = pd.DataFrame(
        {
            "UNRATE": [1, 2, 3, 4, 5],
            "x1": [10, 11, 12, 13, 14],
        },
        index=pd.date_range("2020-01-01", periods=5, freq="MS"),
    )

    out = make_supervised(df, target="UNRATE", horizon=2, add_target_lags=[2])

    # y_future(t) = y(t+2)
    assert out.y_future.loc["2020-01-01"] == 3
    assert out.y_future.loc["2020-02-01"] == 4

    # UNRATE_lag2(t) = y(t-2)
    assert pd.isna(out.X.loc["2020-01-01", "UNRATE_lag2"])
    assert pd.isna(out.X.loc["2020-02-01", "UNRATE_lag2"])
    assert out.X.loc["2020-03-01", "UNRATE_lag2"] == 1

    # X ne contient pas la cible brute
    assert "UNRATE" not in out.X.columns


def test_make_supervised_raises_if_target_missing():
    df = pd.DataFrame({"x1": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="MS"))
    with pytest.raises(KeyError):
        make_supervised(df, target="UNRATE", horizon=1)


def test_make_supervised_invalid_horizon():
    df = pd.DataFrame({"UNRATE": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="MS"))
    with pytest.raises(ValueError):
        make_supervised(df, target="UNRATE", horizon=0)
