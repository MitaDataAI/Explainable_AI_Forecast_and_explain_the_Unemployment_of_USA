import pandas as pd
from sklearn.linear_model import LinearRegression

from explainable_ai_forecast.experiments.backtest import pseudo_oos_expanding


def test_pseudo_oos_expanding_runs_and_returns_predictions():
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    X = pd.DataFrame({"x1": range(24)}, index=idx).astype(float)
    y_future = pd.Series(range(24), index=idx).astype(float)

    # introduire des NaN en fin (comme un shift horizon)
    y_future.iloc[-3:] = float("nan")

    res = pseudo_oos_expanding(
        X,
        y_future,
        model=LinearRegression(),
        min_train_n=6,
        start="2020-06-01",
        end="2021-06-01",
    )

    assert not res.predictions.empty
    assert {"y_true", "y_pred"} <= set(res.predictions.columns)
    assert res.predictions.index.is_monotonic_increasing