import pandas as pd
from sklearn.linear_model import LinearRegression

from explainable_ai_forecast.experiments.model_training_backtest import (
    pseudo_oos_expanding,
)


def test_pseudo_oos_expanding_runs_and_returns_predictions():
    # Index mensuel
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")

    # Features simples
    X = pd.DataFrame({"x1": range(24)}, index=idx).astype(float)

    # y_future simulé (comme un shift horizon)
    y_future = pd.Series(range(24), index=idx).astype(float)
    y_future.iloc[-3:] = float("nan")  # NaN en fin

    res = pseudo_oos_expanding(
        X=X,
        y_future=y_future,
        model=LinearRegression(),
        min_train_n=6,
        start="2020-06-01",
        end="2021-06-01",
    )

    # --- assertions ---
    assert res is not None
    assert hasattr(res, "predictions")

    assert not res.predictions.empty
    assert {"y_true", "y_pred"}.issubset(res.predictions.columns)

    # index = dates, triées
    assert isinstance(res.predictions.index, pd.DatetimeIndex)
    assert res.predictions.index.is_monotonic_increasing