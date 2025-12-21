import pandas as pd

from explainable_ai_forecast.experiments.metrics import regression_metrics


def test_regression_metrics_basic():
    preds = pd.DataFrame(
        {"y_true": [1.0, 2.0, 3.0], "y_pred": [1.0, 2.5, 2.0]},
        index=pd.date_range("2020-01-01", periods=3, freq="MS"),
    )
    m = regression_metrics(preds)
    assert "rmse" in m and "mae" in m and "r2" in m and "n" in m
    assert m["n"] == 3