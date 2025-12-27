import numpy as np
import pandas as pd

from explainable_ai_forecast.experiments.data_preparation_preprocess import (
    PreprocSpec,
    fit_preproc,
    apply_preproc,
)

def test_preproc_winsor_clips_outliers():
    X = pd.DataFrame({"a": [0, 1, 2, 100], "b": [10, 11, 12, 13]}, dtype=float)
    spec = PreprocSpec(winsor_level=0.25, normalize=False)  # quantiles agressifs pour test
    fitted = fit_preproc(X, spec)

    X2 = apply_preproc(X, fitted)
    # le max de a doit être clip (moins que 100)
    assert X2["a"].max() < 100


def test_preproc_normalize_center_scale():
    X = pd.DataFrame({"a": [1, 2, 3, 4]}, dtype=float)
    spec = PreprocSpec(winsor_level=0.0, normalize=True)
    fitted = fit_preproc(X, spec)

    X2 = apply_preproc(X, fitted)
    # mean ~ 0
    assert abs(float(X2["a"].mean())) < 1e-9
    # std ~ 1 (ddof=0)
    assert abs(float(X2["a"].std(ddof=0)) - 1.0) < 1e-9


def test_apply_preproc_fillna():
    X = pd.DataFrame({"a": [np.nan, 1.0]}, dtype=float)
    spec = PreprocSpec(winsor_level=0.0, normalize=False)
    fitted = fit_preproc(pd.DataFrame({"a": [0.0, 1.0]}, dtype=float), spec)

    X2 = apply_preproc(X, fitted, fillna_value=0.0)
    assert float(X2["a"].iloc[0]) == 0.0