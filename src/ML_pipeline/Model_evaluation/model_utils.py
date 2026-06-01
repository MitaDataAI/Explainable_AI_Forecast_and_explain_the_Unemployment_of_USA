import numpy as np
import pandas as pd

# =========================================================
# (1) Helpers
# =========================================================
def _unwrap_estimator_from_mlf(maybe_mlf, preferred_key=None):
    if hasattr(maybe_mlf, "models_") and isinstance(getattr(maybe_mlf, "models_"), dict):
        d = maybe_mlf.models_
        if preferred_key is not None and preferred_key in d:
            return d[preferred_key]
        return next(iter(d.values()))
    return maybe_mlf


def _predict_any(model_obj, X_feat: pd.DataFrame, *, model_key=None) -> np.ndarray:
    est = _unwrap_estimator_from_mlf(model_obj, preferred_key=model_key)

    drop_cols = [c for c in ["ds", "unique_id", "y"] if c in X_feat.columns]
    X = X_feat.drop(columns=drop_cols, errors="ignore")

    fn = getattr(est, "feature_names_in_", None)
    if fn is not None:
        fn = [c for c in fn if c in X.columns]
        if len(fn) > 0:
            X = X[fn]

    return np.asarray(est.predict(X), float)
