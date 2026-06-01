
import numpy as np

# =========================================================
# (0) Métrique deviance = MSE
# =========================================================
def mse_deviance(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean((y_true[m] - y_pred[m]) ** 2)) if np.any(m) else np.nan