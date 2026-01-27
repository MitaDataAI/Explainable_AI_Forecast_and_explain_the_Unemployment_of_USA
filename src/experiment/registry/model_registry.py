# ----------------------------
# Model registry (Notebook)
# ----------------------------

from sklearn.linear_model import LinearRegression, Ridge
from statsforecast.models import AutoRegressive


# -------------------------
# MLForecast models only
# -------------------------
MLFORECAST_MODELS = {
    "linear": LinearRegression,
    "ridge": Ridge,
}


# -------------------------
# StatsForecast models only
# (pas utilisé encore)
# -------------------------
STATSFORECAST_MODELS = {
    "ar": AutoRegressive,
}