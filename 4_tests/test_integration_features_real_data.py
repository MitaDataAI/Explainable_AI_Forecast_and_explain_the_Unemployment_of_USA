from pathlib import Path
import pandas as pd

from explainable_ai_forecast.experiments.data import load_features
from explainable_ai_forecast.experiments.features import make_supervised


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "1_data" / "processed" / "features" / "unemployment_features_stationary.csv"


def test_real_data_make_supervised_smoke():
    df, _ = load_features(str(DATA_PATH), date_col="date")

    out = make_supervised(
        df,
        target="UNRATE",
        horizon=12,
        add_target_lags=[12],
    )

    # Colonnes attendues
    assert "UNRATE_lag12" in out.X.columns
    assert "UNRATE" not in out.X.columns

    # Cohérence dimensions (on garde le même index, NaN gérés plus tard)
    assert out.X.shape[0] == df.shape[0]
    assert out.y_future.shape[0] == df.shape[0]

    # NaN attendus:
    # - au début pour le lag (12 premiers mois)
    assert pd.isna(out.X["UNRATE_lag12"].iloc[:12]).all()

    # - à la fin pour y_future (12 derniers mois)
    assert pd.isna(out.y_future.iloc[-12:]).all()