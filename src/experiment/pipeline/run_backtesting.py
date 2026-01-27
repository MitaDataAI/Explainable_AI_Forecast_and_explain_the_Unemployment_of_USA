from mlforecast.utils import PredictionIntervals


def run_backtesting(mlf, ts):
    # ----------------------------------
    # Minimal backtesting WITH PI
    # ----------------------------------
    h = 1
    step_size = 1
    n_windows = 2          # ⚠️ OBLIGATOIRE pour les conformal intervals
    method = "conformal_distribution"
    levels = [95]

    pi = PredictionIntervals(
        h=h,
        n_windows=n_windows,
        method=method,
    )

    bkt_df = mlf.cross_validation(
        df=ts,
        h=h,
        step_size=step_size,
        n_windows=n_windows,
        prediction_intervals=pi,
        level=levels,
        fitted=True,
    )

    return bkt_df