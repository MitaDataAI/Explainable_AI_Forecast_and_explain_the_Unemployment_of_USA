from pathlib import Path
import pandas as pd

from explainable_ai_forecast.experiments.model_evaluation_error_analysis import (
    make_mae_dm_pivot,
)

# ============================================================
# Configuration
# ============================================================

COMPARE_DIR = Path("artifacts/experiments/comparison/8a88d7fbc4d5")

LABEL_MAP = {
    "AR(1)": "AR1",
    "AR(p auto)": "ARP",
    "linear": "LINREG",
}

segments = [
    ("1990-01-01", "1999-12-31", "1990-1999"),
    ("2000-01-01", "2008-08-31", "2000-08/2008"),
    ("2008-09-01", "2019-11-30", "09/2008-11/2019"),
]

# ============================================================
# Chargement des prédictions (format long)
# ============================================================

df_long = pd.read_csv(COMPARE_DIR / "predictions_long.csv")

required_cols = {"date", "method", "y_pred", "y_true"}
missing = required_cols - set(df_long.columns)
if missing:
    raise ValueError(f"predictions_long.csv missing columns: {missing}")

df_long["date"] = pd.to_datetime(df_long["date"], errors="coerce")
df_long = df_long.dropna(subset=["date"])

if df_long.duplicated(subset=["date", "method"]).any():
    df_long = (
        df_long
        .groupby(["date", "method"], as_index=False)
        .agg(
            y_pred=("y_pred", "mean"),
            y_true=("y_true", "mean"),
        )
    )

# ============================================================
# Long -> Wide
# ============================================================

wide = (
    df_long
    .pivot(index="date", columns="method", values="y_pred")
    .join(df_long.groupby("date")["y_true"].mean().rename("true"))
)

wide = wide.rename(columns=LABEL_MAP).reset_index()

# ============================================================
# MAE + Diebold–Mariano
# ============================================================

table = make_mae_dm_pivot(
    df_wide=wide,
    periods=segments,
    include_overall=True,
    overall_label="Ensemble",
    min_obs=20,
    round_digits=4,
    dm_lags=11,
)

ORDER = ["AR1", "ARP", "LINREG"]
table = table.reindex(index=ORDER)

EXPECTED_COLS = [
    "Ensemble",
    "1990-1999",
    "2000-08/2008",
    "09/2008-11/2019",
]
table = table.reindex(columns=EXPECTED_COLS)

# ============================================================
# Sauvegarde
# ============================================================

out_csv = COMPARE_DIR / "error_analysis_mae_dm.csv"
table.to_csv(out_csv)

print(table)
print(f"\nSaved: {out_csv}")