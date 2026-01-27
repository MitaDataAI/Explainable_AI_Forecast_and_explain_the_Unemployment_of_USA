# ============================
# Feature Engineering Config
# ============================

SERIES_COLS = [
    "UNRATE","TB3MS","RPI","INDPRO","DPCERA3M086SBEA",
    "SP500","BUSLOANS","CPIAUCSL","OILPRICEX","M2SL","USREC",
]

# ---------- RAW ----------
RAW_LONG_FILENAME = "unemployment_features_raw_long.csv"

# ---------- STATIONARY ----------
STATIONARY_FILENAME = "unemployment_features_stationary.csv"
REPORT_FILENAME = "stationarity_transformations_report.csv"

STATIONARITY_RULES = {
    "UNRATE": {"method": "diff", "lags": 12, "order": 1},
    "TB3MS": {"method": "diff", "lags": 3, "order": 1},
    "OILPRICEX": {"method": "logdiff", "lags": 3, "order": 2},
    "BUSLOANS": {"method": "logdiff", "lags": 3, "order": 2},
    "M2SL": {"method": "logdiff", "lags": 3, "order": 2},
    "CPIAUCSL": {"method": "logdiff", "lags": 3, "order": 2},
    "USREC": {"method": "none", "lags": 0, "order": 0},
    "_default": {"method": "logdiff", "lags": 3, "order": 1},
}

VARIABLE_LABELS = {
    "UNRATE": "Unemployment",
    "TB3MS": "3-month treasury bill",
    "RPI": "Real personal income",
    "INDPRO": "Industrial production",
    "DPCERA3M086SBEA": "Consumption",
    "SP500": "S&P 500",
    "BUSLOANS": "Business loans",
    "CPIAUCSL": "CPI",
    "OILPRICEX": "Oil price",
    "M2SL": "M2 Money",
    "USREC": "US recession indicator",
}

DROP_FIRST_N = 12