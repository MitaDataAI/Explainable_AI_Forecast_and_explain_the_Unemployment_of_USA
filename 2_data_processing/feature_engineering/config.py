"""
Config "feature_engineering" : paramètres figés issus des notebooks.

Ici on fige les règles de stationnarisation :
- method: diff | logdiff | none
- lags: horizon de différenciation (Δh)
- order: 1 (I(1)) ou 2 (I(2))
"""

# Colonnes / séries attendues (utilisé aussi pour select_series_long)
SERIES_COLS = [
    "UNRATE",
    "TB3MS",
    "RPI",
    "INDPRO",
    "DPCERA3M086SBEA",
    "SP500",
    "BUSLOANS",
    "CPIAUCSL",
    "OILPRICEX",
    "M2SL",
    "USREC",
]

# Nom du fichier de sortie
OUTPUT_FILENAME = "unemployment_features_stationary.csv"

# Règles de stationnarisation (figées)
STATIONARITY_RULES = {
    # règles spécifiques inchangées
    "UNRATE": {"method": "diff", "lags": 12, "order": 1},  # chômage Δ12
    "TB3MS": {"method": "diff", "lags": 3, "order": 1},    # T-bill Δ3

    # 2e différenciation du log pour ces séries
    "OILPRICEX": {"method": "logdiff", "lags": 3, "order": 2},
    "BUSLOANS": {"method": "logdiff", "lags": 3, "order": 2},
    "M2SL": {"method": "logdiff", "lags": 3, "order": 2},
    "CPIAUCSL": {"method": "logdiff", "lags": 3, "order": 2},

    # Binaire
    "USREC": {"method": "none", "lags": 0, "order": 0},

    # règle par défaut (toutes les autres séries)
    "_default": {"method": "logdiff", "lags": 3, "order": 1},
}

# Libellés métier (pour affichage / reporting)
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