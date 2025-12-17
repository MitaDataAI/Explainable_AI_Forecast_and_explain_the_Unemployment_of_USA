"""
Config "feature_engineering" : paramètres figés issus des notebooks.

Ici on fige les règles de stationnarisation :
- transform: diff | logdiff | none
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

# Règles de stationnarisation (décidées dans notebook)
STATIONARITY_RULES = {
    # Target (variation en points)
    "UNRATE": {"transform": "diff", "order": 1},

    # Taux
    "TB3MS": {"transform": "diff", "order": 1},

    # I(1) log-diff
    "INDPRO": {"transform": "logdiff", "order": 1},
    "RPI": {"transform": "logdiff", "order": 1},
    "SP500": {"transform": "logdiff", "order": 1},
    "DPCERA3M086SBEA": {"transform": "logdiff", "order": 1},

    # I(2) diff²
    "BUSLOANS": {"transform": "diff", "order": 2},
    "CPIAUCSL": {"transform": "diff", "order": 2},
    "M2SL": {"transform": "diff", "order": 2},
    "OILPRICEX": {"transform": "diff", "order": 2},

    # Binaire
    "USREC": {"transform": "none", "order": 0},
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