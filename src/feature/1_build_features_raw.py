# --------------------------------------------------
# Setup PYTHONPATH (AVANT les imports du projet)
# --------------------------------------------------
import sys
from pathlib import Path

# Trouver la racine projet en remontant jusqu'à trouver le dossier "configs"
PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "configs").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

if not (PROJECT_ROOT / "configs").exists():
    
    raise RuntimeError("Impossible de trouver la racine projet (dossier 'configs' introuvable).")

sys.path.append(str(PROJECT_ROOT))

# (optionnel debug)
print("PROJECT_ROOT =", PROJECT_ROOT)
print("EXISTS configs? =", (PROJECT_ROOT / "configs").exists())

# --------------------------------------------------
# Imports projet
# --------------------------------------------------
from feature_core import (
    load_from_postgres,
    select_series_long,
    summarize_long,
    save_long_to_parquet,   # ✅ parquet
)

from configs.feature_engineering import (
    SERIES_COLS,
    RAW_LONG_FILENAME,      # ex: "unemployment_features_raw_long.csv"
)


def main():
    # 1) Chargement des données brutes (LONG)
    df_long = load_from_postgres()

    # 2) Sélection des séries utiles (piloté par config)
    df_sel = select_series_long(df_long, SERIES_COLS)

    # 3) Vérification (LONG)
    summarize_long(df_sel)

    # 4) Sauvegarde (LONG) -> PARQUET
    parquet_filename = Path(RAW_LONG_FILENAME).with_suffix(".parquet").name

    output_path = save_long_to_parquet(
        df_long=df_sel,
        script_file=__file__,
        filename=parquet_filename,
    )

    print(f"\nFeatures RAW (LONG) sauvegardées ici :\n{output_path}")


if __name__ == "__main__":
    main()