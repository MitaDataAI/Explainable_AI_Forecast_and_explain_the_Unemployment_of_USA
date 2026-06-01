# 2_data_processing/ETL/2_build_mapping.py

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from configs.ETL import ETLConfig

CFG = ETLConfig()

DATASET_PATH = PROJECT_ROOT / "1_data/raw/dataset_fred.csv"
MAPPING_PATH = PROJECT_ROOT / "1_data/processed/fred_md_to_fred_mapping.csv"


def canonical_name(col: str) -> str:
    c = str(col).strip().upper()

    if c in {"S&P 500", "S&P500", "SP500", "P500"}:
        return "SP500"

    if c in {"OILPRICEX", "OILPRICEX", "OILPRICE", "DCOILWTICO"}:
        return "OILPRICEX"

    if c in {"DATE", "SASDATE"}:
        return "DATE"

    return c


def main():
    print(f"Dataset: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError("dataset_fred.csv introuvable")

    df = pd.read_csv(DATASET_PATH, nrows=5)

    cols = list(df.columns)

    rows = []

    for col in cols:
        canonical = canonical_name(col)

        rows.append(
            {
                "raw_name": col,
                "normalized_name": str(col).strip().upper(),
                "canonical_name": canonical,
                "status": "OK" if canonical != "DATE" else "SKIP",
            }
        )

        print(f"{col} -> {canonical}")

    out = pd.DataFrame(rows)

    MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(MAPPING_PATH, index=False)

    print(f"\n✅ Mapping sauvegardé: {MAPPING_PATH}")


if __name__ == "__main__":
    main()