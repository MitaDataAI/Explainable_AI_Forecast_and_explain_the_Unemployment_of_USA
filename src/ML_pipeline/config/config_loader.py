import json
from pathlib import Path

def load_config(path: str | Path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        return json.load(f)