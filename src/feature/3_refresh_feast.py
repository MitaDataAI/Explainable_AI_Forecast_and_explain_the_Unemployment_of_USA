import sys
from pathlib import Path
import subprocess

# Trouver la racine projet
PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "configs").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

if not (PROJECT_ROOT / "configs").exists():
    raise RuntimeError("Impossible de trouver la racine projet (dossier 'configs' introuvable).")

sys.path.append(str(PROJECT_ROOT))

CURRENT_DIR = Path(__file__).resolve().parent

# Détection automatique des 2 scripts du dossier courant
py_files = [
    p for p in CURRENT_DIR.glob("*.py")
    if p.name != Path(__file__).name
]

RAW_SCRIPT = None
STATIONARY_SCRIPT = None

for p in py_files:
    name = p.name.lower()
    if "raw" in name:
        RAW_SCRIPT = p
    elif "station" in name:
        STATIONARY_SCRIPT = p

if RAW_SCRIPT is None:
    raise FileNotFoundError(
        f"Aucun script RAW trouvé dans {CURRENT_DIR}. "
        f"Fichiers vus: {[p.name for p in py_files]}"
    )

if STATIONARY_SCRIPT is None:
    raise FileNotFoundError(
        f"Aucun script STATIONARY trouvé dans {CURRENT_DIR}. "
        f"Fichiers vus: {[p.name for p in py_files]}"
    )

# Dossier Feast
FEAST_REPO = PROJECT_ROOT / "2_data_processing" / "feature_store" / "feast_repo" / "feature_repo"


def run_python(script_path: Path):
    print(f"\n=== RUN PYTHON: {script_path.name} ===")
    subprocess.run([sys.executable, str(script_path)], check=True)


def run_cmd(cmd: list[str], cwd: Path):
    if not cwd.exists():
        raise FileNotFoundError(f"Dossier introuvable : {cwd}")
    print(f"\n=== RUN CMD: {' '.join(cmd)} ===")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main():
    print("RAW_SCRIPT =", RAW_SCRIPT)
    print("STATIONARY_SCRIPT =", STATIONARY_SCRIPT)
    print("FEAST_REPO =", FEAST_REPO)

    run_python(RAW_SCRIPT)
    run_python(STATIONARY_SCRIPT)
    run_cmd(["feast", "apply"], cwd=FEAST_REPO)

    print("\nFEAST actualisé avec succès.")


if __name__ == "__main__":
    main()