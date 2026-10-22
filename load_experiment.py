"""Load a prepared experiment. Author: Moh'd Khier Al Kfari."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "data_access"))
from load_data import choose_and_load

if __name__ == "__main__":
    try:
        data = choose_and_load()
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Data loader error: {error}")
    for name, item in data.items():
        print(f"Loaded {name}: {list(item['features'])}")
