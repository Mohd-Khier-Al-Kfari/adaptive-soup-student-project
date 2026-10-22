"""Plot a selected prepared experiment. Author: Moh'd Khier Al Kfari."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "data_access"))
from load_data import choose_and_load

if __name__ == "__main__":
    try:
        data = choose_and_load()
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Plot loader error: {error}")
    try:
        from plot_examples import plot_selected_experiment
    except ModuleNotFoundError as error:
        if error.name == "matplotlib":
            raise SystemExit(
                "Plot dependency error: matplotlib is not installed. "
                "Install the repository environment or run 'python -m pip install -r requirements.txt'."
            )
        raise
    first = next(iter(data.values()))
    experiment = (first.get("metadata") or {}).get("experiment", "cross_dataset")
    for path in plot_selected_experiment(data, experiment):
        print(f"Saved plot: {path}")
