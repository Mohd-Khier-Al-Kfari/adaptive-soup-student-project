"""Minimal loader for prepared HAR experiments. Author: Moh'd Khier Al Kfari.

Examples:
    from load_data import load_experiment
    data = load_experiment("cross_age")
    x_train = data["RealWorld_below30"]["features"]["train"]
"""
from pathlib import Path
import json
import pickle

PREPARED = Path(__file__).resolve().parents[2] / "data" / "prepared"

def available_experiments():
    """Return only experiment folders that contain prepared data."""
    result = {}
    for experiment in ("cross_dataset", "cross_gender", "cross_age"):
        root = PREPARED / experiment
        if not root.is_dir():
            continue
        entries = []
        for dataset_folder in root.iterdir():
            if not dataset_folder.is_dir() or dataset_folder.name == "metadata":
                continue
            if experiment == "cross_dataset" and (dataset_folder / "features.pkl").is_file():
                entries.append(dataset_folder.name)
                continue
            for group_folder in dataset_folder.iterdir():
                if group_folder.is_dir() and (group_folder / "features.pkl").is_file():
                    entries.append(f"{dataset_folder.name}/{group_folder.name}")
        if entries:
            result[experiment] = sorted(entries)
    return result

def _load_one(experiment, dataset, group):
    """Load one prepared dataset/group without copying arrays in memory."""
    if experiment not in ("cross_dataset", "cross_gender", "cross_age"):
        raise ValueError("experiment must be cross_dataset, cross_gender, or cross_age")
    folder = PREPARED / experiment / dataset / (group or "")
    features_path = folder / "features.pkl"
    labels_path = folder / "evaluation_labels.pkl"
    if not features_path.is_file() or not labels_path.is_file():
        available = available_experiments()
        raise FileNotFoundError(f"Prepared experiment not found: {folder}. Available: {available}")
    with features_path.open("rb") as handle:
        features = pickle.load(handle)
    with labels_path.open("rb") as handle:
        labels = pickle.load(handle)
    metadata_path = PREPARED / experiment / "metadata" / f"{dataset}{('_' + group) if group else ''}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else None
    return {"features": features, "labels": labels, "metadata": metadata}

def load_experiment(experiment):
    """Load every prepared dataset/group belonging to one experiment.

    Returns a dictionary keyed by names such as ``MHEALTH`` or
    ``RealWorld_below30``. Each value has ``features``, ``labels`` and
    ``metadata``; each features object contains ``train``, ``val`` and ``test``.
    """
    options = available_experiments()
    if experiment not in options:
        raise ValueError(f"Unknown or empty experiment: {experiment}. Available: {options}")
    loaded = {}
    for entry in options[experiment]:
        if "/" in entry:
            dataset, group = entry.split("/", 1)
        else:
            dataset, group = entry, None
        loaded[dataset if group is None else f"{dataset}_{group}"] = _load_one(experiment, dataset, group)
    return loaded

def choose_and_load():
    """Small interactive entry point for students."""
    options = [name for name in ("cross_dataset", "cross_gender", "cross_age")
               if name in available_experiments()]
    if not options:
        raise RuntimeError(
            "No prepared experiments are available. Run 'python run_pipeline.py' "
            "from the repository root and enable preprocessing and postprocessing first."
        )
    for number, name in enumerate(options, 1):
        print(f"{number}. {name}")
    while True:
        answer = input("Select experiment by number or name: ").strip().lower()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            answer = options[int(answer) - 1]
        if answer in options:
            break
        print("Invalid selection. Please enter a listed number or experiment name.")
    return load_experiment(answer)

if __name__ == "__main__":
    data = choose_and_load()
    for name, item in data.items():
        print(f"Loaded {name}: {list(item['features'])}")
