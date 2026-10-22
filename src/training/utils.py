"""Training utilities. Author: Moh'd Khier Al Kfari."""
from __future__ import annotations
import json, os, random
from pathlib import Path
import numpy as np
SEED = 52
def seed_everything(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except ImportError: pass
def json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")
def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
