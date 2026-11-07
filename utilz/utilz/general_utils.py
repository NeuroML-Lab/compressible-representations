import numpy as np
import torch
import random
from pathlib import Path
import json
from typing import List


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def metric_key(metric: str) -> str:
    if metric not in {"top1", "top5"}:
        raise ValueError(f"Unknown metric {metric}")
    return metric

# write json
def safe_json(obj: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_device(device_id: int) -> torch.device:
    return torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    

def choose_indices(n_total: int, n_select: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    n = min(n_total, n_select)
    return rng.choice(n_total, size=n, replace=False).tolist()


