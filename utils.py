import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_device(eval_on_cpu: bool = False) -> torch.device:
    if eval_on_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_annealed_step(
    t: int,
    tmax: int,
    lambda_min: float,
    lambda_max: float,
) -> float:
    if tmax <= 0:
        return lambda_min
    return lambda_min + 0.5 * (lambda_max - lambda_min) * (1.0 + np.cos(np.pi * t / tmax))
