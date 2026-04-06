import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


class DotDict(dict):
    def __getattr__(self, item):
        value = self[item]
        if isinstance(value, dict):
            return DotDict(value)
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _to_dotdict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return DotDict({k: _to_dotdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_dotdict(v) for v in obj]
    return obj


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config() -> DotDict:
    parser = argparse.ArgumentParser(description="SelectQ reproduction")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg["config_path"] = str(Path(args.config).resolve())
    return _to_dotdict(cfg)
