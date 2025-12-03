"""
src/utils.py
Essential utilities: ensure_dir, set_seed, save_yaml, save_json, get_logger
"""

from pathlib import Path
import random
import json
import logging
from typing import Any, Union, Optional

# Optional imports (handled gracefully)
try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None

try:
    import yaml
except Exception:
    yaml = None


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist. Returns Path object."""
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 42, cuda_deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): seed value
        cuda_deterministic (bool): set deterministic CuDNN behavior (may slow training)
    """
    random.seed(seed)
    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass

    if torch is not None:
        try:
            torch.manual_seed(seed)
            if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if cuda_deterministic and hasattr(torch.backends, "cudnn"):
                try:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                except Exception:
                    pass
        except Exception:
            pass


def save_yaml(obj: Any, path: Union[str, Path]) -> None:
    """
    Save a Python object to YAML. Uses PyYAML safe_dump.

    Args:
        obj: Python object (dict/list/primitive)
        path: target file path (including .yaml)
    """
    if yaml is None:
        raise ImportError("PyYAML is required to save YAML files. Install with `pip install pyyaml`.")
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save a Python object to JSON.

    Args:
        obj: Python object (dict/list/primitive)
        path: target file path (including .json)
    """
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def get_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_to_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """
    Return a configured logger.

    Args:
        name: logger name
        level: logging level
        log_to_file: optional path to write file logs
    """
    logger = logging.getLogger(name or "deepfake")
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(name)s — %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_to_file:
        file_path = Path(log_to_file)
        ensure_dir(file_path.parent)
        fh = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
