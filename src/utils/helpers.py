# ==============================================================================
# src/utils/helpers.py
# General utility functions: seeding, timing, file I/O helpers.
# ==============================================================================

import json
import os
import time
import random
from contextlib import contextmanager
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and TensorFlow.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")


@contextmanager
def Timer(description: str = "Operation"):
    """
    Context manager that measures and logs elapsed time.

    Usage:
        with Timer("Training MobileNetV2"):
            model.fit(...)

    Args:
        description: Label for the timed operation.
    """
    start = time.time()
    logger.info(f"⏱  Starting: {description}")
    yield
    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    logger.info(f"✅ Completed: {description} in {minutes}m {seconds:.1f}s")


def get_elapsed_time(start_time: float) -> float:
    """Return elapsed time in seconds since start_time."""
    return time.time() - start_time


def save_json(data: Any, filepath: str) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Serializable data to save.
        filepath: Output file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.debug(f"JSON saved: {filepath}")


def load_json(filepath: str) -> Any:
    """Load and return data from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(data: List[Dict], filepath: str) -> None:
    """
    Save a list of dictionaries to a CSV file.

    Args:
        data: List of row dictionaries.
        filepath: Output CSV path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    logger.debug(f"CSV saved: {filepath}")


def get_model_size_mb(model_path: str) -> float:
    """
    Get the file size of a saved model in megabytes.

    Args:
        model_path: Path to the saved model file.

    Returns:
        Model file size in MB.
    """
    if os.path.isfile(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    elif os.path.isdir(model_path):
        total = 0
        for dirpath, _, filenames in os.walk(model_path):
            for f in filenames:
                total += os.path.getsize(os.path.join(dirpath, f))
        return total / (1024 * 1024)
    return 0.0


def format_number(num: int) -> str:
    """Format large numbers with commas: 1234567 -> '1,234,567'."""
    return f"{num:,}"


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist, return the path."""
    os.makedirs(path, exist_ok=True)
    return path
