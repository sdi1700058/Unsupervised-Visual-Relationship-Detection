"""Shared path utilities for the FOSAE project."""

import os

# Resolve project root relative to this file: latplan/util/paths.py -> project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR  = os.path.join(PROJECT_ROOT, "out")


def find_dataset(filename):
    """Look for dataset in data/ first, fall back to latplan/puzzles/.

    Args:
        filename: e.g. "puzzle-mnist-3-3.npz" or "blocks-5-3.npz"

    Returns:
        Absolute path to the dataset file.
    """
    import latplan
    p = os.path.join(DATA_DIR, filename)
    if os.path.exists(p):
        return p
    return os.path.join(latplan.__path__[0], "puzzles", filename)
