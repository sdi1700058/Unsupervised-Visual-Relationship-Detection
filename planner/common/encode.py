#!/usr/bin/env python3
"""planner/common/encode.py — shared model load + frame encode.

Loads a trained FirstOrderSAE from `model_dir`, loads the npz the model was
trained on (via `loaded_videos.json` manifest OR direct `--npz-path`), and
returns encoded latents for arbitrary frame indices.

Used by export_latents.py.
"""

import json
import os
from pathlib import Path


def load_model(model_dir):
    """Load a trained FirstOrderSAE.

    Returns
    -------
    ae : model.model.FirstOrderSAE
        Loaded model with weights. `ae.parameters` holds U/A/P/etc.
    """
    import model.model
    ae = model.model.load(str(model_dir), allow_failure=False)
    return ae


def load_npz_states(model_dir, npz_path=None):
    """Return the (N, num_objs, feat_dim) states array + auxiliary metadata.

    Loads via the same pipeline as `model/train/extract_fol.py::load_vidvrd_data` /
    `load_actiongenome_data` — auto-detects domain from `loaded_videos.json`
    manifest.

    Returns
    -------
    states       : np.ndarray (N, num_objs, feat_dim) float32, model-ready.
    bboxes       : np.ndarray (N, num_objs, 4) uint16 or float, raw pixel coords
                   BEFORE onehot conversion. `None` for puzzle domains.
    per_state_names : list[list[str]] object names per state.
    frame_ids    : list[str] frame identifiers.
    """
    import numpy as np
    from model.util.cache import load_cached

    model_dir = Path(model_dir)
    manifest_path = model_dir / "loaded_videos.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no loaded_videos.json at {manifest_path}. "
            "For puzzle/blocksworld domains, pass --npz-path explicitly."
        )
    with manifest_path.open() as f:
        manifest = json.load(f)

    if npz_path is None:
        npz_path = manifest.get("npz_path")
        if not npz_path:
            raise ValueError(
                f"manifest at {manifest_path} has no 'npz_path'. "
                "Pass --npz-path explicitly."
            )
    npz_path = str(npz_path)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"npz not found: {npz_path}")

    hit = load_cached(npz_path)
    if hit is None:
        raise RuntimeError(f"load_cached returned None for {npz_path}")
    # hit = (images, bboxes, per_state_names, frame_ids, meta)
    images, bboxes_raw, per_state_names, frame_ids, meta = hit

    # Detect domain from manifest to apply the correct preprocessing
    # (mirrors load_vidvrd_data / load_actiongenome_data in model/train/extract_fol.py).
    from model.puzzles.util import preprocess
    from model.train.strips import bboxes_to_onehot

    # PICSIZE differs per loader — pull from puzzle_labeled_objects (canonical).
    from model.puzzles.puzzle_labeled_objects import PICSIZE

    picsize_grid = (np.array(PICSIZE) // 5).astype(int)
    Y, X = picsize_grid[0], picsize_grid[1]
    num_states, num_objs = images.shape[0], images.shape[1]

    images_f = images.astype(np.float32) / 256
    images_f = preprocess(images_f)
    bboxes_onehot = bboxes_to_onehot(bboxes_raw, X, Y)
    states = np.concatenate(
        (images_f.reshape((num_states, num_objs, -1)),
         bboxes_onehot.reshape((num_states, num_objs, -1))),
        axis=-1,
    ).astype(np.float32)

    return states, bboxes_raw, per_state_names, frame_ids


def encode_frame(ae, states, frame_idx):
    """Encode a single frame to its binary latent code.

    Parameters
    ----------
    ae         : loaded FirstOrderSAE
    states     : (N, num_objs, feat_dim) — output of `load_npz_states`
    frame_idx  : int (negative indices allowed)

    Returns
    -------
    z : np.ndarray (U*P,) int  — binary latent (rounded)
    """
    import numpy as np
    N = states.shape[0]
    if frame_idx < 0:
        frame_idx = N + frame_idx
    if not (0 <= frame_idx < N):
        raise IndexError(f"frame_idx {frame_idx} out of range [0, {N})")
    x = states[frame_idx:frame_idx + 1]         # (1, num_objs, feat_dim)
    z_continuous = ae.encode(x)                  # (1, U*P) float
    z = np.asarray(z_continuous).round().astype(np.int8).reshape(-1)
    return z


def apply_in_batches(fn, states, batch=None):
    """`fn` over `states`, `batch` rows at a time, concatenated.

    The peak memory of one call is what this controls and nothing else: the
    rows are independent, so the result equals `fn(states)` exactly.
    """
    import numpy as np
    states = np.asarray(states)
    if not batch or len(states) <= batch:
        return np.asarray(fn(states))
    return np.concatenate([np.asarray(fn(states[i:i + batch]))
                           for i in range(0, len(states), batch)])


def encode_all(ae, states, batch=None):
    """Every state encoded, as an (N, U*P) int8 array.

    `batch` encodes that many states per call to the model. Without it the
    whole array goes in one call, which is what this did until 2026-10-11:
    `export_latents.py` on the 14,314 states of the KIT role clips was
    killed at 8 GB on 2026-10-10, so nothing could be exported from a real
    clip set. Batching changes no value, only the peak memory.
    """
    import numpy as np
    return apply_in_batches(ae.encode, states, batch).round().astype(
        np.int8)
