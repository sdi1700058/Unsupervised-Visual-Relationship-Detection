"""NPZ cache helpers per SPEC §I6 / §V7-V9.

Cache key = (modality, dataset, category, fps).
Cache root = data/npz/<modality>/<dataset>/<category>-<fps>fps.npz.
Per-category only (V9) — all-cat npz NOT cached.
"""

import os
import json

from latplan.util.paths import DATA_DIR


def npz_cache_path(modality, dataset, category, fps):
    """Return absolute path for the per-category video cache file.

    Returns None when caching is disabled for this key (V9: all-cat → None).
    """
    if category is None:
        return None
    return os.path.join(DATA_DIR, "npz", modality, dataset, f"{category}-{fps}fps.npz")


def load_cached(path):
    """Load cached arrays. Returns (images, bboxes, names, frame_ids, meta) or None on miss."""
    if path is None or not os.path.exists(path):
        return None
    import numpy as np
    with np.load(path, allow_pickle=True) as data:
        images    = data["images"]
        bboxes    = data["bboxes"]
        names     = data["names"].tolist()
        frame_ids = data["frame_ids"].tolist()
        meta_raw  = data["meta"].item() if "meta" in data.files else b"{}"
        meta      = json.loads(meta_raw.decode("utf-8") if isinstance(meta_raw, bytes) else meta_raw)
    return images, bboxes, names, frame_ids, meta


def save_cache(path, images, bboxes, names, frame_ids, meta):
    """Persist cached arrays + meta json blob (V8: raw arrays only, no one-hot)."""
    if path is None:
        return
    import numpy as np
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        images=images,
        bboxes=bboxes,
        names=np.array(names, dtype=object),
        frame_ids=np.array(frame_ids, dtype=object),
        meta=json.dumps(meta).encode("utf-8"),
    )
