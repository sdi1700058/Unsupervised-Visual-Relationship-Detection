"""NPZ cache helpers per SPEC §I6 / §V7-V9.

Cache key = (modality, dataset, category, fps, num_objs, patch_size, fill).
Cache root = data/npz/<modality>/<dataset>/<category>-<fps>fps[-mo<n>][-p<s>][-fill].npz.
Per-category only (V9) — all-cat npz NOT cached.

The last three key parts matter because they change the shape or the
content of the cached arrays. A cache keyed on category and fps alone
returns patch-32 tensors to a caller that asked for patch 8, silently and
with no shape error, because the model reads the patch dim off the data.
"""

import os
import json

from latplan.util.paths import DATA_DIR


def npz_cache_path(modality, dataset, category, fps,
                   num_objs=None, patch_size=None, fill_annotations=False):
    """Return absolute path for the per-category video cache file.

    Returns None when caching is disabled for this key (V9: all-cat → None,
    and likewise for a multi-category bake, which is not a single key).

    num_objs, patch_size and fill_annotations join the key because each one
    changes what the arrays hold. Passing None for either number keeps the
    old short name, so a cache written before these were part of the key is
    still found by a caller that does not set them.
    """
    if category is None or not isinstance(category, str):
        return None
    name = f"{category}-{fps}fps"
    if num_objs is not None:
        name += f"-mo{num_objs}"
    if patch_size is not None:
        name += f"-p{patch_size}"
    if fill_annotations:
        name += "-fill"
    return os.path.join(DATA_DIR, "npz", modality, dataset, name + ".npz")


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
