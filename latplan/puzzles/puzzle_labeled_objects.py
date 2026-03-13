#!/usr/bin/env python3
"""
Labeled-objects data loader for FOSAE.

Reads fosae_labeled_dataset_unsloth.json (VLM-annotated COCO images).
Each image entry has objects with 'class' labels and 'bbox' [x1,y1,x2,y2].

Object features per slot:
  - Grayscale image crop resized to PATCH_SIZE x PATCH_SIZE  (preprocessed)
  - Normalised bounding-box  [cx/W, cy/H, w/W, h/H]          (4 values)

States:  (num_states, MAX_OBJECTS, FEATURE_DIM)
Objects are ordered by area (largest first) so dominant objects get stable slots.
Within each image, duplicate class names are disambiguated with a numeric suffix:
  "chair" -> "chair_0", "chair_1", "chair_2"

The returned object_names list is used downstream by extract_fol.py to annotate
predicate arguments with real semantic labels instead of generic "obj_0" indices.
"""

import os
import json
import numpy as np
from PIL import Image

# --- Tunable constants ---
PATCH_SIZE   = 16    # each object crop is resized to PATCH_SIZE x PATCH_SIZE (grey)
MAX_OBJECTS  = 8     # maximum number of objects per state (pad with zeros if fewer)
FEATURE_DIM  = PATCH_SIZE * PATCH_SIZE + 4   # 256 + 4 = 260

_DEFAULT_DATASET = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "data", "gen", "fosae_labeled_dataset_unsloth.json")
_DEFAULT_IMAGES = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "data", "gen", "raw_images")


# --- Low-level helpers ---

def _preprocess_patch(arr: np.ndarray) -> np.ndarray:
    """Equalize + normalize + enhance a float array, matching latplan convention."""
    from skimage import exposure
    arr = arr.astype(float)
    arr = exposure.equalize_hist(arr)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    arr = np.clip((arr - 0.5) * 3, -0.5, 0.5) + 0.5
    return arr


def _crop_object(pil_img: Image.Image, bbox, patch_size: int = PATCH_SIZE) -> np.ndarray:
    """Crop and resize object to *greyscale* (patch_size, patch_size) in [0,1]."""
    x1, y1, x2, y2 = bbox
    W, H = pil_img.size
    x1, x2 = max(0, int(x1)), min(W, int(x2))
    y1, y2 = max(0, int(y1)), min(H, int(y2))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((patch_size, patch_size), dtype=np.float32)
    patch = pil_img.crop((x1, y1, x2, y2)).convert("L")
    patch = patch.resize((patch_size, patch_size), Image.BILINEAR)
    arr = np.array(patch, dtype=np.float32) / 255.0
    return _preprocess_patch(arr).astype(np.float32)


def _bbox_features(bbox, img_w: int, img_h: int) -> np.ndarray:
    """Return [cx/W, cy/H, w/W, h/H] normalised centre+size representation.

    Clamps to [0, 1] because VLM-generated bboxes can occasionally extend
    outside image boundaries.
    """
    x1, y1, x2, y2 = bbox
    # Clamp to valid image region before normalising
    x1 = max(0, min(img_w, x1))
    x2 = max(0, min(img_w, x2))
    y1 = max(0, min(img_h, y1))
    y2 = max(0, min(img_h, y2))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = x2 - x1
    bh = y2 - y1
    return np.array([cx / img_w, cy / img_h, bw / img_w, bh / img_h],
                    dtype=np.float32)


def _unique_names(objects) -> list:
    """
    Given a list of object dicts (each with 'class' key), return a list of
    *unique* string names: if a class appears once, use its name as-is;
    if it appears multiple times, append _0, _1, ... suffixes.
    """
    from collections import Counter
    counts = Counter(o["class"] for o in objects)
    seen   = {}
    names  = []
    for obj in objects:
        cls = obj["class"]
        if counts[cls] == 1:
            names.append(cls)
        else:
            idx = seen.get(cls, 0)
            names.append(f"{cls}_{idx}")
            seen[cls] = idx + 1
    return names


# --- Per-image state constructor ---

def entry_to_state(entry: dict, images_dir: str,
                   num_objs: int = MAX_OBJECTS,
                   patch_size: int = PATCH_SIZE):
    """
    Convert one dataset entry to
      state  : (num_objs, FEATURE_DIM) float32 array
      names  : list[str] of length num_objs (semantic labels or 'pad_k')

    Objects are sorted by descending bbox area so that the model's top slots
    consistently correspond to dominant objects.
    """
    img_path = os.path.join(images_dir, entry["image"])
    pil_img  = Image.open(img_path).convert("RGB")
    W, H     = pil_img.size

    objs = entry.get("objects", [])
    # Sort by descending area for stable ordering
    objs = sorted(objs, key=lambda o: (o["bbox"][2]-o["bbox"][0])*(o["bbox"][3]-o["bbox"][1]),
                  reverse=True)
    objs = objs[:num_objs]   # truncate to num_objs

    feature_dim = patch_size * patch_size + 4
    vectors = []
    names   = _unique_names(objs)

    for obj in objs:
        patch = _crop_object(pil_img, obj["bbox"], patch_size)
        bbox_feat = _bbox_features(obj["bbox"], W, H)
        vectors.append(np.concatenate([patch.flatten(), bbox_feat]))

    # Pad with zeros for missing objects
    pad_names = [f"pad_{i}" for i in range(len(objs), num_objs)]
    for _ in pad_names:
        vectors.append(np.zeros(feature_dim, dtype=np.float32))
    names = names + pad_names

    return np.stack(vectors, axis=0).astype(np.float32), names   # (num_objs, feat_dim)


# --- Dataset builder ---

def build_dataset(dataset_path: str = None, images_dir: str = None,
                  num_objs: int = MAX_OBJECTS, patch_size: int = PATCH_SIZE,
                  skip_empty: bool = True):
    """
    Load all images and return:
      states       : (N, num_objs, feature_dim) float32
      object_names : list[list[str]]  shape (N, num_objs)
      image_ids    : list[str]        COCO filenames for each state

    Parameters
    ----------
    skip_empty : if True, drop images with no detected objects.
    """
    if dataset_path is None:
        dataset_path = _DEFAULT_DATASET
    if images_dir is None:
        images_dir = _DEFAULT_IMAGES

    dataset_path = os.path.normpath(dataset_path)
    images_dir   = os.path.normpath(images_dir)

    with open(dataset_path) as f:
        data = json.load(f)

    states, all_names, image_ids = [], [], []
    for entry in data:
        if skip_empty and len(entry.get("objects", [])) == 0:
            continue
        state, names = entry_to_state(entry, images_dir, num_objs, patch_size)
        states.append(state)
        all_names.append(names)
        image_ids.append(entry["image"])

    return np.array(states, dtype=np.float32), all_names, image_ids


def build_transitions(states: np.ndarray, mode: str = "sequential"):
    """
    Build transition pairs (pre, suc) from a state array.

    mode='sequential' : state[i] -> state[i+1] for all i        (N-1 pairs)
    mode='all_pairs'  : every ordered pair (i,j) with i!=j        (N*(N-1) pairs)

    Returns
    -------
    transitions : (2, num_pairs, num_objs, feature_dim)  [pre=0, suc=1]
    """
    n = len(states)
    if mode == "sequential":
        pres = states[:-1]
        sucs = states[1:]
    elif mode == "all_pairs":
        idx_pre, idx_suc = zip(*[(i, j) for i in range(n)
                                         for j in range(n) if i != j])
        pres = states[np.array(idx_pre)]
        sucs = states[np.array(idx_suc)]
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'sequential' or 'all_pairs'.")

    return np.array([pres, sucs])   # (2, num_pairs, num_objs, feature_dim)


# --- Convenience: flat object_names for a single representative state ---

def canonical_object_names(all_object_names: list) -> list:
    """
    Return the object name list for the first state in the dataset.
    Used as the ``object_names`` argument to extract_fol_from_model when
    all states share the same object set (e.g. a fixed scene with moving objects).

    For mixed-scene datasets (like COCO) pass None to extract_fol_from_model
    and annotate each state individually.
    """
    if not all_object_names:
        return []
    return all_object_names[0]
