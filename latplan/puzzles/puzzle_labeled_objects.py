#!/usr/bin/env python3
"""
Labeled-objects data loader for FOSAE.

Reads fosae_labeled_dataset_unsloth.json (VLM-annotated COCO images).
Each image entry has objects with 'class' labels and 'bbox' [x1,y1,x2,y2]
in the original image's pixel coordinates.

Follows the same encoding as the blocksworld domain so that blocks_activation
and blocks_renderer can be reused verbatim:

  Object feature vector (assembled in strips.py, same as blocksworld):
    [ RGB patch (PATCH_SIZE^2 * 3 floats, sigmoid)
    | x1_onehot (X bins)  | y1_onehot (Y bins)
    | x2_onehot (X bins)  | y2_onehot (Y bins) ]

  Canvas:  CANVAS_H x CANVAS_W px, 5-px grid  →  Y = CANVAS_H//5, X = CANVAS_W//5
  FEATURE_DIM = PATCH_SIZE*PATCH_SIZE*3 + 2*X + 2*Y  =  3072 + 200  =  3272

build_dataset() returns raw (uint8 patches, uint16 pixel bboxes) so that the
caller (strips.py / visualize_fol.py / extract_fol.py) can apply the standard
preprocess() + bboxes_to_onehot() pipeline, identical to blocksworld.
"""

import os
import json
import numpy as np
from PIL import Image

# --- Tunable constants ---
PATCH_SIZE  = 32          # object crop resized to PATCH_SIZE x PATCH_SIZE x 3
MAX_OBJECTS = 10          # maximum objects per state (pad with zeros if fewer)

# Canvas onto which all bboxes are mapped (matches blocks-5-3 picsize)
CANVAS_H = 200
CANVAS_W = 300
PICSIZE  = [CANVAS_H, CANVAS_W, 3]   # same format as blocks .npz picsize field

_DEFAULT_DATASET = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "data", "gen", "fosae_labeled_dataset_unsloth.json")
_DEFAULT_IMAGES = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "data", "gen", "raw_images")


# --- Low-level helpers ---

def _crop_object(pil_img: Image.Image, bbox, patch_size: int = PATCH_SIZE) -> np.ndarray:
    """Crop bbox from image and resize to (patch_size, patch_size, 3) uint8."""
    x1, y1, x2, y2 = bbox
    W, H = pil_img.size
    x1, x2 = max(0, int(x1)), min(W, int(x2))
    y1, y2 = max(0, int(y1)), min(H, int(y2))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    patch = pil_img.crop((x1, y1, x2, y2))
    patch = patch.resize((patch_size, patch_size), Image.BILINEAR)
    return np.array(patch, dtype=np.uint8)   # (P, P, 3) in [0, 255]


def _scale_bbox_to_canvas(bbox, img_w: int, img_h: int) -> tuple:
    """Scale (x1,y1,x2,y2) from original image pixels to CANVAS pixel coords."""
    x1, y1, x2, y2 = bbox
    x1_c = int(np.clip(x1 * CANVAS_W / img_w, 0, CANVAS_W - 1))
    y1_c = int(np.clip(y1 * CANVAS_H / img_h, 0, CANVAS_H - 1))
    x2_c = int(np.clip(x2 * CANVAS_W / img_w, 0, CANVAS_W - 1))
    y2_c = int(np.clip(y2 * CANVAS_H / img_h, 0, CANVAS_H - 1))
    return (x1_c, y1_c, x2_c, y2_c)


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
    Convert one dataset entry to raw (uint8 patches, uint16 canvas bboxes, names).

    Returns
    -------
    patches  : (num_objs, patch_size, patch_size, 3)  uint8
    bboxes   : (num_objs, 4)  uint16  pixel coords in CANVAS space
    names    : list[str] of length num_objs
    """
    img_path = os.path.join(images_dir, entry["image"])
    pil_img  = Image.open(img_path).convert("RGB")
    W, H     = pil_img.size

    objs = entry.get("objects", [])
    # Sort by descending area for stable slot ordering
    objs = sorted(objs, key=lambda o: (o["bbox"][2]-o["bbox"][0])*(o["bbox"][3]-o["bbox"][1]),
                  reverse=True)
    objs = objs[:num_objs]

    names   = _unique_names(objs)
    patches = []
    bboxes  = []

    for obj in objs:
        patches.append(_crop_object(pil_img, obj["bbox"], patch_size))
        bboxes.append(_scale_bbox_to_canvas(obj["bbox"], W, H))

    # Pad with zeros for missing objects
    pad_names = [f"pad_{i}" for i in range(len(objs), num_objs)]
    for _ in pad_names:
        patches.append(np.zeros((patch_size, patch_size, 3), dtype=np.uint8))
        bboxes.append((0, 0, 0, 0))

    return (np.array(patches, dtype=np.uint8),    # (num_objs, P, P, 3)
            np.array(bboxes,   dtype=np.uint16),   # (num_objs, 4)
            names + pad_names)


# --- Dataset builder ---

def build_dataset(dataset_path: str = None, images_dir: str = None,
                  num_objs: int = MAX_OBJECTS, patch_size: int = PATCH_SIZE,
                  skip_empty: bool = True, max_images: int = None):
    """
    Load all images and return raw patches + pixel bboxes.

    Returns
    -------
    images       : (N, num_objs, patch_size, patch_size, 3)  uint8
    bboxes       : (N, num_objs, 4)  uint16  pixel coords in CANVAS space
    object_names : list[list[str]]  shape (N, num_objs)
    image_ids    : list[str]  COCO filenames for each state

    The caller is responsible for applying preprocessing and bbox one-hot
    encoding (identical to the blocksworld pipeline in strips.py):

        images = images.astype(np.float32) / 256
        images = preprocess(images)                       # latplan.puzzles.util
        picsize_grid = (np.array(PICSIZE) // 5).astype(int)
        Y, X = picsize_grid[0], picsize_grid[1]           # 40, 60
        bboxes_onehot = bboxes_to_onehot(bboxes, X, Y)   # strips.py
        states = np.concatenate(
            (images.reshape((N, num_objs, -1)),
             bboxes_onehot.reshape((N, num_objs, -1))), axis=-1)
    """
    if dataset_path is None:
        dataset_path = _DEFAULT_DATASET
    if images_dir is None:
        images_dir = _DEFAULT_IMAGES

    dataset_path = os.path.normpath(dataset_path)
    images_dir   = os.path.normpath(images_dir)

    with open(dataset_path) as f:
        data = json.load(f)

    if max_images is not None:
        data = data[:max_images]

    images_list, bboxes_list, all_names, image_ids = [], [], [], []
    for entry in data:
        if skip_empty and len(entry.get("objects", [])) == 0:
            continue
        patches, bboxes, names = entry_to_state(entry, images_dir, num_objs, patch_size)
        images_list.append(patches)
        bboxes_list.append(bboxes)
        all_names.append(names)
        image_ids.append(entry["image"])

    return (np.array(images_list, dtype=np.uint8),    # (N, num_objs, P, P, 3)
            np.array(bboxes_list,  dtype=np.uint16),   # (N, num_objs, 4)
            all_names, image_ids)


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
