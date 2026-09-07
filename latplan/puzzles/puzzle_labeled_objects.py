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

Source frames are **letterboxed** onto that canvas: one scale factor for both
axes, centred, leftover left empty. Scaling the axes independently -- what this
did until 2026-09-07 -- made an object's canvas shape a function of the source
video's aspect ratio, and the bbox block is the network's only position signal.
See `_scale_bbox_to_canvas` for the measurement and the rejected alternatives.

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

# Canvas onto which all bboxes are letterboxed. See _scale_bbox_to_canvas.
#
# The 200x300 was inherited from the blocksworld demo this loader was cloned
# from, and the comment here used to read "matches blocks-5-3 picsize" as if
# that were a reason. It is not one: blocksworld renders its own scenes at a
# size it chose, and no result in this thesis depends on the two agreeing.
# What the numbers actually have to satisfy is arithmetic, and only this:
#
#   * both divisible by 5, because the bbox one-hot grid is PICSIZE // 5;
#   * fixed for the whole corpus, because 2*(W//5) + 2*(H//5) = 200 of the
#     3272 feature dimensions and the network shape cannot vary per clip.
#
# 3:2 is a compromise nothing in VidVRD asked for -- 72% of its clips are 16:9
# -- and the letterbox therefore leaves 34 of 40 y-bins usable on the common
# case. 320x180 would be exactly 16:9, divide by 5, and give 64+36 bins for the
# *same* 3272-dim feature vector. It was rejected here only because that last
# coincidence makes it dangerous: an export written under one grid and read
# under the other has the right shape and wrong contents, so nothing raises.
# Changing it is a separate, deliberate migration, not a side effect of Q15.
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


def _letterbox_transform(img_w: int, img_h: int) -> tuple:
    """Return (scale, offset_x, offset_y) fitting img_w x img_h onto the canvas.

    One scale factor for both axes -- that is the whole point -- chosen as the
    larger one that still fits, with the leftover split evenly into bars.
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(
            "source frame is %sx%s; a bbox cannot be placed on the canvas "
            "without a real frame size" % (img_w, img_h))
    s = min(CANVAS_W / float(img_w), CANVAS_H / float(img_h))
    return s, (CANVAS_W - img_w * s) / 2.0, (CANVAS_H - img_h * s) / 2.0


def _scale_bbox_to_canvas(bbox, img_w: int, img_h: int) -> tuple:
    """Letterbox (x1,y1,x2,y2) from source image pixels onto the canvas.

    This output *is* the model's position signal: `strips.py` quantises it into
    the four one-hot runs that make up the bbox half of every feature vector.
    Nothing else tells the network where an object is.

    Until 2026-09-07 the two axes were scaled independently onto the 3:2
    canvas, which is an anisotropic map for any source that is not itself 3:2.
    A census of `data/video/vidvrd/annotations` run on 2026-09-07 found 1000
    clips at 45 distinct resolutions, aspect 0.564 (406x720) to 2.353
    (1920x816). An object's canvas width:height therefore came out as its true
    ratio times `1.5 / source_aspect`: measured against the old code, a square
    object rendered 63x100 px in a 1920x816 clip and 150x56 px in a 406x720
    one, so the same shape was presented to the network 4.2x differently
    depending only on which camera shot it. Shape, overlap and containment --
    exactly the relations this thesis asks FOSAE to find -- were all being
    reported as functions of the source file.

    Letterbox (uniform scale, centred, bars left empty) is used because it is
    the only one of the candidates that is both shape-preserving and lossless:

      * **stretch**, the old behaviour: fills the canvas, destroys shape. The
        defect above.
      * **crop / cover** (`max` instead of `min`): fills the canvas and keeps
        shape, but pushes content past the edge, where a clamped object decodes
        as "at the border" and a fully-excluded one as all-zero, which the
        metrics read as *absent*. Silent deletion is worse than empty bars.
      * **per-clip canvas**: no distortion at all, but the one-hot grid is
        PICSIZE // 5, so the feature width would vary per clip and no single
        network could consume the corpus.
      * **normalise to a unit square and pass the aspect as its own feature**:
        defensible, but it changes the feature layout, hence every export and
        every trained model. Out of scope for a geometry fix.

    The cost is stated rather than hidden, measured 2026-09-07 by mapping the
    full frame and reading off the bins it reaches: a 16:9 clip fills all 60
    x-bins but only **34 of the 40 y-bins** (y in [16, 184]), and a 4:3 clip
    fills all 40 y-bins but only **54 of the 60 x-bins**. The rest never fire.
    That is lost resolution, fixed per aspect ratio and identical for every
    clip of that shape; it is not distortion, and no object's shape depends on
    it. Trading it away was the point: a bin that never fires costs the network
    capacity, a bin that means something different per clip costs it the truth.

    Coordinates are rounded to the nearest canvas pixel, not truncated as
    before. Truncation cost up to a full pixel and, worse, is discontinuous at
    integers: the letterbox sends the two source boxes in
    `test_two_sources_of_different_aspect_give_one_canvas_box` to the same
    mathematical coordinates, but binary rounding leaves one of them at
    59.99999999999999, which truncates to 59 while the other truncates to 60.
    Half-up rounding is written out rather than `np.round`, which rounds halves
    to even and would make the map depend on the parity of the bin.

    Note the patch crop (`_crop_object`) still resizes anisotropically to a
    square. That is deliberate and is not this defect: it depends on the
    object's own shape, identically for every source resolution, and the patch
    carries appearance while this function carries geometry.
    """
    s, off_x, off_y = _letterbox_transform(img_w, img_h)
    x1, y1, x2, y2 = bbox

    def px(value, limit):
        return int(np.clip(np.floor(value + 0.5), 0, limit - 1))

    return (px(x1 * s + off_x, CANVAS_W), px(y1 * s + off_y, CANVAS_H),
            px(x2 * s + off_x, CANVAS_W), px(y2 * s + off_y, CANVAS_H))


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
