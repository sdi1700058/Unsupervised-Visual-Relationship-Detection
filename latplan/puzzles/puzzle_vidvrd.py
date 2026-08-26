#!/usr/bin/env python3
"""
VidVRD data loader for FOSAE.

Reads ImageNet-VidVRD annotation JSONs (one per video) + pre-extracted JPEG frames.
Output arrays are identical in shape and layout to puzzle_labeled_objects so
strips.py can reuse the same preprocessing pipeline (blocks_activation, bboxes_to_onehot).

Download data first:  bash sh/download_vidvrd.sh

Directory layout expected:
  data/video/vidvrd/annotations/{train,test}/<video_id>.json
  data/video/vidvrd/frames_<N>fps/{train,test}/<video_id>/<NNNNNN>.jpg
"""

import os
import json
import glob
import numpy as np
from PIL import Image

from latplan.puzzles.puzzle_labeled_objects import (
    _crop_object, _scale_bbox_to_canvas, PATCH_SIZE, MAX_OBJECTS, CANVAS_H, CANVAS_W, PICSIZE)
from latplan.util.cache import npz_cache_path, load_cached, save_cache
from latplan.util.paths  import DATA_DIR

_DEFAULT_ROOT       = os.path.join(DATA_DIR, "video", "vidvrd")
_DEFAULT_ANN_DIR    = os.path.join(_DEFAULT_ROOT, "annotations")
def _default_frames_dir(fps):
    return os.path.join(_DEFAULT_ROOT, f"frames_{fps}fps")

# Sidecar populated by build_dataset() so callers (strips.py, extract_fol.py,
# visualize_fol.py) can audit which videos were actually loaded without
# breaking the existing 4-tuple return signature.
last_load_metadata = {}


def _video_primary_category(ann):
    """Primary subject of a VidVRD annotation = tid that appears in the most
    frames (mean bbox area as tiebreaker). None if no trajectories/subjects."""
    subjects = ann.get("subject/objects", [])
    if not subjects:
        return None
    tid_to_cat = {o["tid"]: o["category"] for o in subjects}
    tid_count = {}
    tid_area  = {}
    for frame_objs in ann.get("trajectories", []):
        for o in frame_objs:
            tid = o["tid"]
            tid_count[tid] = tid_count.get(tid, 0) + 1
            b = o["bbox"]
            tid_area[tid] = tid_area.get(tid, 0) + (b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"])
    if not tid_count:
        return None
    best_tid = max(tid_count, key=lambda t: (tid_count[t], tid_area.get(t, 0)))
    return tid_to_cat.get(best_tid)


def build_dataset(annotations_dir=None, frames_dir=None,
                  num_objs=MAX_OBJECTS, max_videos=None, split="train",
                  category_filter=None, fps=3, video_id_filter=None,
                  fill_annotations=False, patch_size=None):
    """
    Load all annotated frames from VidVRD.

    Returns
    -------
    images       : (N, num_objs, PATCH_SIZE, PATCH_SIZE, 3)  uint8
    bboxes       : (N, num_objs, 4)  uint16  pixel coords in CANVAS space
    object_names : list[list[str]]  shape (N, num_objs)
    frame_ids    : list[str]  '<video_id>/<fid>' — preserves video order for sequential transitions

    States are ordered: all frames of video_0, then video_1, ... so that
    build_transitions(..., mode='sequential') only pairs frames within the same video.

    video_id_filter : str | list[str] | None
        Restrict loading to one or more exact video-ids (e.g.
        'ILSVRC2015_train_00010001'). Bypasses the default per-category cache.
    """
    if annotations_dir is None:
        annotations_dir = os.path.normpath(os.path.join(_DEFAULT_ANN_DIR, split))
    if frames_dir is None:
        frames_dir = os.path.normpath(os.path.join(_default_frames_dir(fps), split))

    # patch_size overrides the module default for THIS bake only (env-overridable
    # via setup-dataset.py --patch-size). model.py:blocks_activation auto-detects
    # the patch dim from the data tensor at train time, so no model.py change is
    # needed when this is increased.
    _patch_size = patch_size if patch_size is not None else PATCH_SIZE

    # SPEC §V7-V9: per-category npz cache short-circuit. Bypassed when
    # max_videos or video_id_filter is set (would contaminate shared cache).
    cache_path = npz_cache_path("video", "vidvrd", category_filter, fps) \
        if (max_videos is None and video_id_filter is None) else None
    if cache_path is not None:
        hit = load_cached(cache_path)
        if hit is not None:
            images, bboxes, names, frame_ids, meta = hit
            last_load_metadata.clear()
            last_load_metadata.update(meta)
            print(f"[vidvrd-loader] cache hit {cache_path} "
                  f"({meta.get('num_videos','?')} videos, {meta.get('num_states','?')} states)")
            return images, bboxes, names, frame_ids

    ann_files = sorted(glob.glob(os.path.join(annotations_dir, "*.json")))
    if not ann_files:
        raise FileNotFoundError(f"No annotation JSONs in {annotations_dir}. Run sh/download_vidvrd.sh first.")
    if video_id_filter is not None:
        wanted = {video_id_filter} if isinstance(video_id_filter, str) else set(video_id_filter)
        ann_files = [f for f in ann_files
                     if os.path.splitext(os.path.basename(f))[0] in wanted]
        if not ann_files:
            raise RuntimeError(f"video_id_filter {video_id_filter!r} matched 0 annotation JSONs in {annotations_dir}")
    strict = os.environ.get("VIDVRD_STRICT_CATEGORY", "1") == "1"

    # The category filter runs before max_videos, not after. Slicing first
    # would make `--max-videos N` mean "look at the first N annotation files"
    # rather than "load N videos of this category" — and since the files sort
    # by video id, a category that happens to sort late loads nothing at all.
    if category_filter is not None:
        kept = []
        for ann_path in ann_files:
            with open(ann_path) as f:
                ann = json.load(f)
            if strict:
                match = _video_primary_category(ann) == category_filter
            else:
                match = category_filter in {obj["category"]
                                            for obj in ann.get("subject/objects", [])}
            if match:
                kept.append(ann_path)
        if not kept:
            raise RuntimeError(
                f"category_filter {category_filter!r} matched 0 videos in "
                f"{annotations_dir} with VIDVRD_STRICT_CATEGORY="
                f"{'1' if strict else '0'}. Strict mode keeps only videos whose "
                f"primary subject is that category; set VIDVRD_STRICT_CATEGORY=0 "
                f"to keep any video the category appears in.")
        print(f"[vidvrd-loader] category {category_filter!r}: {len(kept)} of "
              f"{len(ann_files)} videos match (strict={strict})")
        ann_files = kept

    if max_videos is not None:
        if max_videos < len(ann_files):
            print(f"[vidvrd-loader] capping at {max_videos} of {len(ann_files)} videos")
        ann_files = ann_files[:max_videos]

    images_list, bboxes_list, all_names, frame_ids = [], [], [], []
    loaded_video_ids = []
    loaded_primary   = {}

    for ann_path in ann_files:
        with open(ann_path) as f:
            ann = json.load(f)

        vid_id = ann["video_id"]
        W, H   = ann["width"], ann["height"]
        tid_to_cat = {obj["tid"]: obj["category"] for obj in ann.get("subject/objects", [])}
        primary    = _video_primary_category(ann)

        vid_frames_dir = os.path.join(frames_dir, vid_id)

        trajectories = ann.get("trajectories", [])

        # `fill_annotations=True`: backward-fill leading empties from the first
        # non-empty entry, then forward-fill mid-stream empties from the last
        # non-empty entry. Gives dense per-frame supervision even when the
        # annotator only marked a sub-range. The IMAGE per filled frame is the
        # real source-PTS jpeg; only the bbox/class list is reused. Valid when
        # tracked objects are continuously visible (dog-frisbee, sheep, etc).
        if fill_annotations:
            first_idx = next((i for i, t in enumerate(trajectories) if t), None)
            if first_idx is not None and first_idx > 0:
                first_objs   = trajectories[first_idx]
                trajectories = list(trajectories)
                for i in range(first_idx):
                    trajectories[i] = first_objs

        _last_objs = None
        for fid, frame_objs in enumerate(trajectories):
            if not frame_objs:
                if fill_annotations and _last_objs is not None:
                    frame_objs = _last_objs
                else:
                    continue
            else:
                _last_objs = frame_objs
            # ffmpeg `-frame_pts true` writes source-frame-index filenames (B5).
            frame_path = os.path.join(vid_frames_dir, f"{fid:06d}.jpg")
            if not os.path.exists(frame_path):
                continue

            pil_img = Image.open(frame_path).convert("RGB")

            # Sort by bbox area descending for stable slot assignment
            def _area(o):
                b = o["bbox"]
                return (b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"])
            frame_objs = sorted(frame_objs, key=_area, reverse=True)[:num_objs]

            patches, bboxes, names = [], [], []
            for obj in frame_objs:
                b    = obj["bbox"]
                bbox = (b["xmin"], b["ymin"], b["xmax"], b["ymax"])
                patches.append(_crop_object(pil_img, bbox, patch_size=_patch_size))
                bboxes.append(_scale_bbox_to_canvas(bbox, W, H))
                names.append(tid_to_cat.get(obj["tid"], f"obj{obj['tid']}"))

            # Pad to num_objs
            for i in range(len(frame_objs), num_objs):
                patches.append(np.zeros((_patch_size, _patch_size, 3), dtype=np.uint8))
                bboxes.append((0, 0, 0, 0))
                names.append(f"pad_{i}")

            images_list.append(np.array(patches, dtype=np.uint8))
            bboxes_list.append(np.array(bboxes,  dtype=np.uint16))
            all_names.append(names)
            frame_ids.append(f"{vid_id}/{fid:06d}")

        if vid_id not in loaded_primary:
            loaded_video_ids.append(vid_id)
            loaded_primary[vid_id] = primary

    if not images_list:
        raise RuntimeError("No frames loaded. Check annotations_dir and frames_dir paths.")

    last_load_metadata.clear()
    last_load_metadata.update({
        "category_filter": category_filter,
        "video_id_filter": list(video_id_filter) if isinstance(video_id_filter, (list, tuple, set)) else video_id_filter,
        "strict": strict,
        "video_ids": loaded_video_ids,
        "primary_categories": loaded_primary,
        "num_videos": len(loaded_video_ids),
        "num_states": len(images_list),
        "fps": fps,
        "fill_annotations": fill_annotations,
        "patch_size": _patch_size,
    })
    print(f"[vidvrd-loader] category_filter={category_filter} strict={strict} "
          f"loaded {len(loaded_video_ids)}/{len(ann_files)} videos, "
          f"{len(images_list)} states")

    images_arr = np.array(images_list, dtype=np.uint8)
    bboxes_arr = np.array(bboxes_list, dtype=np.uint16)

    if cache_path is not None:
        save_cache(cache_path, images_arr, bboxes_arr, all_names, frame_ids, dict(last_load_metadata))
        print(f"[vidvrd-loader] cache write {cache_path}")

    return images_arr, bboxes_arr, all_names, frame_ids


def build_transitions(states, frame_ids, mode="sequential"):
    """
    Build transition pairs respecting video boundaries.

    mode='sequential': only pair consecutive frames from the same video (V3).
    mode='all_pairs' : every ordered pair regardless of video (not recommended for VidVRD).
    """
    if mode == "sequential":
        pres, sucs = [], []
        for i in range(len(states) - 1):
            vid_i = frame_ids[i].split("/")[0]
            vid_j = frame_ids[i + 1].split("/")[0]
            if vid_i == vid_j:
                pres.append(states[i])
                sucs.append(states[i + 1])
        if not pres:
            raise RuntimeError("No sequential transitions found. Check that multiple frames per video were loaded.")
        return np.array([pres, sucs])
    elif mode == "all_pairs":
        n = len(states)
        idx = [(i, j) for i in range(n) for j in range(n) if i != j]
        ip, is_ = zip(*idx)
        return np.array([states[np.array(ip)], states[np.array(is_)]])
    else:
        raise ValueError(f"Unknown mode '{mode}'")
