#!/usr/bin/env python3
"""ActionGenome loader for FOSAE (SPEC §I9).

Per-frame schema:
  object_bbox_and_relationship.pkl  dict[<vid>.mp4/<NNNNNN>.png, list[obj_dict]]
  person_bbox.pkl                   dict[<vid>.mp4/<NNNNNN>.png, person_dict]
  frame_list.txt                    canonical frame keys
Frame extraction: `sh/extract_ag_frames.sh` (C1) → `data/video/actiongenome/frames/<vid>.mp4/<NNNNNN>.png`.

Slot 0 = person (always present in AG); remaining slots = objects sorted by bbox area.
Uses the same patch / canvas / one-hot pipeline as puzzle_labeled_objects (3272-dim feature).
"""

import os
import pickle
import numpy as np
from PIL import Image

from latplan.puzzles.puzzle_labeled_objects import (
    _crop_object, _scale_bbox_to_canvas, PATCH_SIZE, MAX_OBJECTS,
    CANVAS_H, CANVAS_W, PICSIZE)
from latplan.util.cache import npz_cache_path, load_cached, save_cache
from latplan.util.paths  import DATA_DIR


_DEFAULT_ROOT    = os.path.join(DATA_DIR, "video", "actiongenome")
_DEFAULT_ANN_DIR = os.path.join(_DEFAULT_ROOT, "annotations")
_DEFAULT_FRAMES  = os.path.join(_DEFAULT_ROOT, "frames")

last_load_metadata = {}


def _load_pkls(ann_dir):
    with open(os.path.join(ann_dir, "object_bbox_and_relationship.pkl"), "rb") as f:
        obj = pickle.load(f)
    with open(os.path.join(ann_dir, "person_bbox.pkl"), "rb") as f:
        per = pickle.load(f)
    return obj, per


def _video_primary_object_category(frames_of_vid, obj_anno):
    """Pick most-visible non-person object class across this video's frames."""
    counts = {}
    areas  = {}
    for fkey in frames_of_vid:
        for o in obj_anno.get(fkey, []):
            if not o.get("visible") or o.get("bbox") is None:
                continue
            c = o.get("class")
            if c is None or c == "person":
                continue
            counts[c] = counts.get(c, 0) + 1
            x1, y1, x2, y2 = o["bbox"]
            areas[c] = areas.get(c, 0) + max(0, x2 - x1) * max(0, y2 - y1)
    if not counts:
        return None
    return max(counts, key=lambda c: (counts[c], areas.get(c, 0)))


def build_dataset(annotations_dir=None, frames_dir=None,
                  num_objs=MAX_OBJECTS, max_videos=None, split="train",
                  category_filter=None, fps="native",
                  video_id_filter=None):
    """Load AG annotated frames; return (images, bboxes, names, frame_ids).

    Parameters mirror `puzzle_vidvrd.build_dataset` for consistency.

    fps : str | int
        Cache-key suffix only (AG frames are extracted at the video's native
        FPS; downsampling is not configurable upstream). Default `'native'`.

    video_id_filter : str | list[str] | None
        Restrict loading to one or more exact video-id directories (e.g.
        ``'001YG.mp4'``). Used by the smallest-overfit pipeline. When set,
        the default per-category cache is bypassed (keyspace would collide).
    """
    if annotations_dir is None: annotations_dir = _DEFAULT_ANN_DIR
    if frames_dir is None:      frames_dir      = _DEFAULT_FRAMES

    # SPEC §V7-V9: per-category npz cache. Skipped when max_videos or
    # video_id_filter are set (both would otherwise contaminate the shared
    # per-category cache).
    cache_path = npz_cache_path("video", "actiongenome", category_filter, fps) \
        if (max_videos is None and video_id_filter is None) else None
    if cache_path is not None:
        hit = load_cached(cache_path)
        if hit is not None:
            images, bboxes, names, frame_ids, meta = hit
            last_load_metadata.clear()
            last_load_metadata.update(meta)
            print(f"[ag-loader] cache hit {cache_path} "
                  f"({meta.get('num_videos','?')} videos, {meta.get('num_states','?')} states)")
            return images, bboxes, names, frame_ids

    obj_anno, per_anno = _load_pkls(annotations_dir)
    frame_list_path = os.path.join(annotations_dir, "frame_list.txt")
    with open(frame_list_path) as f:
        all_frames = [l.strip() for l in f if l.strip()]

    # split filter via per-frame metadata['set']
    if split is not None:
        kept = []
        for k in all_frames:
            anns = obj_anno.get(k, [])
            if not anns:
                continue
            meta = anns[0].get("metadata") or {}
            if meta.get("set") == split:
                kept.append(k)
        all_frames = kept

    vid_to_frames = {}
    for k in all_frames:
        vid = k.split("/", 1)[0]
        vid_to_frames.setdefault(vid, []).append(k)
    for vid in vid_to_frames:
        vid_to_frames[vid].sort()

    strict = os.environ.get("AG_STRICT_CATEGORY", "1") == "1"
    video_ids = sorted(vid_to_frames.keys())
    if video_id_filter is not None:
        wanted = {video_id_filter} if isinstance(video_id_filter, str) else set(video_id_filter)
        video_ids = [v for v in video_ids if v in wanted]
        if not video_ids:
            raise RuntimeError(f"video_id_filter {video_id_filter!r} matched 0 videos")
    if max_videos is not None:
        video_ids = video_ids[:max_videos]

    images_list, bboxes_list, all_names, frame_ids = [], [], [], []
    loaded_video_ids, loaded_primary = [], {}

    for vid in video_ids:
        frames_of_vid = vid_to_frames[vid]
        primary = _video_primary_object_category(frames_of_vid, obj_anno)
        if category_filter is not None:
            if strict:
                if primary != category_filter:
                    continue
            else:
                hit = False
                for fk in frames_of_vid:
                    for o in obj_anno.get(fk, []):
                        if o.get("visible") and o.get("class") == category_filter:
                            hit = True
                            break
                    if hit:
                        break
                if not hit:
                    continue

        vid_frames_dir = os.path.join(frames_dir, vid)

        for fkey in frames_of_vid:
            objs   = obj_anno.get(fkey, [])
            person = per_anno.get(fkey, {}) or {}
            visible_objs = [o for o in objs if o.get("visible") and o.get("bbox") is not None]

            person_bbox = None
            pb = person.get("bbox")
            if pb is not None and len(pb) > 0:
                person_bbox = tuple(map(float, pb[0]))

            if not visible_objs and person_bbox is None:
                continue

            frame_filename = fkey.split("/", 1)[1]
            frame_path = os.path.join(vid_frames_dir, frame_filename)
            if not os.path.exists(frame_path):
                continue

            pil_img = Image.open(frame_path).convert("RGB")
            W, H = pil_img.size

            slots = []
            if person_bbox is not None:
                slots.append(("person", person_bbox))

            def _area(o):
                x1, y1, x2, y2 = o["bbox"]
                return max(0, x2 - x1) * max(0, y2 - y1)

            for o in sorted(visible_objs, key=_area, reverse=True):
                slots.append((o["class"], tuple(map(float, o["bbox"]))))
                if len(slots) >= num_objs:
                    break

            patches, bboxes, names = [], [], []
            for cls, bbox in slots[:num_objs]:
                patches.append(_crop_object(pil_img, bbox))
                bboxes.append(_scale_bbox_to_canvas(bbox, W, H))
                names.append(cls)

            for i in range(len(slots), num_objs):
                patches.append(np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8))
                bboxes.append((0, 0, 0, 0))
                names.append(f"pad_{i}")

            images_list.append(np.array(patches, dtype=np.uint8))
            bboxes_list.append(np.array(bboxes,  dtype=np.uint16))
            all_names.append(names)
            frame_ids.append(f"{vid}/{frame_filename}")

        if vid not in loaded_primary:
            loaded_video_ids.append(vid)
            loaded_primary[vid] = primary

    if not images_list:
        raise RuntimeError("No frames loaded. Check annotations_dir and frames_dir paths.")

    last_load_metadata.clear()
    last_load_metadata.update({
        "category_filter":    category_filter,
        "video_id_filter":    list(video_id_filter) if isinstance(video_id_filter, (list,tuple,set)) else video_id_filter,
        "strict":             strict,
        "video_ids":          loaded_video_ids,
        "primary_categories": loaded_primary,
        "num_videos":         len(loaded_video_ids),
        "num_states":         len(images_list),
        "fps":                fps,
    })
    print(f"[ag-loader] category_filter={category_filter} strict={strict} "
          f"loaded {len(loaded_video_ids)}/{len(video_ids)} videos, "
          f"{len(images_list)} states")

    images_arr = np.array(images_list, dtype=np.uint8)
    bboxes_arr = np.array(bboxes_list, dtype=np.uint16)

    if cache_path is not None:
        save_cache(cache_path, images_arr, bboxes_arr, all_names, frame_ids, dict(last_load_metadata))
        print(f"[ag-loader] cache write {cache_path}")

    return images_arr, bboxes_arr, all_names, frame_ids


def build_transitions(states, frame_ids, mode="sequential"):
    """AG transitions: consecutive annotated frames within same video.

    V3: sequential only (paper-consistent). all_pairs available but
    not recommended for video-world domains.
    """
    if mode == "sequential":
        pres, sucs = [], []
        for i in range(len(states) - 1):
            vi = frame_ids[i].split("/", 1)[0]
            vj = frame_ids[i + 1].split("/", 1)[0]
            if vi == vj:
                pres.append(states[i])
                sucs.append(states[i + 1])
        if not pres:
            raise RuntimeError("No sequential transitions found.")
        return np.array([pres, sucs])
    elif mode == "all_pairs":
        n = len(states)
        idx = [(i, j) for i in range(n) for j in range(n) if i != j]
        ip, is_ = zip(*idx)
        return np.array([states[np.array(ip)], states[np.array(is_)]])
    else:
        raise ValueError(f"Unknown mode '{mode}'")
