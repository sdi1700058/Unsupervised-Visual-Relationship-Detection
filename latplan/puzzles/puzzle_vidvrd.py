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


def _bbox_area(obj):
    b = obj["bbox"]
    return (b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"])


def _slot_order_by_tid(trajectories, num_objs):
    """`{tid: slot}` for one clip, so a track keeps one slot in every frame.

    **The defect this exists to close.** Slots were filled per frame by
    descending bbox area, independently of every other frame. An object that
    leaves the frame moves every object behind it up one slot and turns the
    tail into padding. FOSAE learns its predicates on slot positions, so a
    transition whose slot 1 is the dog before and the frisbee after is not the
    transition that happened. Measured on the VidVRD production read: **138
    adjacent pairs change the slot name order, 15 change the object count.**

    VidVRD ships `tid`, a per-clip track id, and this loader used it only to
    look up a category name.

    Ranked by total bbox area over the clip, then by how many frames the track
    appears in, then by tid. Total area first because it is the same quantity
    the per-frame sort used, so a frame in which every track is present comes
    out in the order the default already produced; the rest of the ranking only
    breaks ties, and tid makes it total, so the order is a property of the clip
    rather than of whichever frame is being read.

    Tracks beyond `num_objs` are left out of the mapping entirely. Dropping the
    same track from every frame is the point: rotating it in whenever it
    happens to be large is the disease being cured.
    """
    area, count = {}, {}
    for frame_objs in trajectories:
        for obj in frame_objs or []:
            tid = obj.get("tid")
            if tid is None:
                continue
            area[tid] = area.get(tid, 0) + _bbox_area(obj)
            count[tid] = count.get(tid, 0) + 1
    ranked = sorted(area, key=lambda t: (-area[t], -count[t], str(t)))
    return dict((tid, slot) for slot, tid in enumerate(ranked[:num_objs])), len(ranked)


def _place_by_tid(frame_objs, slot_of, num_objs, counts):
    """One frame's objects into fixed slots. `None` marks a vacancy.

    An object carrying no tid cannot be pinned. It falls back to the old
    behaviour -- area order into whatever slots are still free -- and is
    counted, because a fallback nobody counts is the silence this whole change
    is against.
    """
    placed = [None] * num_objs
    untracked = []
    for obj in frame_objs:
        tid = obj.get("tid")
        if tid is None:
            untracked.append(obj)
            continue
        slot = slot_of.get(tid)
        if slot is None or placed[slot] is not None:
            counts["dropped_past_slots"] += 1
            continue
        placed[slot] = obj
        counts["placed_by_id"] += 1
    if untracked:
        free = [i for i, p in enumerate(placed) if p is None]
        for obj, slot in zip(sorted(untracked, key=_bbox_area, reverse=True), free):
            placed[slot] = obj
            counts["placed_by_area"] += 1
        counts["dropped_past_slots"] += max(0, len(untracked) - len(free))
    return placed


def _describe_slots(counts):
    """One line for a log, so the ordering in force is never a guess."""
    tail = ("%d placements, %d objects dropped past slot %d; the widest video "
            "held %d identities"
            % (counts["placed_by_id"] + counts["placed_by_area"],
               counts["dropped_past_slots"], counts["num_objs"],
               counts["max_identities_in_a_video"]))
    if not counts["strict"]:
        return ("slots by bbox area per frame, so an object that leaves moves "
                "every later object up one slot: " + tail
                + ". Set STRICT_ADJACENCY=1 to pin each identity to one slot")
    text = ("slots pinned by %s under STRICT_ADJACENCY=1: " % counts["id_source"]) + tail
    if counts["placed_by_area"]:
        text += ("; %d of those had no %s and fell back to area order"
                 % (counts["placed_by_area"], counts["id_source"]))
    return text


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
            # An object with no tid cannot be the primary subject of anything,
            # and raising here would refuse the whole video over one box.
            tid = o.get("tid")
            if tid is None:
                continue
            tid_count[tid] = tid_count.get(tid, 0) + 1
            tid_area[tid] = tid_area.get(tid, 0) + _bbox_area(o)
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

    # STRICT_ADJACENCY=1 pins each track to one slot for the whole clip; see
    # _slot_order_by_tid. The same flag that refuses non-adjacent transitions,
    # because both are the same promise: a transition is one step, between two
    # states whose slots mean the same objects.
    from latplan.util.adjacency import strict_from_env
    strict_slots = strict_from_env()

    # SPEC §V7-V9: per-category npz cache short-circuit. Bypassed when
    # max_videos or video_id_filter is set (would contaminate shared cache).
    cache_path = npz_cache_path("video", "vidvrd", category_filter, fps,
                                num_objs=num_objs, patch_size=_patch_size,
                                fill_annotations=fill_annotations) \
        if (max_videos is None and video_id_filter is None) else None
    # Pinning changes what the arrays hold, so it has to change the key too --
    # cache.py's own rule. Without this a strict bake reads back the
    # area-ordered tensors an earlier bake wrote, silently, and trains on the
    # ordering it was set to avoid. The suffix is added here rather than in
    # npz_cache_path so that a cache written before today is still found by
    # default.
    if cache_path is not None and strict_slots:
        cache_path = cache_path[:-len(".npz")] + "-tid.npz"
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
    # category_filter takes one category, several, or None for every video.
    wanted_cats = None
    if category_filter is not None:
        wanted_cats = ({category_filter} if isinstance(category_filter, str)
                       else set(category_filter))

    if wanted_cats:
        kept = []
        for ann_path in ann_files:
            with open(ann_path) as f:
                ann = json.load(f)
            if strict:
                match = _video_primary_category(ann) in wanted_cats
            else:
                match = bool(wanted_cats & {obj["category"]
                                            for obj in ann.get("subject/objects", [])})
            if match:
                kept.append(ann_path)
        if not kept:
            raise RuntimeError(
                f"category_filter {category_filter!r} matched 0 videos in "
                f"{annotations_dir} with VIDVRD_STRICT_CATEGORY="
                f"{'1' if strict else '0'}. Strict mode keeps only videos whose "
                f"primary subject is that category; set VIDVRD_STRICT_CATEGORY=0 "
                f"to keep any video the category appears in.")
        print(f"[vidvrd-loader] categories {sorted(wanted_cats)}: {len(kept)} of "
              f"{len(ann_files)} videos match (strict={strict})")
        ann_files = kept
    else:
        print(f"[vidvrd-loader] no category filter: all {len(ann_files)} videos")

    if max_videos is not None:
        if max_videos < len(ann_files):
            print(f"[vidvrd-loader] capping at {max_videos} of {len(ann_files)} videos")
        ann_files = ann_files[:max_videos]

    images_list, bboxes_list, all_names, frame_ids = [], [], [], []
    loaded_video_ids = []
    loaded_primary   = {}

    # What the load throws away. Discarded in silence until 2026-09-06, so
    # `num_states` reported what survived and nothing reported what did not.
    _dropped = {"annotated_frames": 0, "empty_annotation": 0,
                "missing_frame_file": 0, "filled": 0}

    # Which ordering produced the slots, and what it cost. Counted in both
    # modes so the two are comparable.
    _slots = {"strict": strict_slots, "id_source": "tid", "num_objs": num_objs,
              "placed_by_id": 0, "placed_by_area": 0, "dropped_past_slots": 0,
              "max_identities_in_a_video": 0}

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

        # One ranking per clip, computed before any frame is read, so no frame
        # can influence where the next one puts its objects.
        slot_of, n_identities = _slot_order_by_tid(trajectories, num_objs)
        _slots["max_identities_in_a_video"] = max(
            _slots["max_identities_in_a_video"], n_identities)

        _last_objs = None
        for fid, frame_objs in enumerate(trajectories):
            _dropped["annotated_frames"] += 1
            if not frame_objs:
                if fill_annotations and _last_objs is not None:
                    frame_objs = _last_objs
                    _dropped["filled"] += 1
                else:
                    # Counted, not silent. Measured 3,375 of 7,770 trajectory
                    # entries empty on a 60-video read.
                    _dropped["empty_annotation"] += 1
                    continue
            else:
                _last_objs = frame_objs
            # ffmpeg `-frame_pts true` writes source-frame-index filenames (B5).
            frame_path = os.path.join(vid_frames_dir, f"{fid:06d}.jpg")
            if not os.path.exists(frame_path):
                # The big one. `trajectories` is indexed at the source rate
                # while the extracted filenames carry the source PTS, so at
                # frames_3fps only one annotated frame in ten has a file:
                # measured 441 of 4,395 loaded, 3,954 discarded, and
                # num_states reported 6 without mentioning the 54 lost.
                _dropped["missing_frame_file"] += 1
                continue

            pil_img = Image.open(frame_path).convert("RGB")

            if strict_slots:
                slotted = _place_by_tid(frame_objs, slot_of, num_objs, _slots)
            else:
                # The old ordering, kept as the default so every existing run
                # reproduces: bbox area descending, per frame, independently.
                ranked = sorted(frame_objs, key=_bbox_area, reverse=True)
                _slots["dropped_past_slots"] += max(0, len(ranked) - num_objs)
                _slots["placed_by_area"] += min(len(ranked), num_objs)
                slotted = ranked[:num_objs]
                slotted += [None] * (num_objs - len(slotted))

            patches, bboxes, names = [], [], []
            for i, obj in enumerate(slotted):
                if obj is None:
                    patches.append(np.zeros((_patch_size, _patch_size, 3), dtype=np.uint8))
                    bboxes.append((0, 0, 0, 0))
                    names.append(f"pad_{i}")
                    continue
                b    = obj["bbox"]
                bbox = (b["xmin"], b["ymin"], b["xmax"], b["ymax"])
                tid  = obj.get("tid")
                patches.append(_crop_object(pil_img, bbox, patch_size=_patch_size))
                bboxes.append(_scale_bbox_to_canvas(bbox, W, H))
                names.append(tid_to_cat.get(tid, f"obj{tid}"))

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
    last_load_metadata["dropped"] = dict(_dropped)
    last_load_metadata["slots"]   = dict(_slots)
    print("[vidvrd-loader] %s" % _describe_slots(_slots))
    print(f"[vidvrd-loader] category_filter={category_filter} strict={strict} "
          f"loaded {len(loaded_video_ids)}/{len(ann_files)} videos, "
          f"{len(images_list)} states")
    _kept = _dropped["annotated_frames"] - (_dropped["empty_annotation"]
                                            + _dropped["missing_frame_file"])
    if _dropped["annotated_frames"]:
        print("[vidvrd-loader] kept %d of %d annotated frames: %d had no "
              "extracted jpg, %d had no annotation%s"
              % (_kept, _dropped["annotated_frames"],
                 _dropped["missing_frame_file"], _dropped["empty_annotation"],
                 (", %d filled from a neighbour" % _dropped["filled"])
                 if _dropped["filled"] else ""))
        if _dropped["missing_frame_file"] > _kept:
            print("[vidvrd-loader] WARNING: more frames were discarded than "
                  "loaded. The trajectory index and the extracted filenames "
                  "are probably at different rates.")

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

    'sequential' compares the frame NUMBER, not only the video id. Until
    2026-09-06 it compared the id alone, so every frame this loader skipped in
    silence -- a missing jpg, an empty annotation -- became an "adjacent" pair.
    Measured at 30fps over 60 videos: 45 of 4,335 transitions spanned 16 to 271
    real frames as one action step. Those pairs are still emitted by default so
    that existing runs reproduce, but they are now counted and reported.
    Set STRICT_ADJACENCY=1 to drop them instead. See latplan/util/adjacency.py.
    """
    if mode == "sequential":
        from latplan.util.adjacency import (sequential_pairs, strict_from_env,
                                            describe)
        pairs, stats = sequential_pairs(frame_ids, strict=strict_from_env())
        print("[vidvrd-loader] %s" % describe(stats))
        last_load_metadata["transitions"] = dict(stats)
        pres = [states[i] for i, _ in pairs]
        sucs = [states[j] for _, j in pairs]
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
