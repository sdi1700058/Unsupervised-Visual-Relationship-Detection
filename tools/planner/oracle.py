#!/usr/bin/env python3
"""Build a planner export whose latents are perfect by construction.

The evaluation has two halves that a single number cannot separate. When
`mse_ratio` comes back bad we cannot tell whether FOSAE's relations are
poor or whether a classical planner simply cannot do frame interpolation
from a binary state at all. This answers the second half on its own:

    Suppose FOSAE produced ideal relations. What would planning output?

So we skip the encoder. The latents here are a lossless discretisation of
the annotated bounding boxes — no learning, no reconstruction error beyond
the quantisation step. Anything the planner fails to do on these latents,
it would also fail to do on a perfect model, and no amount of training
would fix it. The resulting `mse_ratio` is the **ceiling**: every trained
model is measured against it rather than against zero.

The encoding is deliberately the same shape the real decoder emits. In
`common/decode.py` the bbox block of a FOSAE feature vector is four
one-hot runs — x1 over X bins, y1 over Y bins, then x2 and y2 — with
X, Y = PICSIZE // 5, i.e. 60 and 40 on the 200x300 canvas. This module
writes exactly that, so the oracle is an oracle *of the representation the
model is actually trying to learn*, not of some easier one.

One-hot rather than a binary or Gray code because Hamming distance has to
mean something: moving one coordinate by one bin flips exactly two bits.
A binary code would flip up to log2(bins) bits for the same motion and the
mined action set would be far larger than the real transition structure.

The output is an ordinary export, so `plan_video.py`, `harness.py` and all
three planning methods consume it unchanged:

    python3 tools/planner/oracle.py data/npz/.../dog-....npz \\
        --out eval/exports/oracle-dog.npz --bins 8

    python3 tools/planner/plan_video.py eval/exports/oracle-dog.npz \\
        --method pddl --init 0 --goal 4
"""

import argparse
import sys

import numpy as np

# The canvas the loaders rasterise onto (puzzle_labeled_objects.CANVAS_*).
# Duplicated as plain numbers so this module never imports keras.
CANVAS_H = 200
CANVAS_W = 300

# The real decoder's bbox resolution: PICSIZE // 5.
DEFAULT_BINS_X = CANVAS_W // 5
DEFAULT_BINS_Y = CANVAS_H // 5


def quantise(values, n_bins, extent):
    """Map pixel coordinates to bin indices in [0, n_bins).

    `extent` is the canvas size along that axis. The top edge is inclusive,
    so a coordinate sitting exactly on the boundary lands in the last bin
    rather than out of range.
    """
    values = np.asarray(values, dtype=np.float64)
    idx = np.floor(values / float(extent) * n_bins)
    return np.clip(idx, 0, n_bins - 1).astype(np.int64)


def dequantise(idx, n_bins, extent):
    """Map bin indices back to the pixel coordinate at each bin's centre."""
    idx = np.asarray(idx, dtype=np.float64)
    return (idx + 0.5) * (float(extent) / n_bins)


def _code_width(n_bins):
    """Bits needed to hold a bin index in a plain binary code."""
    return max(1, int(np.ceil(np.log2(max(n_bins, 2)))))


def bits_per_object(bins_x, bins_y, encoding="onehot"):
    """Latent bits one object occupies under the chosen positional code.

    `onehot` matches `common/decode.py`: four runs, x1 | y1 | x2 | y2, one bit
    per bin. `binary` writes each bin index as a plain binary number, which is
    far narrower and — measured in `EVAL.md` — far better behaved as an action
    model: one-hot gives "move one bin" a different effect at all 31 start
    positions, binary gives it 5.
    """
    if encoding == "onehot":
        return 2 * bins_x + 2 * bins_y
    if encoding == "binary":
        return 2 * _code_width(bins_x) + 2 * _code_width(bins_y)
    raise ValueError("unknown encoding %r; use onehot or binary" % encoding)


def boxes_to_latents(boxes, bins_x=DEFAULT_BINS_X, bins_y=DEFAULT_BINS_Y,
                     width=CANVAS_W, height=CANVAS_H, encoding="onehot"):
    """Encode (N, num_objs, 4) pixel boxes as (N, num_objs*bits) binary.

    Each object contributes four one-hot runs laid out x1 | y1 | x2 | y2,
    matching `common/decode.py`.

    A padded slot — an all-zero box, which is how the loaders mark "no
    object here" — encodes to an all-zero block rather than to bin 0 of
    every coordinate. Otherwise the oracle would invent an object in the
    top-left corner of every frame and the planner would have to move it.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must be (N, num_objs, 4); got {boxes.shape}")

    n_states, n_objs, _ = boxes.shape
    per_obj = bits_per_object(bins_x, bins_y, encoding)
    out = np.zeros((n_states, n_objs * per_obj), dtype=np.int8)

    occupied = np.abs(boxes).sum(axis=-1) > 0        # (N, num_objs)

    runs = (
        (0, bins_x, width),                          # x1
        (1, bins_y, height),                         # y1
        (2, bins_x, width),                          # x2
        (3, bins_y, height),                         # y2
    )

    for o in range(n_objs):
        base = o * per_obj
        offset = 0
        rows = np.nonzero(occupied[:, o])[0]
        for coord, n_bins, extent in runs:
            idx = quantise(boxes[:, o, coord], n_bins, extent)
            if encoding == "onehot":
                out[rows, base + offset + idx[rows]] = 1
                offset += n_bins
            else:
                width_bits = _code_width(n_bins)
                # +1 so that bin 0 is not the all-zero word, which would be
                # indistinguishable from an absent slot.
                for bit in range(width_bits):
                    set_rows = rows[(((idx[rows] + 1) >> bit) & 1) == 1]
                    out[set_rows, base + offset + bit] = 1
                offset += width_bits
    return out


def latents_to_boxes(latents, n_objs, bins_x=DEFAULT_BINS_X,
                     bins_y=DEFAULT_BINS_Y, width=CANVAS_W, height=CANVAS_H,
                     encoding="onehot"):
    """Decode binary latents back to (N, num_objs, 4) pixel boxes.

    argmax per run, exactly as `common/decode.py` does, so a run with no
    bit set decodes to bin 0 — and an all-zero object block therefore comes
    back as an all-zero box, round-tripping the padding.
    """
    latents = np.asarray(latents)
    if latents.ndim == 1:
        latents = latents[None, :]
    n_states = latents.shape[0]
    per_obj = bits_per_object(bins_x, bins_y, encoding)
    expected = n_objs * per_obj
    if latents.shape[1] != expected:
        raise ValueError(
            f"latent width {latents.shape[1]} does not match {n_objs} objects "
            f"at {per_obj} bits each ({expected})")

    boxes = np.zeros((n_states, n_objs, 4), dtype=np.float32)
    runs = (
        (0, bins_x, width),
        (1, bins_y, height),
        (2, bins_x, width),
        (3, bins_y, height),
    )
    for o in range(n_objs):
        base = o * per_obj
        block = latents[:, base:base + per_obj]
        empty = block.sum(axis=1) == 0
        offset = 0
        for coord, n_bins, extent in runs:
            if encoding == "onehot":
                run = block[:, offset:offset + n_bins]
                idx = run.argmax(axis=1)
                offset += n_bins
            else:
                width_bits = _code_width(n_bins)
                run = block[:, offset:offset + width_bits]
                word = np.zeros(n_states, dtype=np.int64)
                for bit in range(width_bits):
                    word |= (run[:, bit].astype(np.int64) << bit)
                idx = np.clip(word - 1, 0, n_bins - 1)
                offset += width_bits
            boxes[:, o, coord] = dequantise(idx, n_bins, extent)
        boxes[empty, o, :] = 0.0
    return boxes


def round_trip_error(boxes, bins_x=DEFAULT_BINS_X, bins_y=DEFAULT_BINS_Y,
                     width=CANVAS_W, height=CANVAS_H, encoding="onehot"):
    """Mean squared pixel error introduced by the discretisation alone.

    This is the floor the oracle cannot go below, and it is worth reporting
    next to the planner's error: if the two are close, the planner is doing
    as well as this representation permits.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    z = boxes_to_latents(boxes, bins_x, bins_y, width, height,
                         encoding=encoding)
    back = latents_to_boxes(z, boxes.shape[1], bins_x, bins_y, width, height,
                            encoding=encoding)
    d = back.astype(np.float64) - boxes
    return float((d * d).sum(axis=-1).mean())


def load_canvas_scaler():
    """Return `(_scale_bbox_to_canvas, CANVAS_W, CANVAS_H)` from the loader.

    `puzzle_labeled_objects` needs only os, json, numpy and PIL, but importing
    it through the package runs `latplan/__init__.py`, which pulls in
    TensorFlow. That blocks every environment without the training stack —
    including the one this module's own docstring promises to work in.

    So the package import is tried first, and on failure the module is loaded
    directly from its file. Either way the **same function** is imported, never
    copied, which SPEC V5 requires: the canvas geometry must have one
    definition.
    """
    try:
        from latplan.puzzles.puzzle_labeled_objects import (
            _scale_bbox_to_canvas, CANVAS_W, CANVAS_H)
        return _scale_bbox_to_canvas, CANVAS_W, CANVAS_H
    except ImportError:
        pass

    import importlib.util
    import os

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "latplan", "puzzles", "puzzle_labeled_objects.py")
    if not os.path.exists(path):
        raise SystemExit(
            "cannot find latplan/puzzles/puzzle_labeled_objects.py, which "
            "defines the canvas geometry. Run from the repository root.")

    spec = importlib.util.spec_from_file_location("_plo_direct", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise SystemExit(
            "reading VidVRD annotations needs numpy and pillow, and one is "
            "missing (%s). The rest of oracle.py needs only numpy." % exc)
    return (module._scale_bbox_to_canvas, module.CANVAS_W, module.CANVAS_H)


def boxes_from_something_else_frames(frames, width, height, num_objs=3):
    """Convert one Something-Else video's annotations into canvas boxes.

    Something-Else adds per-frame boxes to Something-Something v2: 180,049
    videos, 8.18M annotated frames, MIT licensed, and every clip an instance of
    a named action (`DATASETS_CONSIDERED.md`). Each label carries `box2d` with
    `x1, y1, x2, y2` and a `standard_category` that is a **per-video slot id** —
    `0000`, `0001`, `hand`.

    Columns are ordered by that slot id, not by list position. The order inside
    a frame's `labels` is not stable, so keying on position would let a slot
    swap between frames and silently corrupt every trajectory.

    A slot absent from a frame stays all-zero, which the metrics read as "not
    in this frame" rather than "at the origin". That case is normal here: on
    the median sample clip the least-present slot covers only 78% of frames.

    Returns (boxes, meta) with boxes shaped (n_frames, num_objs, 4).
    """
    import numpy as np

    scale, canvas_w, canvas_h = load_canvas_scaler()

    slots = []
    for frame in frames:
        for label in frame.get("labels") or []:
            cat = label.get("standard_category")
            if cat is not None and cat not in slots:
                slots.append(cat)
    # Objects first in id order, hand last: the hand is a participant in nearly
    # every clip, so putting it last keeps object slots stable as num_objs
    # varies.
    slots.sort(key=lambda c: (c == "hand", c))
    slots = slots[:num_objs]

    boxes = np.zeros((len(frames), num_objs, 4), dtype=np.float32)
    absent = 0
    for i, frame in enumerate(frames):
        present = {}
        for label in frame.get("labels") or []:
            cat = label.get("standard_category")
            if cat in slots:
                b = label["box2d"]
                present[cat] = scale([b["x1"], b["y1"], b["x2"], b["y2"]],
                                     width, height)
        for j, slot in enumerate(slots):
            if slot in present:
                boxes[i, j] = present[slot]
            else:
                absent += 1

    return boxes, {"frames": len(frames), "slots": slots, "absent": absent,
                   "source_size": (width, height),
                   "canvas": (canvas_w, canvas_h)}


def boxes_from_something_else(path, video_id=None, num_objs=3,
                              width=None, height=None):
    """Read one video out of a Something-Else annotation file.

    The released annotations are a dict mapping video id to a list of
    per-frame records. Something-Something v2 frames are 240 pixels high with
    a variable width; the sample records carry no size, so `width`/`height`
    default to the 427x240 that the release documents.
    """
    import json

    with open(path) as fh:
        data = json.load(fh)
    if video_id is None:
        video_id = sorted(data)[0]
    if video_id not in data:
        raise SystemExit("video %s is not in %s (it holds %d videos)"
                         % (video_id, path, len(data)))

    boxes, meta = boxes_from_something_else_frames(
        data[video_id], width or 427, height or 240, num_objs=num_objs)
    meta["video_id"] = video_id
    return boxes, meta


def boxes_from_vidvrd(ann_path, num_objs=3, fill=True):
    """Read one VidVRD annotation JSON into canvas-space boxes.

    Lets the ceiling be measured on real trajectories rather than on a
    synthetic curve, without needing the extracted frames, keras, or the
    cluster. Only the annotation JSON is read.

    The canvas scaling comes from `puzzle_labeled_objects` rather than being
    copied here (SPEC V5). That import pulls the latplan package chain, so
    this one function needs the project environment even though the rest of
    the module needs only numpy.

    `fill` carries the last non-empty frame forward through frames the
    dataset left unannotated, matching `--fill-annotations` in the loader.

    Returns (boxes, meta) with boxes shaped (n_frames, num_objs, 4).
    """
    import json

    _scale_bbox_to_canvas, W, H = load_canvas_scaler()

    with open(ann_path) as f:
        ann = json.load(f)

    src_w, src_h = ann["width"], ann["height"]
    trajectories = ann.get("trajectories", [])

    rows, last = [], None
    for frame in trajectories:
        objs = frame if frame else None
        if objs is None:
            if fill and last is not None:
                objs = last
            else:
                continue
        else:
            last = objs

        # Biggest boxes first, so the slot assignment is stable frame to
        # frame the same way the loader orders them.
        def area(o):
            b = o["bbox"]
            return (b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"])

        chosen = sorted(objs, key=area, reverse=True)[:num_objs]
        row = np.zeros((num_objs, 4), dtype=np.float32)
        for i, o in enumerate(chosen):
            b = o["bbox"]
            row[i] = _scale_bbox_to_canvas(
                (b["xmin"], b["ymin"], b["xmax"], b["ymax"]), src_w, src_h)
        rows.append(row)

    if not rows:
        raise SystemExit(f"{ann_path}: no annotated frames")

    meta = {"video_id": ann.get("video_id"), "fps": ann.get("fps"),
            "frames": len(rows), "source_size": (src_w, src_h)}
    return np.stack(rows), meta


def _load_boxes(path):
    """Return (gt_boxes, frame_ids) from either an export or a dataset npz."""
    # allow_pickle is needed only for frame_ids, which numpy stores as an
    # object array. The file is one we baked ourselves under data/npz or
    # eval/exports, never a downloaded artefact.
    data = np.load(path, allow_pickle=True)
    files = set(data.files)

    if "gt_boxes" in files:                       # already a planner export
        boxes = data["gt_boxes"]
        frame_ids = data["frame_ids"] if "frame_ids" in files else None
    elif "bboxes" in files:                       # a baked dataset npz
        boxes = data["bboxes"]
        frame_ids = data["frame_ids"] if "frame_ids" in files else None
    else:
        raise SystemExit(
            f"{path}: expected a planner export (gt_boxes) or a baked dataset "
            f"npz (bboxes); found {sorted(files)}")

    if frame_ids is not None:
        frame_ids = np.asarray([str(f) for f in frame_ids.tolist()])
    return np.asarray(boxes, dtype=np.float32), frame_ids


def build_export(boxes, out_path, bins_x=DEFAULT_BINS_X,
                 bins_y=DEFAULT_BINS_Y, width=CANVAS_W, height=CANVAS_H,
                 frame_ids=None, dedupe=True, encoding="onehot"):
    """Write an oracle export and return its summary."""
    latents = boxes_to_latents(boxes, bins_x, bins_y, width, height,
                               encoding=encoding)
    decoded = latents_to_boxes(latents, boxes.shape[1], bins_x, bins_y,
                               width, height, encoding=encoding)

    pre, suc = latents[:-1], latents[1:]
    if len(pre):
        actions = np.concatenate([pre, suc], axis=1)
        if dedupe:
            # Consecutive frames often repeat, especially with filled
            # annotations, and a duplicate transition adds nothing to the
            # action set while costing the PDDL writer a row.
            _, keep = np.unique(actions, axis=0, return_index=True)
            actions = actions[np.sort(keep)]
    else:
        actions = np.zeros((0, latents.shape[1] * 2), dtype=np.int8)

    payload = {
        "latents": latents.astype(np.int8),
        "gt_boxes": boxes.astype(np.float32),
        "decoded_boxes": decoded.astype(np.float32),
        "actions": actions.astype(np.int8),
        "n_bits": np.int64(latents.shape[1]),
        "model_name": np.str_(f"oracle-bins{bins_x}x{bins_y}"),
    }
    if frame_ids is not None:
        payload["frame_ids"] = np.asarray(frame_ids)

    np.savez_compressed(out_path, **payload)

    return {
        "states": int(len(latents)),
        "objects": int(boxes.shape[1]),
        "n_bits": int(latents.shape[1]),
        "distinct_latents": int(len(np.unique(latents, axis=0))),
        "transitions": int(len(actions)),
        "quantisation_mse": round_trip_error(boxes, bins_x, bins_y,
                                             width, height,
                                             encoding=encoding),
        "encoding": encoding,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build a planner export from ground-truth boxes, with no "
                    "model in the loop, to measure the planning ceiling.")
    ap.add_argument("source",
                    help="a baked dataset npz (bboxes), a planner export "
                         "(gt_boxes), or a VidVRD annotation .json")
    ap.add_argument("--out", required=True, help="path for the oracle export")
    ap.add_argument("--bins", type=int, default=None,
                    help="bins per axis. Default matches the real decoder: "
                         f"{DEFAULT_BINS_X} in x, {DEFAULT_BINS_Y} in y. A "
                         "smaller value shrinks the PDDL state space, at the "
                         "cost of coarser boxes.")
    ap.add_argument("--max-objects", type=int, default=3,
                    help="object slots when reading a VidVRD annotation json")
    ap.add_argument("--no-fill", action="store_true",
                    help="do not carry the last annotated frame forward")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N states")
    ap.add_argument("--encoding", choices=("onehot", "binary"),
                    default="onehot",
                    help="positional code. onehot matches common/decode.py; "
                         "binary is far narrower and gives an action a "
                         "repeatable effect far more often (EVAL.md)")
    ap.add_argument("--something-else", action="store_true",
                    help="read the source as a Something-Else annotation file "
                         "rather than a VidVRD one")
    ap.add_argument("--video-id", default=None,
                    help="which video to read from a Something-Else file; "
                         "defaults to the first")
    args = ap.parse_args(argv)

    bins_x = args.bins if args.bins else DEFAULT_BINS_X
    bins_y = args.bins if args.bins else DEFAULT_BINS_Y

    if args.source.endswith(".json") and args.something_else:
        boxes, ann_meta = boxes_from_something_else(
            args.source, video_id=args.video_id,
            num_objs=args.max_objects)
        frame_ids = None
        print(f"read {ann_meta['frames']} frames of Something-Else video "
              f"{ann_meta['video_id']}; slots {ann_meta['slots']}, "
              f"{ann_meta['absent']} slot-frames absent")
    elif args.source.endswith(".json"):
        boxes, ann_meta = boxes_from_vidvrd(args.source,
                                            num_objs=args.max_objects,
                                            fill=not args.no_fill)
        frame_ids = None
        print(f"read {ann_meta['frames']} annotated frames from "
              f"{ann_meta['video_id']} ({ann_meta['source_size'][0]}x"
              f"{ann_meta['source_size'][1]})")
    else:
        boxes, frame_ids = _load_boxes(args.source)
    if args.limit:
        boxes = boxes[:args.limit]
        if frame_ids is not None:
            frame_ids = frame_ids[:args.limit]

    info = build_export(boxes, args.out, bins_x, bins_y,
                        frame_ids=frame_ids, encoding=args.encoding)

    print(f"wrote {args.out}")
    for k in ("states", "objects", "encoding", "n_bits", "distinct_latents",
              "transitions", "quantisation_mse"):
        print(f"  {k:<18}{info[k]}")
    if info["distinct_latents"] < info["states"]:
        dupes = info["states"] - info["distinct_latents"]
        print(f"  note: {dupes} states share a latent with another state; "
              "at this bin count the boxes moved less than one bin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
