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


def bits_per_object(bins_x, bins_y):
    return 2 * bins_x + 2 * bins_y


def boxes_to_latents(boxes, bins_x=DEFAULT_BINS_X, bins_y=DEFAULT_BINS_Y,
                     width=CANVAS_W, height=CANVAS_H):
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
    per_obj = bits_per_object(bins_x, bins_y)
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
        for coord, n_bins, extent in runs:
            idx = quantise(boxes[:, o, coord], n_bins, extent)
            rows = np.nonzero(occupied[:, o])[0]
            out[rows, base + offset + idx[rows]] = 1
            offset += n_bins
    return out


def latents_to_boxes(latents, n_objs, bins_x=DEFAULT_BINS_X,
                     bins_y=DEFAULT_BINS_Y, width=CANVAS_W, height=CANVAS_H):
    """Decode binary latents back to (N, num_objs, 4) pixel boxes.

    argmax per run, exactly as `common/decode.py` does, so a run with no
    bit set decodes to bin 0 — and an all-zero object block therefore comes
    back as an all-zero box, round-tripping the padding.
    """
    latents = np.asarray(latents)
    if latents.ndim == 1:
        latents = latents[None, :]
    n_states = latents.shape[0]
    per_obj = bits_per_object(bins_x, bins_y)
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
            run = block[:, offset:offset + n_bins]
            boxes[:, o, coord] = dequantise(run.argmax(axis=1), n_bins, extent)
            offset += n_bins
        boxes[empty, o, :] = 0.0
    return boxes


def round_trip_error(boxes, bins_x=DEFAULT_BINS_X, bins_y=DEFAULT_BINS_Y,
                     width=CANVAS_W, height=CANVAS_H):
    """Mean squared pixel error introduced by the discretisation alone.

    This is the floor the oracle cannot go below, and it is worth reporting
    next to the planner's error: if the two are close, the planner is doing
    as well as this representation permits.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    z = boxes_to_latents(boxes, bins_x, bins_y, width, height)
    back = latents_to_boxes(z, boxes.shape[1], bins_x, bins_y, width, height)
    d = back.astype(np.float64) - boxes
    return float((d * d).sum(axis=-1).mean())


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
                 frame_ids=None, dedupe=True):
    """Write an oracle export and return its summary."""
    latents = boxes_to_latents(boxes, bins_x, bins_y, width, height)
    decoded = latents_to_boxes(latents, boxes.shape[1], bins_x, bins_y,
                               width, height)

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
                                             width, height),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build a planner export from ground-truth boxes, with no "
                    "model in the loop, to measure the planning ceiling.")
    ap.add_argument("source",
                    help="a baked dataset npz (bboxes) or a planner export "
                         "(gt_boxes)")
    ap.add_argument("--out", required=True, help="path for the oracle export")
    ap.add_argument("--bins", type=int, default=None,
                    help="bins per axis. Default matches the real decoder: "
                         f"{DEFAULT_BINS_X} in x, {DEFAULT_BINS_Y} in y. A "
                         "smaller value shrinks the PDDL state space, at the "
                         "cost of coarser boxes.")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N states")
    args = ap.parse_args(argv)

    bins_x = args.bins if args.bins else DEFAULT_BINS_X
    bins_y = args.bins if args.bins else DEFAULT_BINS_Y

    boxes, frame_ids = _load_boxes(args.source)
    if args.limit:
        boxes = boxes[:args.limit]
        if frame_ids is not None:
            frame_ids = frame_ids[:args.limit]

    info = build_export(boxes, args.out, bins_x, bins_y,
                        frame_ids=frame_ids)

    print(f"wrote {args.out}")
    for k in ("states", "objects", "n_bits", "distinct_latents",
              "transitions", "quantisation_mse"):
        print(f"  {k:<18}{info[k]}")
    if info["distinct_latents"] < info["states"]:
        dupes = info["states"] - info["distinct_latents"]
        print(f"  note: {dupes} states share a latent with another state; "
              "at this bin count the boxes moved less than one bin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
