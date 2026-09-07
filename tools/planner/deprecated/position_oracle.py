#!/usr/bin/env python3
"""The position oracle. DEPRECATED 2026-09-07 -- kept because three pipelines
still call it.

**What it is.** Builds a planner export whose latents are ground-truth
bounding boxes, quantised. That answers "could a classical planner do frame
interpolation from a binary state at all", which is a ceiling on the *planner*.

**Why it is deprecated.** The supervisors asked for an oracle of the
**relations**, not of the positions -- "έστω πως το FOSAE παρήγαγε τις ιδανικές
σχέσεις, τί θα έβγαζε το planning". That is
`tools/planner/relation_oracle.py`. This one answers a question nobody asked,
and having two things called "the oracle" was actively confusing.

**Why it is still here rather than deleted.** Three pipelines build their
oracle exports with it and would break:

    experiments/M_evaluation_methods/build_oracle_dataset.sh
    experiments/M6_effect_determinism/run.sh
    sh/videonet_pipeline.sh

Moving those onto the relation oracle is a real change to what they measure,
not a path edit, so it is a decision rather than a refactor.

**Do not add callers.** The geometry and the per-dataset box readers this uses
live in `tools/planner/box_geometry.py`, which is not deprecated -- those are
used throughout and were only ever called "oracle" by accident of file name.

    python3 tools/planner/deprecated/position_oracle.py <annotation.json> \
        --out eval/exports/oracle-x.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
    os.pardir)))

from tools.planner.box_geometry import (          # noqa: E402
    CANVAS_H, CANVAS_W, DEFAULT_BINS_X, DEFAULT_BINS_Y,
    bits_per_object, boxes_to_latents, latents_to_boxes, round_trip_error,
    boxes_from_vidvrd, boxes_from_actiongenome_clip,
    boxes_from_something_else, _load_boxes)


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
        # The bin count travels with the export, because the quantisation
        # floor is a property of the encoding and not of the canvas. Before
        # this was written, `common/harness.py` computed the floor at the
        # decoder's 60x40 whatever the export used, so `floor_ratio` on
        # `eval/exports/oracle-real-5005-b16.npz` would have divided a
        # 16x16 error by a 60x40 floor: 335.07 against 19.94, a factor of
        # 16.8. A reader of `model_name` can see the bins; arithmetic cannot.
        "bins_x": np.int64(bins_x),
        "bins_y": np.int64(bins_y),
        "encoding": np.str_(encoding),
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
