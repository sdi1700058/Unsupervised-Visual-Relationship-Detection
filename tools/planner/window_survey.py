#!/usr/bin/env python3
"""Find the window at which the interpolation metric becomes winnable.

`mse_ratio` compares the planner against linear interpolation between the two
endpoint frames. Below a certain window that comparison is unwinnable for any
model, because discretising the boxes into the decoder's bbox bins costs more
error than the straight line does. `EVAL.md §4.2` derives it; this measures
where the crossover actually falls, across many clips rather than one.

Two clips are an anecdote. This reports a distribution, so the thesis can say
what window VidVRD needs rather than what one dog video needed.

    python3 tools/planner/window_survey.py                 # all train clips
    python3 tools/planner/window_survey.py --limit 100 --max-objects 5
    python3 tools/planner/window_survey.py --csv eval/window_survey.csv

Reads annotation JSON only — no extracted frames, no keras, no cluster. The
canvas scaling comes from `puzzle_labeled_objects` (SPEC V5), so this needs the
project environment.
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from tools.planner.oracle import (                       # noqa: E402
    DEFAULT_BINS_X, DEFAULT_BINS_Y, boxes_from_vidvrd, boxes_to_latents,
    round_trip_error)
from tools.planner.common.metrics import moving_gt_steps   # noqa: E402
from tools.planner.common.windows import linear_interp_bboxes  # noqa: E402

WINDOWS = (2, 4, 6, 8, 10, 12, 16, 20, 24, 32)


def survey_clip(path, num_objs, bins_x, bins_y, windows=WINDOWS):
    """Return per-clip statistics, or None when the clip is too short."""
    boxes, meta = boxes_from_vidvrd(path, num_objs=num_objs, fill=True)
    n = len(boxes)
    if n < min(windows) + 1:
        return None

    floor = round_trip_error(boxes, bins_x, bins_y)
    z = boxes_to_latents(boxes, bins_x, bins_y)
    distinct = len(np.unique(z, axis=0))

    # Two properties of the annotations themselves, independent of any bin
    # count. `duplicate_frac` below can be lowered by finer bins; these cannot,
    # because they measure whether the objects moved at all. Measured on
    # 00005005: 56% static even at one bin per pixel, and one of three slots
    # empty for the whole clip. See DATASETS.md.
    moving = moving_gt_steps(boxes)
    static_frac = 1.0 - moving / float(max(n - 1, 1))
    areas = (np.clip(boxes[..., 2] - boxes[..., 0], 0, None)
             * np.clip(boxes[..., 3] - boxes[..., 1], 0, None))
    empty_slots = int((areas.max(axis=0) == 0).sum())

    crossover, per_window = None, {}
    for w in windows:
        if w >= n:
            break
        # Average over several starts so one lucky window does not decide it.
        ratios = []
        for start in range(0, n - w, max(1, (n - w) // 8)):
            mid = list(range(start + 1, start + w))
            base = linear_interp_bboxes(boxes[start], boxes[start + w],
                                        len(mid))
            e = base - boxes[mid]
            mse = float((e * e).sum(axis=-1).mean())
            if mse > 0:
                ratios.append(floor / mse)
        if not ratios:
            continue
        per_window[w] = float(np.median(ratios))
        if crossover is None and per_window[w] < 1.0:
            crossover = w

    return {
        "video_id": meta["video_id"],
        "frames": n,
        "distinct": distinct,
        "duplicate_frac": 1.0 - distinct / float(n),
        "static_frac": static_frac,
        "empty_slots": empty_slots,
        "floor": floor,
        "crossover": crossover,
        "per_window": per_window,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Measure where mse_ratio becomes winnable, across clips.")
    ap.add_argument("--annotations",
                    default="data/video/vidvrd/annotations/train",
                    help="directory of VidVRD annotation JSON")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-objects", type=int, default=3)
    ap.add_argument("--bins", type=int, default=None,
                    help=f"bins per axis; default matches the decoder "
                         f"({DEFAULT_BINS_X} x {DEFAULT_BINS_Y})")
    ap.add_argument("--csv", default=None, help="write per-clip rows here")
    args = ap.parse_args(argv)

    bins_x = args.bins or DEFAULT_BINS_X
    bins_y = args.bins or DEFAULT_BINS_Y

    files = sorted(glob.glob(os.path.join(args.annotations, "*.json")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"no annotation JSON under {args.annotations}")

    rows, skipped = [], 0
    for i, f in enumerate(files, 1):
        try:
            r = survey_clip(f, args.max_objects, bins_x, bins_y)
        except (SystemExit, KeyError, ValueError):
            skipped += 1
            continue
        if r is None:
            skipped += 1
            continue
        rows.append(r)
        if i % 100 == 0:
            print(f"  ...{i}/{len(files)}", file=sys.stderr)

    if not rows:
        sys.exit("no clip produced a usable trajectory")

    print(f"{len(rows)} clips surveyed, {skipped} skipped, "
          f"bins {bins_x}x{bins_y}, {args.max_objects} object slots\n")

    frames = np.array([r["frames"] for r in rows])
    floors = np.array([r["floor"] for r in rows])
    dups = np.array([r["duplicate_frac"] for r in rows])
    statics = np.array([r["static_frac"] for r in rows])
    empties = np.array([r["empty_slots"] for r in rows], dtype=float)
    print(f"{'':22}{'median':>10}{'mean':>10}{'p10':>10}{'p90':>10}")
    for name, a in (("annotated frames", frames),
                    ("quantisation floor", floors),
                    ("duplicate latents", dups),
                    ("static frame pairs", statics),
                    ("empty object slots", empties)):
        print(f"{name:22}{np.median(a):>10.2f}{a.mean():>10.2f}"
              f"{np.percentile(a, 10):>10.2f}{np.percentile(a, 90):>10.2f}")

    solved = [r["crossover"] for r in rows if r["crossover"] is not None]
    never = len(rows) - len(solved)
    print(f"\ncrossover window — the smallest window where the quantisation "
          f"floor drops below\nthe linear-interpolation baseline, so "
          f"mse_ratio < 1 becomes reachable:\n")
    if solved:
        s = np.array(solved)
        print(f"  median {int(np.median(s))}, mean {s.mean():.1f}, "
              f"p90 {int(np.percentile(s, 90))}, max {int(s.max())}")
        for w in WINDOWS:
            c = int((s <= w).sum())
            if c:
                print(f"    window >= {w:>2}: {c:>4} clips "
                      f"({100.0 * c / len(rows):.0f}%) winnable")
    print(f"    never within {max(WINDOWS)}: {never} clips "
          f"({100.0 * never / len(rows):.0f}%)")

    # Clip screening. `duplicate_frac` falls when the bins are refined;
    # `static_frac` does not, because it counts frames where the annotated
    # boxes are bit-identical. A clip that is mostly static teaches "nothing
    # changed" and cannot be scored, whatever the representation.
    mostly_static = int((statics >= 0.5).sum())
    print(f"\nclip screening:")
    print(f"  {mostly_static} clips ({100.0 * mostly_static / len(rows):.0f}%) "
          f"are static in at least half their frame pairs")
    print(f"  {int((empties > 0).sum())} clips hold fewer objects than "
          f"--max-objects, so at least one slot is empty throughout")
    worth = [r for r in rows
             if r["static_frac"] < 0.5 and r["crossover"] is not None]
    print(f"  {len(worth)} clips ({100.0 * len(worth) / len(rows):.0f}%) both "
          f"move and cross — these are the ones worth training and scoring on")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w") as fh:
            fh.write("video_id,frames,distinct,duplicate_frac,static_frac,"
                     "empty_slots,floor,crossover\n")
            for r in rows:
                fh.write(f"{r['video_id']},{r['frames']},{r['distinct']},"
                         f"{r['duplicate_frac']:.4f},{r['static_frac']:.4f},"
                         f"{r['empty_slots']},{r['floor']:.4f},"
                         f"{r['crossover'] if r['crossover'] else ''}\n")
        print(f"\nper-clip rows written to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
