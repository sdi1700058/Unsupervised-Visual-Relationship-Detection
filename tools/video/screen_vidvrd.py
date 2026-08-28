#!/usr/bin/env python3
"""Rank VidVRD clips by how much of their annotation is real.

VidVRD does not annotate every frame. `setup-dataset.py --fill-annotations`
carries the last annotated box forward into the gaps, which makes those
transitions **identical on both sides**: the boxes do not move because there
was nothing to move them. A model trained on that data spends a large fraction
of its transitions learning "nothing changed", and every evaluation window
falling inside a gap measures noise.

Measured across all 800 training clips on 2026-08-28:

    fraction of frames filled   median 0.367   p90 0.765
    motion per annotated step   median 10.11 px   p90 33.05 px
    clips needing no fill       281 of 800  (35%)
    clips more than half filled 302 of 800  (38%)

So a third of the corpus is fully annotated and needs no filling at all. That
subset is what this tool exists to find.

    python3 tools/video/screen_vidvrd.py
    python3 tools/video/screen_vidvrd.py --no-fill-only --min-disp 5 \\
        --min-frames 40 --list clean.txt

Reads annotation JSON with the standard library and numpy only — no latplan,
no frames, no keras, no cluster. It reports raw pixel displacement in the
video's own resolution rather than canvas pixels, because the question here is
which clips to bake, which is upstream of any canvas.
"""

import argparse
import glob
import json
import os
import sys


def clip_stats(doc, window=8):
    """Per-clip annotation statistics, or None when the clip has no boxes.

    `fill_frac` is the share of frames that carry no annotation and would
    therefore be invented by `--fill-annotations`. `mean_disp` is the summed
    absolute corner displacement between **consecutive annotated** frames, so
    it measures real motion and never the flat stretches fill would create.
    """
    import numpy as np

    trajectories = doc.get("trajectories") or []
    n_frames = len(trajectories)
    if n_frames == 0:
        return None

    per_object = {}
    annotated = 0
    for index, frame in enumerate(trajectories):
        if frame:
            annotated += 1
        for obj in frame:
            bbox = obj["bbox"]
            per_object.setdefault(obj["tid"], {})[index] = (
                bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"])

    if not per_object:
        return None

    displacements = []
    for frames in per_object.values():
        ordered = sorted(frames)
        for a, b in zip(ordered, ordered[1:]):
            if b == a + 1:
                displacements.append(float(np.abs(
                    np.array(frames[b], dtype=np.float64)
                    - np.array(frames[a], dtype=np.float64)).sum()))

    nonlinearity = window_nonlinearity(per_object, window=window)

    return {
        "nonlinearity": nonlinearity,
        "video_id": doc.get("video_id", "?"),
        "frames": n_frames,
        "annotated": annotated,
        "fill_frac": 1.0 - annotated / float(n_frames),
        "mean_disp": float(np.mean(displacements)) if displacements else 0.0,
        "n_objects": len(per_object),
        "width": doc.get("width", 0),
        "height": doc.get("height", 0),
    }


def window_nonlinearity(per_object, window=8):
    """How much of the motion a straight line already explains.

    `EVAL.md` §4.9 measures the constraint this exists to screen for: linear
    interpolation is the *optimal* predictor of smooth motion, so a planner can
    only beat it where change is discrete. A clip that pans smoothly is
    unwinnable however much it moves.

    For every contiguous run of `window` annotated frames, this interpolates
    between the two endpoints and measures the error against the truth,
    normalised by the object's own diagonal so the score is scale-free and
    comparable across clips. Returns the median over all windows and objects,
    or None when no window is contiguous.

    0 means interpolation is exact. Larger means more of the motion is
    something a straight line misses.
    """
    import numpy as np

    scores = []
    for track in per_object.values():
        frames = sorted(track)
        for start in range(len(frames) - window + 1):
            run = frames[start:start + window]
            if run[-1] - run[0] != window - 1:
                continue                      # not contiguous, skip
            a = np.array(track[run[0]], dtype=np.float64)
            b = np.array(track[run[-1]], dtype=np.float64)
            diag = np.hypot(max(a[2] - a[0], 1e-6), max(a[3] - a[1], 1e-6))
            for step, frame in enumerate(run[1:-1], start=1):
                t = step / float(window - 1)
                predicted = a + t * (b - a)
                truth = np.array(track[frame], dtype=np.float64)
                scores.append(float(np.abs(predicted - truth).mean()) / diag)

    return float(np.median(scores)) if scores else None


def main(argv=None):
    import numpy as np

    ap = argparse.ArgumentParser(
        description="Rank VidVRD clips by annotation density and real motion.")
    ap.add_argument("--annotations",
                    default="data/video/vidvrd/annotations/train",
                    help="directory of VidVRD annotation JSON")
    ap.add_argument("--no-fill-only", action="store_true",
                    help="keep only clips that need no filling at all")
    ap.add_argument("--min-disp", type=float, default=0.0,
                    help="minimum mean pixel displacement per annotated step")
    ap.add_argument("--min-frames", type=int, default=0,
                    help="minimum annotated frames")
    ap.add_argument("--min-nonlinearity", type=float, default=0.0,
                    help="minimum median non-linearity of the motion; a "
                         "straight line already predicts anything below it "
                         "(EVAL.md 4.9)")
    ap.add_argument("--window", type=int, default=8,
                    help="window used for the non-linearity measure")
    ap.add_argument("--sort", choices=("nonlinearity", "disp", "annotated"),
                    default="nonlinearity",
                    help="ranking key. Default is non-linearity, because that "
                         "is what decides whether a planner can beat the "
                         "linear baseline; speed alone does not (measured "
                         "Spearman between the two is only +0.59)")
    ap.add_argument("--top", type=int, default=20,
                    help="how many rows to print")
    ap.add_argument("--list", help="write the surviving video ids here")
    ap.add_argument("--csv", help="write every row here")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.annotations, "*.json")))
    if not files:
        print(f"no annotation JSON under {args.annotations}", file=sys.stderr)
        return 1

    rows = []
    for path in files:
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except (ValueError, IOError) as exc:
            print(f"skip {os.path.basename(path)}: {exc}", file=sys.stderr)
            continue
        stats = clip_stats(doc, window=args.window)
        if stats:
            rows.append(stats)

    if not rows:
        print("no clip carried any annotation", file=sys.stderr)
        return 1

    fill = np.array([r["fill_frac"] for r in rows])
    disp = np.array([r["mean_disp"] for r in rows])
    print(f"{len(rows)} clips scanned\n")
    print(f"  fraction of frames filled   median {np.median(fill):.3f}   "
          f"p90 {np.percentile(fill, 90):.3f}")
    print(f"  motion per annotated step   median {np.median(disp):.2f} px   "
          f"p90 {np.percentile(disp, 90):.2f} px")
    print(f"  clips needing no fill       {int((fill == 0).sum())} of "
          f"{len(rows)}  ({100 * (fill == 0).mean():.0f}%)")
    print(f"  clips more than half filled {int((fill > 0.5).sum())} of "
          f"{len(rows)}  ({100 * (fill > 0.5).mean():.0f}%)")
    nl = np.array([r["nonlinearity"] for r in rows
                   if r["nonlinearity"] is not None])
    if len(nl):
        print(f"  motion non-linearity        median {np.median(nl):.4f}   "
              f"p90 {np.percentile(nl, 90):.4f}   max {nl.max():.4f}")
        print("    (share of an object diagonal that a straight line between "
              "the window\n     endpoints already fails to explain; 0 means "
              "interpolation is exact)")

    kept = [r for r in rows
            if (not args.no_fill_only or r["fill_frac"] == 0.0)
            and r["mean_disp"] >= args.min_disp
            and r["annotated"] >= args.min_frames
            and (r["nonlinearity"] or 0.0) >= args.min_nonlinearity]
    keys = {"nonlinearity": lambda r: -(r["nonlinearity"] or 0.0),
            "disp": lambda r: -r["mean_disp"],
            "annotated": lambda r: -r["annotated"]}
    kept.sort(key=keys[args.sort])

    print(f"\n{len(kept)} clips pass the filter. "
          f"Transitions available: {sum(r['annotated'] - 1 for r in kept)}\n")
    print(f"{'video_id':32} {'annot':>6} {'fill':>6} {'objs':>5} "
          f"{'px/step':>8} {'nonlin':>8}")
    for r in kept[:args.top]:
        nl = "     n/a" if r["nonlinearity"] is None else f"{r['nonlinearity']:>8.3f}"
        print(f"{r['video_id']:32} {r['annotated']:>6} "
              f"{r['fill_frac']:>6.2f} {r['n_objects']:>5} "
              f"{r['mean_disp']:>8.1f} {nl}")

    if args.list:
        os.makedirs(os.path.dirname(args.list) or ".", exist_ok=True)
        with open(args.list, "w") as fh:
            for r in kept:
                fh.write(r["video_id"] + "\n")
        print(f"\nwrote {len(kept)} video ids to {args.list}")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w") as fh:
            fh.write("video_id,frames,annotated,fill_frac,mean_disp,"
                     "nonlinearity,n_objects,width,height\n")
            for r in rows:
                nl = "" if r["nonlinearity"] is None else f"{r['nonlinearity']:.5f}"
                fh.write(f"{r['video_id']},{r['frames']},{r['annotated']},"
                         f"{r['fill_frac']:.4f},{r['mean_disp']:.4f},{nl},"
                         f"{r['n_objects']},{r['width']},{r['height']}\n")
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
