#!/usr/bin/env python3
"""Screen VidOR for clips a planner could beat a straight line on.

VidOR uses the **identical annotation format to VidVRD**, by the same author, so
the criterion in `screen_vidvrd.window_crossover` applies unchanged and this
module is a reader plus a loop rather than new science.

What the corpus measures out at, on a 500-clip sample of the 7,000 training
files (2026-09-04):

| | VidVRD | VidOR |
|---|---|---|
| annotated-frame fraction, median | 0.367 | **1.000** |
| frames per clip, median | 128 | **810** |
| objects per clip, median | 2 | **4** |
| winnable | 56% at window 8 | **62% at window 16**, 25% at window 8 |

**Window 16, not 8.** VidOR clips are six times longer, so consecutive frames
move less and a straight line fits better locally. At window 8 the median
crossover is 3.088 and only a quarter of clips are winnable; at 16 it is 0.655
and nearly two thirds are. Screening at the wrong window is a mistake this
project has already made once (`SPEC.md` V37).

**A caution that has not been settled.** The first VidOR clip scored by the
planner ranked 12th of 315 on this screen and lost every window. The screen
uses every annotated object in source pixels; the oracle export keeps three
objects in canvas space with absent slots, so the two do not measure the same
trajectories. Until the rank correlation between this screen and achieved
`mse_ratio` has been measured, treat the winnable fraction as a property of the
criterion rather than a promise about the planner. Recorded as `WORKPLAN.json`
question Q6.

    python3 tools/video/screen_vidor.py --window 16 --sample 500
    python3 tools/video/screen_vidor.py --list eval/vidor_winnable_w16.txt

Standard library plus numpy, and pillow by way of the canvas scaler.
"""

import argparse
import glob
import json
import os
import random
import statistics as st
import sys

# Importable as a module and runnable as a script from the repository root,
# which is how sh/dataset.sh invokes it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir))


ANN_GLOB = "data/video/vidor/annotations/training/*/*.json"
OUT_DIR = "eval/datasets"


def read_tracks(path):
    """`({tid: {frame: [x1,y1,x2,y2]}}, width, height, frames)` in source pixels."""
    with open(path) as handle:
        clip = json.load(handle)
    trajectories = clip.get("trajectories") or []
    tracks = {}
    for index, frame in enumerate(trajectories):
        for box in frame or []:
            b = box["bbox"]
            tracks.setdefault(box["tid"], {})[index] = [
                b["xmin"], b["ymin"], b["xmax"], b["ymax"]]
    return tracks, clip.get("width"), clip.get("height"), len(trajectories)


def relative_id(path):
    """The clip id as the oracle builder wants it: `0021/6833795682`."""
    marker = "annotations/training/"
    if marker in path:
        return path.split(marker, 1)[1][:-len(".json")]
    return os.path.splitext(os.path.basename(path))[0]


def screen(paths, window=16, min_frames=40):
    """One row per clip: (crossover, id, frames, objects). Unwinnable included."""
    from tools.video.screen_vidvrd import window_crossover

    rows = []
    for path in paths:
        try:
            tracks, width, height, frames = read_tracks(path)
        except (ValueError, KeyError, OSError):
            continue
        if frames < min_frames or len(tracks) < 2:
            continue
        crossover = window_crossover(tracks, width, height, window=window)
        if crossover is None:
            continue
        rows.append((crossover, relative_id(path), frames, len(tracks)))
    rows.sort()
    return rows


def summarise(rows, window):
    values = [r[0] for r in rows]
    winnable = [r for r in rows if r[0] < 1.0]
    return {
        "corpus": "vidor", "window": window,
        "clips_screened": len(rows),
        "winnable": len(winnable),
        "winnable_fraction": (float(len(winnable)) / len(rows)) if rows else 0.0,
        "median_crossover": st.median(values) if values else None,
        "median_frames": st.median([r[2] for r in rows]) if rows else None,
        "median_objects": st.median([r[3] for r in rows]) if rows else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--sample", type=int, default=500,
                    help="how many clips to screen; 0 screens all of them")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-frames", type=int, default=40)
    ap.add_argument("--list", default=None,
                    help="write the winnable clip ids here, most winnable first")
    a = ap.parse_args(argv)

    paths = sorted(glob.glob(ANN_GLOB))
    if not paths:
        print("no annotations at %s; run: bash sh/dataset.sh vidor download"
              % ANN_GLOB)
        return 2
    if a.sample and a.sample < len(paths):
        random.seed(a.seed)
        paths = random.sample(paths, a.sample)

    rows = screen(paths, window=a.window, min_frames=a.min_frames)
    summary = summarise(rows, a.window)
    print("screened %d clips at window %d" % (summary["clips_screened"],
                                              a.window))
    print("  winnable            %d (%.0f%%)"
          % (summary["winnable"], 100 * summary["winnable_fraction"]))
    print("  median crossover    %.3f" % summary["median_crossover"])
    print("  median frames       %.0f" % summary["median_frames"])
    print("  median objects      %.0f" % summary["median_objects"])

    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    with open(os.path.join(OUT_DIR, "vidor_screen.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print("wrote %s/vidor_screen.json" % OUT_DIR)

    if a.list:
        directory = os.path.dirname(a.list)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        with open(a.list, "w") as handle:
            handle.write("# VidOR clips winnable at window %d, from a sample "
                         "of %d\n" % (a.window, summary["clips_screened"]))
            handle.write("# crossover = quantisation floor / linear baseline; "
                         "below 1 is winnable\n")
            handle.write("# most winnable first. See WORKPLAN.json Q6 before "
                         "quoting the fraction.\n")
            for crossover, clip_id, _, _ in rows:
                if crossover < 1.0:
                    handle.write("%s\n" % clip_id)
        print("wrote %s" % a.list)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
