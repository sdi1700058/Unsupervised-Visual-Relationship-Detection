#!/usr/bin/env python3
"""KIT Bimanual into the VidVRD format the FOSAE training builder reads.

WHY. The author locked KIT Bimanual as medium-1 on 2026-10-03. No training
builder reads it: `latplan/puzzles/puzzle_vidvrd.py` reads VidVRD's format,
through `--annotations-dir` and `--frames-dir`. Plan
`notes/plans/2026-10-03-train-the-rungs.md`, steps 2.1 to 2.3.

THE SOURCE, read on 2026-10-03. Per take, `2d_objects/frame_N.json` holds a
list of detections, each `bounding_box {x, y, w, h}` normalised to the image
and `candidates`, the first with `class_name` and `certainty`. The RGB frame
is `rgb/chunk_K/frame_N.png`, 640 by 480, one per JSON: 11,267 of each for
take 0 of the nine tasks of subjects 1, 2 and 6. There is no track id.
`x, y` is the box CENTRE: on 6,252 boxes of three takes, a corner reading puts
959 boxes past the frame edge and a centre reading puts none there.

THE RULES THIS FILE ADDS, each a choice of this project:
  identity   the class name is the track, numbered per clip in name order.
             A class seen twice in one frame keeps its most certain box, and
             the others are counted, because the detector's duplicates
             cannot be told from two real objects of one class
  windows    every whole 150-frame window of each take, named
             `s<subject>_<task>_take<k>_<start:06d>` so subject 6, the last
             subject, is the clip split's val and test

    python3 tools/video/convert_kit.py --derived DERIVED --rgb RGB \\
        --takes s1:task_5_k_cereals:0 ... --out OUT --ids-out IDS
"""

import argparse
import glob
import json
import math
import os
import sys


def _px(value, limit):
    return int(math.floor(min(max(value, 0.0), limit) + 0.5))


def frame_boxes(detections, width, height, counts=None):
    """`{class: (certainty, (x1, y1, x2, y2))}` in pixels, one box a class."""
    out = {}
    for det in detections:
        cand = (det.get("candidates") or [{}])[0]
        name = cand.get("class_name")
        if not name:
            continue
        b = det["bounding_box"]
        box = (_px((b["x"] - b["w"] / 2.0) * width, width),
               _px((b["y"] - b["h"] / 2.0) * height, height),
               _px((b["x"] + b["w"] / 2.0) * width, width),
               _px((b["y"] + b["h"] / 2.0) * height, height))
        certainty = float(cand.get("certainty", 0.0))
        if name in out:
            if counts is not None:
                counts["duplicates"] = counts.get("duplicates", 0) + 1
            if certainty <= out[name][0]:
                continue
        out[name] = (certainty, box)
    return out


def window_starts(n_frames, size=150):
    return list(range(0, n_frames - size + 1, size))


def clip_id(subject, task, take, start):
    return "s%d_%s_take%d_%06d" % (subject, task, take, start)


def clip_json(clip, frames, start, size, width, height):
    """The VidVRD-format document for frames start .. start+size-1."""
    names = sorted(set(n for f in range(start, start + size)
                       for n in frames.get(f, {})))
    tid = dict((n, k) for k, n in enumerate(names))
    trajectories = []
    for f in range(start, start + size):
        row = []
        for name, (_c, (x1, y1, x2, y2)) in sorted(frames.get(f, {}).items()):
            row.append({"tid": tid[name], "generated": 0,
                        "bbox": {"xmin": x1, "ymin": y1, "xmax": x2,
                                 "ymax": y2}})
        trajectories.append(row)
    return {"video_id": clip, "width": int(width), "height": int(height),
            "subject/objects": [{"tid": tid[n], "category": n} for n in names],
            "trajectories": trajectories, "relation_instances": []}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--derived", required=True,
                    help="data/video/kit_bimanual/ext/bimacs_derived_data")
    ap.add_argument("--rgb", required=True,
                    help="data/video/kit_bimanual/rgb_ext/bimacs_rgbd_data")
    ap.add_argument("--takes", nargs="+", required=True,
                    help="subject:task:take, for example 1:task_5_k_cereals:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids-out", required=True)
    ap.add_argument("--size", type=int, default=150)
    args = ap.parse_args(argv)

    from PIL import Image
    ann_dir = os.path.join(args.out, "annotations")
    frames_root = os.path.join(args.out, "frames_30fps")
    os.makedirs(ann_dir, exist_ok=True)
    kept, counts, dropped = [], {"duplicates": 0}, {}
    for spec in args.takes:
        subject, task, take = spec.split(":")
        subject, take = int(subject), int(take)
        base = "subject_%d/%s/take_%d" % (subject, task, take)
        pngs = dict((int(os.path.basename(p)[len("frame_"):-len(".png")]), p)
                    for p in glob.glob(os.path.join(args.rgb, base, "rgb",
                                                    "chunk_*", "frame_*.png")))
        jsons = glob.glob(os.path.join(args.derived, base, "2d_objects",
                                       "frame_*.json"))
        n = len(jsons)
        if n == 0 or sorted(pngs) != list(range(n)):
            dropped[spec] = "%d box files, %d frames" % (n, len(pngs))
            continue
        with Image.open(pngs[0]) as im:
            width, height = im.size
        frames = {}
        for f in range(n):
            with open(os.path.join(args.derived, base, "2d_objects",
                                   "frame_%d.json" % f)) as handle:
                frames[f] = frame_boxes(json.load(handle), width, height,
                                        counts)
        whole = os.path.join(frames_root, "_video", "s%d_%s_take%d"
                             % (subject, task, take))
        os.makedirs(whole, exist_ok=True)
        for f in range(n):
            dst = os.path.join(whole, "%06d.jpg" % f)
            if not os.path.exists(dst):
                with Image.open(pngs[f]) as im:
                    im.convert("RGB").save(dst, quality=92)
        for start in window_starts(n, args.size):
            clip = clip_id(subject, task, take, start)
            link_dir = os.path.join(frames_root, clip)
            os.makedirs(link_dir, exist_ok=True)
            for k in range(args.size):
                dst = os.path.join(link_dir, "%06d.jpg" % k)
                if not os.path.lexists(dst):
                    os.symlink(os.path.join("..", "_video", os.path.basename(
                        whole), "%06d.jpg" % (start + k)), dst)
            with open(os.path.join(ann_dir, clip + ".json"), "w") as handle:
                json.dump(clip_json(clip, frames, start, args.size, width,
                                    height), handle)
            kept.append(clip)
    with open(args.ids_out, "w") as handle:
        handle.write("\n".join(sorted(kept)) + "\n")
    print("wrote %d clips, %d duplicate boxes refused, dropped takes: %s"
          % (len(kept), counts["duplicates"], json.dumps(dropped)))
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
