#!/usr/bin/env python3
"""Something-Else into the VidVRD format the FOSAE training builder reads.

WHY. The author locked Something-Else as the easy rung on 2026-10-03, and no
training builder reads it: `latplan/puzzles/puzzle_vidvrd.py` reads VidVRD's
format only, and its `--annotations-dir` and `--frames-dir` flags exist for a
dataset written into that format. This writes one VidVRD-format JSON per clip
and decodes each clip's SSv2 video into the frames the builder expects. Plan
`notes/plans/2026-10-03-train-the-rungs.md`, steps 1.2a to 1.2c.

THE SOURCE, read on 2026-10-03 from the head of
`bounding_box_smthsmth_part2.json`: per video, a list of annotated frames
named `<video>/<NNNN>.jpg`, counted from 1, each with labels carrying `box2d`
(`x1 y1 x2 y2`, pixels), a free-text `category` and a `standard_category`,
which is a per-video track id: `0000`, `0001`, `hand`. An annotation can
cover part of a video only: that file's first video starts at `0009.jpg`.

THE RULES THIS FILE ADDS, each a choice of this project:
  identity   `standard_category` is the track. Objects take tids in id order
             and the hand comes last, as `tools/planner/box_geometry.py`
             orders them.
  two hands  a clip where one track id repeats inside a frame is skipped:
             two boxes under one id cannot be told apart.
  repeats    a clip that lists one frame twice is skipped: three clips do,
             with different boxes, found 2026-10-03.
  tracks     at most 3, counting every hand, because the builder drops a
             track past slot 3 for the whole clip.

    python3 tools/video/convert_something_else.py \\
        --boxes RAW/bounding_box_smthsmth_part{2,3,4}.json \\
        --videos data/video/ssv2/ext/20bn-something-something-v2 \\
        --labels data/video/ssv2/raw/labels.json \\
        --splits data/video/ssv2/raw/train.json data/video/ssv2/raw/validation.json \\
        --templates TEMPLATE_IDS --out data/video/something_else/vidvrd_format \\
        --count 250 --ids-out IDS
"""

import argparse
import glob
import json
import os
import random
import subprocess
import sys


def frame_index(name):
    """`<video>/<NNNN>.jpg` counted from 1, as a fid counted from 0."""
    return int(os.path.splitext(os.path.basename(name))[0]) - 1


def _order(std):
    return (std == "hand", std)


def clip_json(video_id, frames, width, height):
    """The VidVRD-format document for one Something-Else clip."""
    cats = {}
    for frame in frames:
        for label in frame.get("labels") or []:
            cats.setdefault(label["standard_category"], label.get("category"))
    stds = sorted(cats, key=_order)
    tid = dict((s, k) for k, s in enumerate(stds))
    n = max(frame_index(f["name"]) for f in frames) + 1 if frames else 0
    trajectories = [[] for _ in range(n)]
    for frame in frames:
        row = trajectories[frame_index(frame["name"])]
        for label in frame.get("labels") or []:
            b = label["box2d"]
            row.append({"tid": tid[label["standard_category"]],
                        "generated": 0,
                        "bbox": {"xmin": int(round(min(max(b["x1"], 0), width))),
                                 "ymin": int(round(min(max(b["y1"], 0), height))),
                                 "xmax": int(round(min(max(b["x2"], 0), width))),
                                 "ymax": int(round(min(max(b["y2"], 0), height)))}})
    return {"video_id": str(video_id), "width": int(width),
            "height": int(height),
            "subject/objects": [{"tid": tid[s], "category": cats[s]}
                                for s in stds],
            "trajectories": trajectories, "relation_instances": []}


def why_not(frames, max_tracks=3, min_frames=30):
    """Why a clip is not usable, or None when it is."""
    if len(frames) < min_frames:
        return "fewer than %d annotated frames" % min_frames
    # Found 2026-10-03: three clips list 114 frames twice with different
    # boxes, so one track holds two boxes in one frame.
    names = [f["name"] for f in frames]
    if len(names) != len(set(names)):
        return "a frame is listed twice"
    tracks = set()
    for frame in frames:
        stds = [l["standard_category"] for l in frame.get("labels") or []]
        if len(stds) != len(set(stds)):
            return "a track id repeats inside one frame"
        tracks.update(stds)
    if len(tracks) > max_tracks:
        return "more than %d tracks" % max_tracks
    if len(tracks) < 2:
        return "fewer than 2 tracks"
    return None


def choose(records, template_of, allowed, count, seed=0, max_tracks=3,
           min_frames=30):
    """`count` usable clips whose template is allowed, drawn by `seed`."""
    pool = sorted(v for v, frames in records.items()
                  if template_of.get(v) in allowed
                  and why_not(frames, max_tracks, min_frames) is None)
    return sorted(random.Random(seed).sample(pool, min(count, len(pool))))


def _template_ids(labels_path, split_paths):
    """`{video id: template id}` from SSv2's labels and split files."""
    with open(labels_path) as f:
        label_id = json.load(f)
    out = {}
    for path in split_paths:
        with open(path) as f:
            for row in json.load(f):
                text = row["template"].replace("[", "").replace("]", "")
                if text in label_id:
                    out[str(row["id"])] = label_id[text]
    return out


def _decode(webm, frames_dir):
    """Decode every frame to `%06d.jpg` from 0, return (count, width, height)."""
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.check_call(["ffmpeg", "-loglevel", "error", "-y", "-i", webm,
                           "-vsync", "0", "-start_number", "0", "-q:v", "2",
                           os.path.join(frames_dir, "%06d.jpg")])
    names = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    from PIL import Image
    with Image.open(names[0]) as im:
        width, height = im.size
    return len(names), width, height


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--boxes", nargs="+", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--templates", required=True,
                    help="file of allowed template ids, one per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--count", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tracks", type=int, default=3)
    ap.add_argument("--min-frames", type=int, default=30)
    ap.add_argument("--ids-out", required=True)
    args = ap.parse_args(argv)

    with open(args.templates) as f:
        allowed = set(line.strip() for line in f if line.strip())
    template_of = _template_ids(args.labels, args.splits)
    records, reasons = {}, {}
    for path in args.boxes:
        with open(path) as f:
            part = json.load(f)
        for video, frames in part.items():
            if template_of.get(video) not in allowed:
                reasons["template does not change a two-object predicate"] = \
                    reasons.get("template does not change a two-object "
                                "predicate", 0) + 1
                continue
            why = why_not(frames, args.max_tracks, args.min_frames)
            if why:
                reasons[why] = reasons.get(why, 0) + 1
            else:
                records[video] = frames
        del part
        print("read %s: %d usable clips so far" % (path, len(records)))
    print("refused, by reason: %s" % json.dumps(reasons, sort_keys=True))
    chosen = choose(records, template_of, allowed, args.count, args.seed,
                    args.max_tracks, args.min_frames)
    print("chose %d of %d usable clips, seed %d"
          % (len(chosen), len(records), args.seed))

    ann_dir = os.path.join(args.out, "annotations")
    frames_root = os.path.join(args.out, "frames_%dfps" % args.fps)
    os.makedirs(ann_dir, exist_ok=True)
    kept, dropped = [], {}
    for video in chosen:
        webm = os.path.join(args.videos, video + ".webm")
        if not os.path.isfile(webm):
            dropped[video] = "no video file"
            continue
        count, width, height = _decode(webm, os.path.join(frames_root, video))
        last = max(frame_index(f["name"]) for f in records[video])
        if last >= count:
            dropped[video] = ("annotated frame %d but the video decodes to %d"
                              % (last, count))
            continue
        with open(os.path.join(ann_dir, video + ".json"), "w") as f:
            json.dump(clip_json(video, records[video], width, height), f)
        kept.append(video)
    with open(args.ids_out, "w") as f:
        f.write("\n".join(kept) + "\n")
    print("wrote %d clips to %s, dropped %d: %s"
          % (len(kept), ann_dir, len(dropped), json.dumps(dropped)))
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
