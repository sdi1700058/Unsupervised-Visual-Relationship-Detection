#!/usr/bin/env python3
"""HA-ViD into the VidVRD format the FOSAE training builder reads.

WHY. The author locked HA-ViD as medium-2 on 2026-10-03, and it is the first
medium rung to train because its videos are unpacked and its boxes carry a
track id. No training builder reads it: `latplan/puzzles/puzzle_vidvrd.py`
reads VidVRD's format, through `--annotations-dir` and `--frames-dir`. Plan
`notes/plans/2026-10-03-train-the-rungs.md`, steps 1.3a to 1.3c.

THE SOURCE, read on 2026-10-03 from the val COCO file. One COCO file per
split. Images are `frame_NNNNNN.PNG`, numbered across every video of the
split in the order of `ObjectDetection_data_video_info.csv`; every frame of a
boxed video is present (18,405 images for 18,405 frames), so the boxes
between CVAT keyframes are interpolated. 1280 by 720. `bbox` is x, y, width,
height. `attributes.track_id` restarts in each video.

THE RULES THIS FILE ADDS, each a choice of this project:
  view       the M0 view only, one of three cameras, so a session counts once
  windows    whole 150-frame windows, two per video, spread evenly, named
             `<video>_<start:06d>` so the builder's sorted order is time order
  identity   the COCO track, numbered per clip in track order
  size       frames decoded at 640 by 360 and boxes halved, to save space;
             the builder crops 8-pixel patches, far below either size

    python3 tools/video/convert_havid.py --ext data/video/ha_vid/ext \\
        --frame-rates data/video/ha_vid/frame_rates.tsv \\
        --out data/video/ha_vid/vidvrd_format --ids-out IDS
"""

import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys


def spans(rows):
    """`[(video, first global frame, frame count)]` in CSV order."""
    out, first = [], 0
    for name, count in rows:
        out.append((name.rsplit(".", 1)[0], first, int(count)))
        first += int(count)
    return out


def boxes_by_video(coco, video_spans, wanted):
    """`{video: {local frame: [(track, category, (x1, y1, x2, y2))]}}`."""
    names = dict((c["id"], c["name"]) for c in coco["categories"])
    where = {}
    for image in coco["images"]:
        index = int(image["file_name"].split("_")[-1].split(".")[0])
        for video, first, count in video_spans:
            if first <= index < first + count:
                if video in wanted:
                    where[image["id"]] = (video, index - first)
                break
    out = {}
    for ann in coco["annotations"]:
        hit = where.get(ann["image_id"])
        if hit is None:
            continue
        x, y, w, h = ann["bbox"]
        track = (ann.get("attributes") or {}).get("track_id")
        if track is None:
            continue
        out.setdefault(hit[0], {}).setdefault(hit[1], []).append(
            (track, names.get(ann["category_id"]), (x, y, x + w, y + h)))
    for frames in out.values():
        for boxes in frames.values():
            boxes.sort(key=lambda b: (b[0], b[1]))
    return out


def window_starts(n_frames, size=150, per_video=2):
    """Starts of `per_video` whole windows, spread evenly over the video."""
    whole = list(range(0, n_frames - size + 1, size))
    if not whole:
        return []
    picks = sorted(set(int(math.floor(k * len(whole) / float(per_video)))
                       for k in range(per_video)))
    return [whole[i] for i in picks if i < len(whole)]


def clip_id(video, start):
    return "%s_%06d" % (video, start)


def _px(value, limit):
    return int(math.floor(min(max(value, 0.0), limit) + 0.5))


def clip_json(clip, boxes, start, size, width, height, scale):
    """The VidVRD-format document for frames start .. start+size-1."""
    category = {}
    for f in range(start, start + size):
        for track, cat, _box in boxes.get(f, []):
            category.setdefault(track, cat)
    tid = dict((t, k) for k, t in enumerate(sorted(category)))
    trajectories = []
    for f in range(start, start + size):
        row = []
        for track, _cat, (x1, y1, x2, y2) in boxes.get(f, []):
            row.append({"tid": tid[track], "generated": 0,
                        "bbox": {"xmin": _px(x1 * scale, width),
                                 "ymin": _px(y1 * scale, height),
                                 "xmax": _px(x2 * scale, width),
                                 "ymax": _px(y2 * scale, height)}})
        trajectories.append(row)
    return {"video_id": clip, "width": int(width), "height": int(height),
            "subject/objects": [{"tid": tid[t], "category": category[t]}
                                for t in sorted(category)],
            "trajectories": trajectories, "relation_instances": []}


def _decode(mp4, out_dir, width):
    os.makedirs(out_dir, exist_ok=True)
    subprocess.check_call(["ffmpeg", "-loglevel", "error", "-y", "-i", mp4,
                           "-vf", "scale=%d:-2" % width, "-vsync", "0",
                           "-start_number", "0", "-q:v", "2",
                           os.path.join(out_dir, "%06d.jpg")])
    return len(glob.glob(os.path.join(out_dir, "*.jpg")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ext", required=True, help="data/video/ha_vid/ext")
    ap.add_argument("--frame-rates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids-out", required=True)
    ap.add_argument("--view", default="M0")
    ap.add_argument("--size", type=int, default=150)
    ap.add_argument("--per-video", type=int, default=2)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args(argv)

    info = glob.glob(os.path.join(args.ext, "ObjectDetection", "**",
                                  "ObjectDetection_data_video_info.csv"),
                     recursive=True)[0]
    rows_by_split = {}
    with open(info) as f:
        for row in csv.DictReader(f):
            rows_by_split.setdefault(row["split"], []).append(
                (row["vid_name"], int(row["num_frames"])))
    decoded = {}
    with open(args.frame_rates) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            decoded[row["video"]] = int(row["decoded"])

    boxes, frames_of = {}, {}
    for split, rows in sorted(rows_by_split.items()):
        video_spans = spans(rows)
        wanted = set(v for v, _first, _n in video_spans
                     if v.endswith(args.view))
        for v, _first, n in video_spans:
            if v in wanted:
                frames_of[v] = n
        path = glob.glob(os.path.join(args.ext, "ObjectDetection", split,
                                      "**", "annotation.json"),
                         recursive=True)[0]
        with open(path) as f:
            coco = json.load(f)
        boxes.update(boxes_by_video(coco, video_spans, wanted))
        del coco
        print("read %s: %d %s videos" % (split, len(wanted), args.view))

    ann_dir = os.path.join(args.out, "annotations")
    frames_root = os.path.join(args.out, "frames_15fps")
    os.makedirs(ann_dir, exist_ok=True)
    scale = args.width / 1280.0
    height = int(round(720 * scale))
    kept, dropped = [], {}
    for video in sorted(boxes):
        n = frames_of[video]
        if decoded.get(video) != n:
            dropped[video] = "CSV says %d frames, frame_rates.tsv %s" % (
                n, decoded.get(video))
            continue
        mp4 = glob.glob(os.path.join(args.ext, "rgb", "**", video + ".mp4"),
                        recursive=True)
        if not mp4:
            dropped[video] = "no mp4"
            continue
        whole = os.path.join(frames_root, "_video", video)
        got = _decode(mp4[0], whole, args.width)
        if got != n:
            dropped[video] = "decoded %d frames, expected %d" % (got, n)
            continue
        for start in window_starts(n, args.size, args.per_video):
            clip = clip_id(video, start)
            doc = clip_json(clip, boxes[video], start, args.size, args.width,
                            height, scale)
            link_dir = os.path.join(frames_root, clip)
            os.makedirs(link_dir, exist_ok=True)
            for k in range(args.size):
                dst = os.path.join(link_dir, "%06d.jpg" % k)
                if not os.path.lexists(dst):
                    os.symlink(os.path.join("..", "_video", video,
                                            "%06d.jpg" % (start + k)), dst)
            with open(os.path.join(ann_dir, clip + ".json"), "w") as f:
                json.dump(doc, f)
            kept.append(clip)
    with open(args.ids_out, "w") as f:
        f.write("\n".join(sorted(kept)) + "\n")
    print("wrote %d clips from %d videos, dropped %d videos: %s"
          % (len(kept), len(boxes) - len(dropped), len(dropped),
             json.dumps(dropped)))
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
