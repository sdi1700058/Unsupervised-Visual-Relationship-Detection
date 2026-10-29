#!/usr/bin/env python3
"""HA-ViD into the VidVRD format the FOSAE training builder reads.

WHY. The author locked HA-ViD as medium-2 on 2026-10-03, and it is the first
medium rung to train because its videos are unpacked and its boxes carry a
track id. No training builder reads it: `model/loaders/vidvrd.py`
reads VidVRD's format, through `--annotations-dir` and `--frames-dir`. Plan
`notes/plans/2026-10-03-train-the-rungs.md`, steps 1.3a to 1.3c
(archived, see workbench/ARCHIVE.md).

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

    python3 tools/datasets/convert_havid.py --ext data/video/ha_vid/ext \\
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
from tools.datasets import ids_out


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



# Role-based slot selection, decided by the author on 2026-10-05. Plan
# notes/plans/2026-10-05-dataset-fixes.md, F4 (archived, see workbench/ARCHIVE.md).
# An action code is a verb
# letter, then two-letter codes for the moved object, the target object and
# the tool, in that order: `s sh c1 dh` is screw, hex screw, cylinder plate
# hole 1, hex screwdriver. The list of codes is Figure 9 of `zheng2023`,
# p.17. Each object code maps to the COCO category of its boxes.
CODE_CATEGORY = {
    "ba": "Ball", "bs": "Ball Seat", "bx": "Box",
    "c1": "Hole C1", "c2": "Hole C2", "c3": "Hole C3", "c4": "Hole C4",
    "cb": "Cylinder Base", "cc": "Cylinder Cap", "ck": "Cylinder Bracket",
    "cs": "Cylinder Subassembly", "dh": "Screwdriver (hex)",
    "dp": "Screwdriver (philips)", "ft": "Shaft",
    "g1": "Hole G1 (GL)", "g2": "Hole G2 (GS)", "g3": "Hole G3 (GW)",
    "gl": "Gear Large", "gs": "Gear Small", "gw": "Worm Gear",
    "hd": "Hand-dial", "hq": "Quarter-turn Handle", "hw": "Hand-wheel",
    "ib": "Bar", "ir": "Rod", "lb": "Linear Bearing",
    "n1": "Hole N1 (NR)", "n2": "Hole N2 (NB)", "n3": "Hole N3 (NL)",
    "n4": "Hole N4 (NS)", "n5": "Hole/Stud N5 (NN)", "n6": "Hole N6 (NU)",
    "nt": "Nut", "pl": "Spacer Large", "ps": "Spacer Small", "sb": "Bolt",
    "sh": "Screw (hex)", "sp": "Screw (philips)", "us": "USB Male",
    "wn": "Wrench (nut)", "ws": "Wrench (shaft)",
}
TOOLS = ("dh", "dp", "wn", "ws")


def parse_code(code):
    """`(verb, moved, target, tool)`, each object as its box category."""
    chunks = [code[k:k + 2] for k in range(1, len(code), 2)]
    moved = chunks[0] if chunks else None
    target = chunks[1] if len(chunks) > 1 else None
    tool = chunks[2] if len(chunks) > 2 else None
    if len(chunks) == 2 and target in TOOLS:
        target, tool = None, target
    return (code[:1], CODE_CATEGORY.get(moved), CODE_CATEGORY.get(target),
            CODE_CATEGORY.get(tool))


def read_segments(text):
    """`[(start, end, code)]` of one `temporal_timestamps` file.

    Its frame numbers are those of the boxes. The per-frame `groundTruth`
    files of the action segmentation release run 10 frames shorter and
    start segments 4 to 5 frames early, so they are not read here.
    """
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] not in ("null", "w"):
            out.append((int(parts[0]), int(parts[1]), parts[2]))
    return out


def _centres(boxes, start, end, track):
    out = {}
    for f in range(start, end + 1):
        for t, _cat, (x1, y1, x2, y2) in boxes.get(f, []):
            if t == track:
                out[f] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return out


def _distance(boxes, start, end, a, b):
    ca, cb = _centres(boxes, start, end, a), _centres(boxes, start, end, b)
    shared = sorted(set(ca) & set(cb))
    if not shared:
        return float("inf")
    return sum(math.hypot(ca[f][0] - cb[f][0], ca[f][1] - cb[f][1])
               for f in shared) / len(shared)


def _tracks_of(boxes, start, end, category):
    return sorted(set(t for f in range(start, end + 1)
                      for t, cat, _box in boxes.get(f, []) if cat == category))


def role_tracks(boxes, start, end, moved, target):
    """`(hand, moved, target)` tracks of one action, or None.

    The boxes give no left or right hand, so the acting hand is the hand
    track nearest the moved object. When a category has several tracks,
    the moved object is the one nearest a hand and the target the one
    nearest the moved object.
    """
    hands = _tracks_of(boxes, start, end, "Hand")
    movers = _tracks_of(boxes, start, end, moved)
    targets = _tracks_of(boxes, start, end, target)
    if not hands or not movers or not targets:
        return None
    m = min(movers, key=lambda t: (min(_distance(boxes, start, end, t, h)
                                       for h in hands), t))
    h = min(hands, key=lambda t: (_distance(boxes, start, end, t, m), t))
    others = [t for t in targets if t != m]
    if not others:
        return None
    g = min(others, key=lambda t: (_distance(boxes, start, end, t, m), t))
    return (h, m, g)


def coverage(boxes, start, end, track):
    """The share of frames start .. end where `track` has a box."""
    return len(_centres(boxes, start, end, track)) / float(end - start + 1)


def role_clip_id(video, start, hand):
    return "%s_pt%06d_%s" % (video, start, hand)


# A key action of Figure 9 of arXiv 2307.05721v1, as the state it leaves
# behind. `place`, `insert`, `screw`, `rotate` and `slide` are the five the
# paper's ontology calls the ones that change the assembly.
KEY_VERB_RELATION = {"p": "placed", "i": "inserted", "s": "screwed",
                     "r": "rotated", "l": "slid"}


def relation_instances(verb, n_tracks, size):
    """This project's rule over one HA-ViD clip, in the VidVRD field.

    A key action with a target object reads as a relation between the moved
    object and the target, TRUE WHEN THE ACTION ENDS.
    `workbench/scripts/analysis/havid_relations.py` states the rule and marks it T3,
    this project's own, because HA-ViD ships no relation of its own.

    A clip is exactly one primitive task: `end = min(end, start + size - 1,
    n - 1)` below clamps the clip to the segment, so the action ends on the
    clip's LAST frame. The relation therefore holds over that one frame and
    the clip offers one transition where it changes. That is thin, and it is
    what the rule gives. `role_clip_json` fixes tid 0 as the hand, 1 as the
    moved object and 2 as the target, so the arguments need no lookup.

    Plan notes/plans/2026-10-10-relations-and-boxes.md, A3
    (archived, see workbench/ARCHIVE.md).
    """
    name = KEY_VERB_RELATION.get(verb)
    if name is None or n_tracks < 3 or size < 1:
        return []
    return [{"predicate": name, "subject_tid": 1, "object_tid": 2,
             "begin_fid": size - 1, "end_fid": size}]


def role_clip_json(clip, boxes, start, size, width, height, scale, tracks):
    """The VidVRD-format document of the three role tracks: tid 0 is the
    hand, 1 the moved object and 2 the target object."""
    tid = dict((t, k) for k, t in enumerate(tracks))
    category = {}
    trajectories = []
    for f in range(start, start + size):
        row = []
        for track, cat, (x1, y1, x2, y2) in boxes.get(f, []):
            if track not in tid:
                continue
            category.setdefault(track, cat)
            row.append({"tid": tid[track], "generated": 0,
                        "bbox": {"xmin": _px(x1 * scale, width),
                                 "ymin": _px(y1 * scale, height),
                                 "xmax": _px(x2 * scale, width),
                                 "ymax": _px(y2 * scale, height)}})
        trajectories.append(row)
    return {"video_id": clip, "width": int(width), "height": int(height),
            "subject/objects": [{"tid": tid[t], "category": category[t]}
                                for t in tracks if t in category],
            "trajectories": trajectories, "relation_instances": []}

def _decode(mp4, out_dir, width, expected=None):
    if expected and len(glob.glob(os.path.join(out_dir, "*.jpg"))) == expected:
        return expected
    os.makedirs(out_dir, exist_ok=True)
    subprocess.check_call(["ffmpeg", "-loglevel", "error", "-y", "-i", mp4,
                           "-vf", "scale=%d:-2" % width, "-vsync", "0",
                           "-start_number", "0", "-q:v", "2",
                           os.path.join(out_dir, "%06d.jpg")])
    return len(glob.glob(os.path.join(out_dir, "*.jpg")))


def _role_clips(args, video, boxes, n, frames_root, ann_dir, height, scale,
                refused):
    """Write one clip for each primitive task of `video` that has all three
    roles boxed. `refused` counts the others by reason."""
    def refuse(reason):
        refused[reason] = refused.get(reason, 0) + 1

    view = "view%d" % {"M0": 0, "S1": 1, "S2": 2}[args.view]
    stem = video[:-len(args.view)]
    kept = []
    for hand in ("lh", "rh"):
        path = os.path.join(args.ext, "HAViD_temporalAnnotation",
                            "%s_%s_pt" % (view, hand), "temporal_timestamps",
                            "%s_%s_%s_pt_temporal.txt" % (stem, view, hand))
        if not os.path.exists(path):
            refuse("no primitive task file")
            continue
        with open(path) as f:
            segments = read_segments(f.read())
        for start, end, code in segments:
            if end - start + 1 < args.min_length:
                refuse("shorter than %d frames" % args.min_length)
                continue
            end = min(end, start + args.size - 1, n - 1)
            _verb, moved, target, _tool = parse_code(code)
            if moved is None or target is None:
                refuse("no moved object or no target in the code")
                continue
            tracks = role_tracks(boxes, start, end, moved, target)
            if tracks is None:
                refuse("a role has no box")
                continue
            if min(coverage(boxes, start, end, t) for t in tracks) \
                    < args.min_coverage:
                refuse("a role is boxed in under %d%% of the frames"
                       % int(100 * args.min_coverage))
                continue
            clip = role_clip_id(video, start, hand)
            size = end - start + 1
            link_dir = os.path.join(frames_root, clip)
            os.makedirs(link_dir, exist_ok=True)
            for k in range(size):
                dst = os.path.join(link_dir, "%06d.jpg" % k)
                if not os.path.lexists(dst):
                    os.symlink(os.path.join("..", "_video", video,
                                            "%06d.jpg" % (start + k)), dst)
            doc = role_clip_json(clip, boxes, start, size,
                                 args.width, height, scale, tracks)
            doc["relation_instances"] = relation_instances(
                _verb, len(doc["subject/objects"]), size)
            with open(os.path.join(ann_dir, clip + ".json"), "w") as f:
                json.dump(doc, f)
            kept.append(clip)
    return kept


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ext", required=True, help="data/video/ha_vid/ext")
    ap.add_argument("--frame-rates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids-out", required=True)
    ap.add_argument("--force-ids-out", action="store_true",
                    help="overwrite an existing ids list. Two tracked\n"
                         "lists were lost without this, OPEN-ISSUES I28")
    ap.add_argument("--view", default="M0")
    ap.add_argument("--size", type=int, default=150)
    ap.add_argument("--per-video", type=int, default=2)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--roles", action="store_true",
                    help="one clip for each primitive task, with only its "
                         "hand, moved object and target object")
    ap.add_argument("--min-length", type=int, default=30)
    ap.add_argument("--min-coverage", type=float, default=0.8)
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

    ann_dir = os.path.join(args.out,
                           "annotations_roles" if args.roles else "annotations")
    frames_root = os.path.join(args.out, "frames_15fps")
    os.makedirs(ann_dir, exist_ok=True)
    scale = args.width / 1280.0
    height = int(round(720 * scale))
    kept, dropped, refused = [], {}, {}
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
        got = _decode(mp4[0], whole, args.width, n)
        if got != n:
            dropped[video] = "decoded %d frames, expected %d" % (got, n)
            continue
        if args.roles:
            kept += _role_clips(args, video, boxes[video], n, frames_root,
                                ann_dir, height, scale, refused)
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
    with ids_out.open_ids_out(args.ids_out,
                              args.force_ids_out) as f:
        f.write("\n".join(sorted(kept)) + "\n")
    print("wrote %d clips from %d videos, dropped %d videos: %s"
          % (len(kept), len(boxes) - len(dropped), len(dropped),
             json.dumps(dropped)))
    if args.roles:
        print("refused primitive tasks, by reason: %s"
              % json.dumps(refused, sort_keys=True))
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
