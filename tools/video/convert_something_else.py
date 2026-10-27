#!/usr/bin/env python3
"""Something-Else into the VidVRD format the FOSAE training builder reads.

WHY. The author locked Something-Else as the easy rung on 2026-10-03, and no
training builder reads it: `model/puzzles/puzzle_vidvrd.py` reads VidVRD's
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
             and the hand comes last, as `planner/box_geometry.py`
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
from tools.video import ids_out


def frame_index(name):
    """`<video>/<NNNN>.jpg` counted from 1, as a fid counted from 0."""
    return int(os.path.splitext(os.path.basename(name))[0]) - 1


HAND = "hand"


def asserted_pairs(text, name):
    """`{(a, b)}`: the argument pairs `text` states `name` TRUE for.

    Judged for each pair, not for the name. `bury` asserts `(in ?a hand)`
    and negates `(in ?b hand)` and `(in ?a ?b)` in one precondition, and a
    reader that judged the name alone called the whole predicate negated
    and threw the true pair away. Found by an adversary on 2026-10-11.
    """
    import re
    negated = set()
    for m in re.finditer(r"\(\s*not\s*\(\s*%s\s+([^()]*)\)"
                         % re.escape(name), text):
        parts = m.group(1).split()
        if len(parts) == 2:
            negated.add((parts[0], parts[1]))
    out = set()
    for m in re.finditer(r"\(\s*%s\s+([^()]*)\)" % re.escape(name), text):
        parts = m.group(1).split()
        if len(parts) == 2 and (parts[0], parts[1]) not in negated:
            out.add((parts[0], parts[1]))
    return out


def _order(std):
    return (std == "hand", std)


def _anchors(doc):
    """The first and last frame where every track of the clip has a box.

    The relations of this dataset hold at the ends of the clip and nowhere
    between, so they need a frame where the whole scene is visible. Frame 0
    does not serve: measured 2026-10-11 over the 237 converted clips, every
    track is boxed in frame 0 in 99 of them and in the last frame in 64, and
    `planner/m7_map.py:284` drops an instance whose range holds a frame
    with a box missing.
    """
    want = set(o["tid"] for o in doc.get("subject/objects") or [])
    if not want:
        return None, None
    full = [i for i, frame in enumerate(doc.get("trajectories") or [])
            if want <= set(b["tid"] for b in frame or [])]
    return (full[0], full[-1]) if full else (None, None)


def relation_instances(doc, placeholders, action, actions, predicates):
    """The domain's statement about this clip, in the VidVRD field.

    `action` is the PDDL action of the clip's SSv2 template, `actions` maps
    an action to its (precondition, effect) text, and `predicates` gives each
    name its arity. EVERY binary predicate the precondition asserts holds at
    the first anchor frame, and every one the effect asserts holds at the
    last. A negated one is not written, because the field lists what is
    true, and a predicate of one argument is a property and not a relation.

    **Not only the predicates the action CHANGES.** The first version of this
    wrote the flips alone and left 98 of the 234 training clips with no
    relation at all, measured 2026-10-11. `lift-end-drop` is the case:
    its precondition holds `(in ?a hand)`, which is true and is a relation,
    while the only predicate it flips is `touching`, which its effect
    NEGATES. Writing flips alone threw the true relation away and wrote
    nothing.

    A placeholder binds to an object track by position, `?a` to the first and
    `?b` to the second, and the constant `hand` to the hand track. Positional
    and not by name, because SSv2 truncates: the placeholder `paper plate`
    meets the track `paper`, and `can opener` meets `copener`. An earlier
    draft of this line claimed the rule holds on 227 of 234 clips. No script
    produced that number and an adversary could not reproduce it on
    2026-10-11, so it is withdrawn. What the order rests on is the converter
    itself: `clip_json` sorts the tracks by `standard_category` with the hand
    last, and SSv2 lists the placeholders in the order of the template's
    slots.

    Plan notes/plans/2026-10-10-relations-and-boxes.md, A2.
    """
    first, last = _anchors(doc)
    if first is None:
        return []
    objs = [o for o in sorted(doc["subject/objects"], key=lambda o: o["tid"])
            if o["category"] != HAND]
    hands = [o["tid"] for o in doc["subject/objects"]
             if o["category"] == HAND]
    if len(objs) != len(placeholders):
        return []

    def tid_of(arg):
        if arg == HAND:
            return hands[0] if hands else None
        if not arg.startswith("?"):
            return None
        k = ord(arg[1]) - ord("a")
        return objs[k]["tid"] if 0 <= k < len(objs) else None

    pre, eff = actions.get(action, ("", ""))
    out = []
    binary = sorted(n for n, arity in predicates.items() if arity == 2)
    for name in binary:
        for text, begin in ((pre, first), (eff, last)):
            for a, b in sorted(asserted_pairs(text, name)):
                s_tid, o_tid = tid_of(a), tid_of(b)
                if s_tid is None or o_tid is None or s_tid == o_tid:
                    continue
                out.append({"predicate": name, "subject_tid": s_tid,
                            "object_tid": o_tid, "begin_fid": begin,
                            "end_fid": begin + 1})
    out.sort(key=lambda r: (r["begin_fid"], r["subject_tid"],
                            r["object_tid"], r["predicate"]))
    return out


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
           min_frames=30, only=None):
    """`count` usable clips whose template is allowed, drawn by `seed`.

    `only` names the clips instead, and then the seed, the count and the
    filters are not consulted. **A seed does not make this repeatable.** The
    clips of `workbench/sh/rung_ids/se-f6.txt` were drawn under the code of
    2026-10-03, and a check added that day, which refuses a clip listing one
    frame twice, moved the usable pool from 37,000 to 36,759. The seeded
    sample of the smaller pool shares NOT ONE clip with the older one,
    measured 2026-10-11 on job 47390003. Any rerun that has to land on the
    same clips passes them with `only`.
    """
    if only is not None:
        return sorted(v for v in only if v in records)
    pool = sorted(v for v, frames in records.items()
                  if template_of.get(v) in allowed
                  and why_not(frames, max_tracks, min_frames) is None)
    return sorted(random.Random(seed).sample(pool, min(count, len(pool))))


def _template_ids(labels_path, split_paths):
    """`{video id: template id}` from SSv2's labels and split files."""
    return _labels(labels_path, split_paths)[0]


def _labels(labels_path, split_paths):
    """`({video id: template id}, {video id: placeholders})`.

    The placeholders name the objects the template's `[something]` slots
    stand for, in order, which is how a PDDL argument reaches a track.
    """
    with open(labels_path) as f:
        label_id = json.load(f)
    tmpl, holders = {}, {}
    for path in split_paths:
        with open(path) as f:
            for row in json.load(f):
                text = row["template"].replace("[", "").replace("]", "")
                if text in label_id:
                    tmpl[str(row["id"])] = label_id[text]
                holders[str(row["id"])] = row.get("placeholders") or []
    return tmpl, holders


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
    ap.add_argument("--force-ids-out", action="store_true",
                    help="overwrite an existing ids list. Two tracked\n"
                         "lists were lost without this, OPEN-ISSUES I28")
    ap.add_argument("--only",
                    help="a file of clip ids, one per line. The conversion "
                         "takes exactly these and ignores --count and "
                         "--seed. A seed does not make this repeatable: the "
                         "usable pool moved from 37,000 to 36,759 when a "
                         "check was added on 2026-10-03, and the seeded "
                         "sample of the new pool shared no clip with the "
                         "old one.")
    ap.add_argument("--domain",
                    help="data/video/something_else/raw/twentybn_domain.pddl."
                         " Without it relation_instances stays empty, which "
                         "is what every clip held until 2026-10-11.")
    args = ap.parse_args(argv)

    with open(args.templates) as f:
        allowed = set(line.strip() for line in f if line.strip())
    template_of, placeholders_of = _labels(args.labels, args.splits)
    # The domain states the relations. Without it the converter writes the
    # boxes alone, which is what it did until 2026-10-11.
    domain_actions, domain_preds, action_of_template = {}, {}, {}
    if args.domain:
        from tools.video.pddl_domain import parse_domain, template_ids
        with open(args.domain) as f:
            domain_text = f.read()
        domain_preds, domain_actions = parse_domain(args.domain)
        action_of_template = dict((v, k) for k, v
                                  in template_ids(domain_text).items())
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
    only = None
    if args.only:
        with open(args.only) as f:
            only = [line.strip() for line in f if line.strip()]
    chosen = choose(records, template_of, allowed, args.count, args.seed,
                    args.max_tracks, args.min_frames, only=only)
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
        doc = clip_json(video, records[video], width, height)
        act = action_of_template.get(str(template_of.get(video)))
        if act:
            doc["relation_instances"] = relation_instances(
                doc, placeholders_of.get(video) or [], act,
                domain_actions, domain_preds)
        with open(os.path.join(ann_dir, video + ".json"), "w") as f:
            json.dump(doc, f)
        kept.append(video)
    with ids_out.open_ids_out(args.ids_out,
                              args.force_ids_out) as f:
        f.write("\n".join(kept) + "\n")
    print("wrote %d clips to %s, dropped %d: %s"
          % (len(kept), ann_dir, len(dropped), json.dumps(dropped)))
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
