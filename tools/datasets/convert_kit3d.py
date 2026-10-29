#!/usr/bin/env python3
"""KIT Bimanual from its 3D boxes into the VidVRD format FOSAE reads.

WHY. The 2D converter, `tools/datasets/convert_kit.py`, reads `2d_objects`,
the YOLO detections. They name one box for each class, so a second bowl is
lost in each frame. They misname objects, for example a hammer as `knife`.
And they hold no hand. `3d_objects` holds tracked instances such as
`bowl_7` and `bowl_18`, and both hands as `LeftHand_N` and `RightHand_N`.
The author decided on 2026-10-05 to convert KIT from them. Plan
`notes/plans/2026-10-05-dataset-fixes.md`, F3 (archived, see workbench/ARCHIVE.md).

THE CAMERA. The 3D boxes are axis-aligned boxes in mm, and the release ships
no camera values. Each take fits its own pinhole camera, focal lengths,
principal point, rotation and translation, by robust least squares from the
3D box centres to the centres of the 2D box of the same class, on frames
where that class has exactly one box in each. Measured on 2026-10-05: one
camera for all takes misses the takes of other subjects by a median of
about 29 px, and a camera for each take, fit on the first half of its
frames, misses the second half by a median of 5.1 px over 15 takes. So a
take whose held-out error is above `--max-error` px is dropped.

THE RULES THIS FILE ADDS, each a choice of this project:
  identity   the track is the `instance_name`. Its category is the class it
             carries most often, because the class of one instance can change
             between frames
  box        the projected 3D box: the rectangle around its 8 projected
             corners, clipped to the image. It is looser than the YOLO box,
             median IoU 0.55, and it is the one source that also boxes the
             hands
  frame rate takes whose `rgb/metadata.csv` gives another rate than
             `--fps` are dropped. Most takes of subject 1 run at 15 fps
  windows    every whole 150-frame window, named as `convert_kit.py` names
             them

    python3 tools/datasets/convert_kit3d.py --derived DERIVED --rgb RGB \\
        --takes 2:task_5_k_cereals:0 ... --out OUT --ids-out IDS
"""

import argparse
import collections
import glob
import json
import math
import os
import sys

import numpy as np
from tools.datasets import ids_out

# The camera looks down -Z of the 3D frame: KIT's object depths are about
# -1000 to -1700 mm. This flip makes the depth positive in front of it.
_FLIP = np.diag([1.0, -1.0, -1.0])
_Q0 = np.array([550.0, 550.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def _rotation(r):
    rx, ry, rz = r
    cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry),
                              math.sin(ry), math.cos(rz), math.sin(rz))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz.dot(Ry).dot(Rx)


def camera_matrix(q):
    """The 3x4 projection of `q = (fx, fy, cx, cy, rx, ry, rz, tx, ty, tz)`."""
    K = np.array([[q[0], 0.0, q[2]], [0.0, q[1], q[3]], [0.0, 0.0, 1.0]])
    R = _FLIP.dot(_rotation(q[4:7]))
    t = _FLIP.dot(np.asarray(q[7:10], dtype=float))
    return K.dot(np.c_[R, t])


def _homogeneous(P, X):
    return np.c_[np.asarray(X, dtype=float), np.ones(len(X))].dot(P.T)


def project(P, X):
    """Pixel coordinates of the 3D points `X`, shape (n, 2)."""
    h = _homogeneous(P, X)
    return h[:, :2] / h[:, 2:3]


def fit_camera(X, U, q0=None):
    """`q` that projects `X` onto `U`, robust to a few wrong matches."""
    from scipy.optimize import least_squares
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)

    def residual(q):
        return (project(camera_matrix(q), X) - U).ravel()

    best = None
    starts = [q0] if q0 is not None else [
        _Q0 * np.array([sx, sy, 1, 1, 1, 1, 1, 1, 1, 1])
        for sx in (1.0, -1.0) for sy in (1.0, -1.0)]
    for start in starts:
        q = least_squares(residual, start, loss="soft_l1", f_scale=10.0).x
        err = np.median(np.hypot(*residual(q).reshape(-1, 2).T))
        if best is None or err < best[0]:
            best = (err, q)
    return best[1]


def oriented(P, X):
    """`P` or `-P`, whichever puts the points `X` in front of the camera.

    Both give the same pixels, so a fit may return either. Found on
    2026-10-05: with the wrong one, `projected_box` refused every box of 47
    of 66 windows as behind the camera.
    """
    return -P if np.median(_homogeneous(P, X)[:, 2]) < 0 else P


def heldout_error(X, U):
    """Median px error on the second half when fit on the first half."""
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    half = len(X) // 2
    q = fit_camera(X[:half], U[:half])
    return float(np.median(np.hypot(*(project(camera_matrix(q), X[half:])
                                      - U[half:]).T)))


def box_corners(b):
    return np.array([[x, y, z] for x in (b["x0"], b["x1"])
                     for y in (b["y0"], b["y1"]) for z in (b["z0"], b["z1"])],
                    dtype=float)


def projected_box(P, b, width, height):
    """`(x1, y1, x2, y2)` in whole pixels, or None when the box is behind
    the camera or wholly outside the image."""
    h = _homogeneous(P, box_corners(b))
    if np.any(h[:, 2] <= 0):
        return None
    uv = h[:, :2] / h[:, 2:3]
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    x1, x2 = int(round(max(x1, 0.0))), int(round(min(x2, float(width))))
    y1, y2 = int(round(max(y1, 0.0))), int(round(min(y2, float(height))))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def matched_centres(d2, d3, width, height):
    """`[(3D centre, 2D centre)]` for each class with one box in each."""
    by2, by3 = collections.defaultdict(list), collections.defaultdict(list)
    for det in d2:
        cand = (det.get("candidates") or [{}])[0]
        if cand.get("class_name"):
            b = det["bounding_box"]
            by2[cand["class_name"]].append((b["x"] * width, b["y"] * height))
    for obj in d3:
        by3[obj["class_name"]].append(box_corners(obj["bounding_box"])
                                      .mean(axis=0))
    return [(by3[c][0], np.array(by2[c][0])) for c in sorted(by2)
            if len(by2[c]) == 1 and len(by3.get(c, ())) == 1]


def instance_classes(frames3d):
    """`{instance_name: the class it carries most often}`."""
    seen = collections.defaultdict(collections.Counter)
    for frame in frames3d:
        for obj in frame:
            seen[obj["instance_name"]][obj["class_name"]] += 1
    return dict((name, counts.most_common(1)[0][0])
                for name, counts in seen.items())


def take_fps(text):
    for line in text.splitlines():
        parts = line.strip().split(",")
        if len(parts) == 3 and parts[0] == "fps":
            return int(parts[2])
    return None


def relation_instances(d3, relations, tracks, start, size, tid_of):
    """KIT's own relations over the clip's tracks, in the VidVRD field.

    `d3` and `relations` are indexed by the frame of the TAKE, and
    `subject_index` and `object_index` point into `d3` OF THE SAME FRAME, so
    index 0 is a different instance in another frame. `begin_fid` counts from
    the start of the clip and `end_fid` is one past the last frame, which is
    how VidVRD writes them. A relation whose subject or object is not one of
    `tracks` is dropped, and an unbroken run of frames is one instance.

    The relations are written as KIT computes them, with no smoothing.
    Measured 2026-10-10 on 7 role clips: the median instance lasts 1 frame
    and 57 per cent last a single frame. Two causes, and the first draft of
    this note gave only one. KIT scores each frame on its own, so a pair near
    a threshold flickers. AND the track itself drops out of `3d_objects`: in
    160 of the 199 role clips, 80 per cent, some frame holds no entry for one
    of the three roles, 5,147 frame-and-track pairs in all, and every relation
    of that track ends there. Smoothing either one would change the ground
    truth, so it is a separate decision. Plan
    notes/plans/2026-10-10-relations-and-boxes.md, A1
    (archived, see workbench/ARCHIVE.md).
    """
    keep = set(tracks)
    open_at, out = {}, []

    def close(key, end):
        begin = open_at.pop(key)
        out.append({"predicate": key[2], "subject_tid": tid_of[key[0]],
                    "object_tid": tid_of[key[1]], "begin_fid": begin,
                    "end_fid": end})

    for off in range(size):
        f = start + off
        objs = d3[f] if f < len(d3) else []
        rows = relations[f] if f < len(relations) else []
        live = set()
        for r in rows:
            i, j = r["subject_index"], r["object_index"]
            if not 0 <= i < len(objs) or not 0 <= j < len(objs):
                continue
            a, b = objs[i]["instance_name"], objs[j]["instance_name"]
            if a in keep and b in keep:
                live.add((a, b, r["relation_name"]))
        for key in sorted(set(open_at) - live):
            close(key, off)
        for key in sorted(live):
            open_at.setdefault(key, off)
    for key in sorted(open_at):
        close(key, size)
    out.sort(key=lambda r: (r["begin_fid"], r["subject_tid"],
                            r["object_tid"], r["predicate"]))
    return out


def clip_json(clip, frames, start, size, width, height, category_of,
              relation_source=None):
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
    rels = []
    if relation_source is not None:
        d3, relations, tracks = relation_source
        rels = relation_instances(d3, relations, tracks, start, size, tid)
    return {"video_id": clip, "width": int(width), "height": int(height),
            "subject/objects": [{"tid": tid[n], "category": category_of[n]}
                                for n in names],
            "trajectories": trajectories, "relation_instances": rels}


# The 14 actions of the labels, in the order of `dreher2019` p.3. The label
# number is the place in this list: the right hand of
# `subject_2/task_5_k_cereals/take_0` reads approach, lift, pour, retreat.
KIT_ACTIONS = ("idle", "approach", "retreat", "lift", "place", "hold",
               "stir", "pour", "cut", "drink", "wipe", "hammer", "saw",
               "screw")
TOUCH = ("contact", "moving together", "fixed moving together")


def read_actions(doc):
    """`{hand: [(start, end, action)]}` of one label file, where the list
    alternates a start frame and an action number and ends with the last
    frame."""
    out = {}
    for hand, flat in doc.items():
        if not isinstance(flat, list):
            continue
        starts, actions = flat[0:-1:2], flat[1:-1:2]
        ends = [s - 1 for s in starts[1:]] + [flat[-1] - 1]
        # Two label files of takes 0 to 4 hold null for one segment; it
        # stays None, an unlabelled stretch.
        out[hand] = [(a, b, None if k is None else KIT_ACTIONS[k])
                     for a, b, k in zip(starts, ends, actions)]
    return out


def cycles(segments):
    """`[(start, end)]`, each a run of actions of one hand from one idle to
    the next: approach, lift, pour, place and retreat around one object.

    A single action is short at 30 fps: 194 of the actions of 41 takes ran
    under 30 frames on 2026-10-05. A cycle is the unit HA-ViD calls a
    primitive task."""
    out, current = [], None
    for start, end, action in segments:
        if action in ("idle", None):
            if current:
                out.append(current)
            current = None
        else:
            current = (current[0], end) if current else (start, end)
    if current:
        out.append(current)
    return out


def _touch_counts(frames3d, relations, start, end, who):
    """`{instance: frames in which it touches an instance named by who}`."""
    counts = collections.Counter()
    for f in range(start, end + 1):
        names = [o["instance_name"] for o in frames3d[f]]
        seen = set()
        for r in relations[f]:
            if r["relation_name"] not in TOUCH:
                continue
            a, b = names[r["subject_index"]], names[r["object_index"]]
            for x, y in ((a, b), (b, a)):
                if who(x) and "Hand" not in y:
                    seen.add(y)
        counts.update(seen)
    return counts


def _top(counter):
    """The most common key, and the FIRST NAME when counts tie.

    `Counter.most_common` returns equal counts in insertion order, and these
    counters are filled from a set of instance names, which Python orders by
    hash. An adversary ran one real clip under three values of
    PYTHONHASHSEED on 2026-10-11 and got three different target objects,
    `banana_10`, `cereals_2` and `knife_6`, all four candidates touching the
    moved object in 80 of 80 frames. Two of the 199 role clips hold such a
    tie. The name settles it, so the conversion repeats.
    """
    return min(counter, key=lambda k: (-counter[k], k))


def kit_role_tracks(frames3d, relations, start, end, hand_prefix,
                    min_share=0.5):
    """`(hand, moved, target)` instances of one action, or None.

    The moved object touches the acting hand in at least `min_share` of
    the frames, and the target is the object that touches the moved object
    most often, in at least one frame in five."""
    n = end - start + 1
    hands = collections.Counter(o["instance_name"] for f in range(start, end + 1)
                                for o in frames3d[f]
                                if o["instance_name"].startswith(hand_prefix))
    if not hands:
        return None
    hand = _top(hands)
    moved = _touch_counts(frames3d, relations, start, end,
                          lambda x: x == hand)
    if not moved or moved[_top(moved)] < min_share * n:
        return None
    m = _top(moved)
    target = _touch_counts(frames3d, relations, start, end,
                           lambda x: x == m)
    target.pop(m, None)
    if not target or target[_top(target)] < 0.2 * n:
        return None
    return (hand, m, _top(target))


def _load(path):
    with open(path) as handle:
        return json.load(handle)


def convert_take(derived, base, width, height, max_error, stride=2):
    """`(frames, category_of, error, frames3d)`, or
    `(None, None, reason, None)`."""
    n = len(glob.glob(os.path.join(derived, base, "3d_objects",
                                   "frame_*.json")))
    if n == 0:
        return None, None, "no 3d_objects", None
    d3 = [_load(os.path.join(derived, base, "3d_objects",
                             "frame_%d.json" % f)) for f in range(n)]
    pairs = []
    for f in range(0, n, stride):
        p2 = os.path.join(derived, base, "2d_objects", "frame_%d.json" % f)
        if os.path.exists(p2):
            pairs += matched_centres(_load(p2), d3[f], width, height)
    if len(pairs) < 40:
        return None, None, "%d matched boxes, too few to fit" % len(pairs), None
    X = np.array([p[0] for p in pairs])
    U = np.array([p[1] for p in pairs])
    error = heldout_error(X, U)
    if error > max_error:
        return None, None, "held-out error %.1f px" % error, None
    P = oriented(camera_matrix(fit_camera(X, U)), X)
    frames = {}
    for f in range(n):
        row = {}
        for obj in d3[f]:
            box = projected_box(P, obj["bounding_box"], width, height)
            if box is not None:
                row[obj["instance_name"]] = (float(obj.get("certainty", 1.0)),
                                             box)
        frames[f] = row
    return frames, instance_classes(d3), error, d3


def role_clips(derived, labels, base, clip_stem, frames, d3, category_of,
               size, min_length, min_coverage, refused):
    """`[(clip, start, length, frames, relation_source)]`: one clip for each
    cycle of a hand, from one idle to the next, with only the hand, the moved
    object and the target. `refused` counts the others by reason.
    `relation_source` is `(d3, relations, tracks)` for `clip_json`, so KIT's
    own relations reach the file by this path and by no other. The window
    path of `main` holds every object of the frame, and no measurement covers
    its relation count."""
    def refuse(reason):
        refused[reason] = refused.get(reason, 0) + 1

    path = os.path.join(labels, base + ".json")
    if not os.path.exists(path):
        refuse("no label file")
        return []
    relations = [_load(os.path.join(derived, base, "spatial_relations",
                                    "frame_%d.json" % f))
                 if os.path.exists(os.path.join(derived, base,
                                                "spatial_relations",
                                                "frame_%d.json" % f)) else []
                 for f in range(len(d3))]
    out = []
    for hand, segments in sorted(read_actions(_load(path)).items()):
        prefix = {"right_hand": "RightHand", "left_hand": "LeftHand"}[hand]
        for start, end in cycles(segments):
            if end - start + 1 < min_length:
                refuse("shorter than %d frames" % min_length)
                continue
            end = min(end, start + size - 1, len(d3) - 1)
            # Approach and retreat touch nothing, so a cycle holds frames
            # without contact, and the share asked of the moved object is
            # lower than for one action.
            tracks = kit_role_tracks(d3, relations, start, end, prefix,
                                     min_share=0.3)
            if tracks is None:
                refuse("no moved object or no target by contact")
                continue
            share = min(sum(1 for f in range(start, end + 1)
                            if t in frames.get(f, {})) / float(end - start + 1)
                        for t in tracks)
            if share < min_coverage:
                refuse("a role is boxed in under %d%% of the frames"
                       % int(100 * min_coverage))
                continue
            keep = dict((f, dict((t, frames[f][t]) for t in tracks
                                 if t in frames.get(f, {})))
                        for f in range(start, end + 1))
            clip = "%s_c%06d_%s" % (clip_stem, start,
                                    "rh" if hand == "right_hand" else "lh")
            out.append((clip, start, end - start + 1, keep,
                        (d3, relations, tracks)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--derived", required=True,
                    help="data/video/kit_bimanual/ext/bimacs_derived_data")
    ap.add_argument("--rgb", required=True,
                    help="data/video/kit_bimanual/rgb_ext/bimacs_rgbd_data")
    ap.add_argument("--takes", nargs="+", required=True,
                    help="subject:task:take, for example 2:task_5_k_cereals:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids-out", required=True)
    ap.add_argument("--force-ids-out", action="store_true",
                    help="overwrite an existing ids list. Two tracked\n"
                         "lists were lost without this, OPEN-ISSUES I28")
    ap.add_argument("--size", type=int, default=150)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-error", type=float, default=10.0)
    ap.add_argument("--roles", action="store_true",
                    help="one clip for each action of a hand that touches an "
                         "object, with only the hand, the moved object and "
                         "the target")
    ap.add_argument("--labels", default=None,
                    help="ext/bimacs_rgbd_data_ground_truth, for --roles")
    ap.add_argument("--min-length", type=int, default=30)
    ap.add_argument("--min-coverage", type=float, default=0.8)
    args = ap.parse_args(argv)

    from PIL import Image
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from convert_kit import clip_id, window_starts
    ann_dir = os.path.join(args.out,
                           "annotations_roles" if args.roles else "annotations")
    frames_root = os.path.join(args.out, "frames_%dfps" % args.fps)
    os.makedirs(ann_dir, exist_ok=True)
    kept, dropped, errors, refused = [], {}, {}, {}
    for spec in args.takes:
        subject, task, take = spec.split(":")
        subject, take = int(subject), int(take)
        base = "subject_%d/%s/take_%d" % (subject, task, take)
        meta = os.path.join(args.rgb, base, "rgb", "metadata.csv")
        if not os.path.exists(meta):
            dropped[spec] = "no rgb/metadata.csv"
            continue
        with open(meta) as handle:
            fps = take_fps(handle.read())
        if fps != args.fps:
            dropped[spec] = "%s fps" % fps
            continue
        pngs = dict((int(os.path.basename(p)[len("frame_"):-len(".png")]), p)
                    for p in glob.glob(os.path.join(args.rgb, base, "rgb",
                                                    "chunk_*", "frame_*.png")))
        if not pngs:
            dropped[spec] = "no rgb frames"
            continue
        with Image.open(pngs[min(pngs)]) as im:
            width, height = im.size
        frames, category_of, error, d3 = convert_take(
            args.derived, base, width, height, args.max_error)
        if frames is None:
            dropped[spec] = error
            continue
        if sorted(pngs) != list(range(len(frames))):
            dropped[spec] = "%d box files, %d frames" % (len(frames),
                                                         len(pngs))
            continue
        errors[spec] = round(error, 1)
        whole = os.path.join(frames_root, "_video", "s%d_%s_take%d"
                             % (subject, task, take))
        os.makedirs(whole, exist_ok=True)
        for f in range(len(frames)):
            dst = os.path.join(whole, "%06d.jpg" % f)
            if not os.path.exists(dst):
                with Image.open(pngs[f]) as im:
                    im.convert("RGB").save(dst, quality=92)
        if args.roles:
            stem = "s%d_%s_take%d" % (subject, task, take)
            for clip, start, length, keep, rel_src in role_clips(
                    args.derived, args.labels, base, stem, frames, d3,
                    category_of, args.size, args.min_length,
                    args.min_coverage, refused):
                link_dir = os.path.join(frames_root, clip)
                os.makedirs(link_dir, exist_ok=True)
                for k in range(length):
                    dst = os.path.join(link_dir, "%06d.jpg" % k)
                    if not os.path.lexists(dst):
                        os.symlink(os.path.join(
                            "..", "_video", os.path.basename(whole),
                            "%06d.jpg" % (start + k)), dst)
                with open(os.path.join(ann_dir, clip + ".json"), "w") as f:
                    json.dump(clip_json(clip, keep, start, length, width,
                                        height, category_of, rel_src), f)
                kept.append(clip)
            continue
        for start in window_starts(len(frames), args.size):
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
                                    height, category_of), handle)
            kept.append(clip)
    with ids_out.open_ids_out(args.ids_out,
                              args.force_ids_out) as handle:
        handle.write("\n".join(sorted(kept)) + "\n")
    print("wrote %d clips. held-out px error by take: %s. dropped takes: %s"
          % (len(kept), json.dumps(errors), json.dumps(dropped)))
    if args.roles:
        print("refused actions, by reason: %s"
              % json.dumps(refused, sort_keys=True))
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
