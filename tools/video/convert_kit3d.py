#!/usr/bin/env python3
"""KIT Bimanual from its 3D boxes into the VidVRD format FOSAE reads.

WHY. The 2D converter, `tools/video/convert_kit.py`, reads `2d_objects`,
the YOLO detections. They name one box for each class, so a second bowl is
lost in each frame. They misname objects, for example a hammer as `knife`.
And they hold no hand. `3d_objects` holds tracked instances such as
`bowl_7` and `bowl_18`, and both hands as `LeftHand_N` and `RightHand_N`.
The author decided on 2026-10-05 to convert KIT from them. Plan
`notes/plans/2026-10-05-dataset-fixes.md`, F3.

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

    python3 tools/video/convert_kit3d.py --derived DERIVED --rgb RGB \\
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


def clip_json(clip, frames, start, size, width, height, category_of):
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
            "subject/objects": [{"tid": tid[n], "category": category_of[n]}
                                for n in names],
            "trajectories": trajectories, "relation_instances": []}


def _load(path):
    with open(path) as handle:
        return json.load(handle)


def convert_take(derived, base, width, height, max_error, stride=2):
    """`(frames, category_of, error)`, or `(None, None, reason)`."""
    n = len(glob.glob(os.path.join(derived, base, "3d_objects",
                                   "frame_*.json")))
    if n == 0:
        return None, None, "no 3d_objects"
    d3 = [_load(os.path.join(derived, base, "3d_objects",
                             "frame_%d.json" % f)) for f in range(n)]
    pairs = []
    for f in range(0, n, stride):
        p2 = os.path.join(derived, base, "2d_objects", "frame_%d.json" % f)
        if os.path.exists(p2):
            pairs += matched_centres(_load(p2), d3[f], width, height)
    if len(pairs) < 40:
        return None, None, "%d matched boxes, too few to fit" % len(pairs)
    X = np.array([p[0] for p in pairs])
    U = np.array([p[1] for p in pairs])
    error = heldout_error(X, U)
    if error > max_error:
        return None, None, "held-out error %.1f px" % error
    P = camera_matrix(fit_camera(X, U))
    frames = {}
    for f in range(n):
        row = {}
        for obj in d3[f]:
            box = projected_box(P, obj["bounding_box"], width, height)
            if box is not None:
                row[obj["instance_name"]] = (float(obj.get("certainty", 1.0)),
                                             box)
        frames[f] = row
    return frames, instance_classes(d3), error


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
    ap.add_argument("--size", type=int, default=150)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-error", type=float, default=10.0)
    args = ap.parse_args(argv)

    from PIL import Image
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from convert_kit import clip_id, window_starts
    ann_dir = os.path.join(args.out, "annotations")
    frames_root = os.path.join(args.out, "frames_%dfps" % args.fps)
    os.makedirs(ann_dir, exist_ok=True)
    kept, dropped, errors = [], {}, {}
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
        frames, category_of, error = convert_take(args.derived, base, width,
                                                  height, args.max_error)
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
    with open(args.ids_out, "w") as handle:
        handle.write("\n".join(sorted(kept)) + "\n")
    print("wrote %d clips. held-out px error by take: %s. dropped takes: %s"
          % (len(kept), json.dumps(errors), json.dumps(dropped)))
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
