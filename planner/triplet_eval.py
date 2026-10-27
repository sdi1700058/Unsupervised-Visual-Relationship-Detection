#!/usr/bin/env python3
"""Triplet mAP of the four rungs, scored by the official VidVRD evaluator.

The design is `workbench/notes/reports/2026-10-25-triplet-metrics.md`, parts
B.2 and B.4. The author approved it on 2026-10-25.

**The evaluator.** `evaluate()` in `evaluation/video_relation_detection.py`
of NExTplusplus/VidVRD-helper, commit b556513, gives every reported number.
Its files stay unchanged. The venv has no `tqdm`, so a do-nothing module
takes its place in `sys.modules` before the import. xdshang/VidVRD-helper,
commit 1b4de17, scores the same files as a cross-check. Both clones are in
`data/deps/eval-src/`. A clone at another commit, or with a modified file,
stops the run.

**The predictors.** All four take the true box tracks and the true
categories, so the regime is oracle trajectory (VrdONE-oracle, 43.15).

- `perfect` returns the truth with score 1.0. With no cut it must give mAP
  1.000. Any other value means that the harness is broken.
- `frequency` gives each ordered pair of true tracks the 10 most frequent
  predicates of its category pair in the train clips. It sees no pixel.
- `ridge` and `onetoone` name the model relations. A model relation is
  (p, a, b): unit u binds slots a and b, and its bit (u, p) is on. The slot
  map of the bake gives the tracks. `ridge` fits the ridge readout of
  `predicate_probe.py` on the train clips. `onetoone` gives each FOSAE
  predicate the one name that holds most often when the predicate holds,
  on the train clips.

**From frames to instances.** A name is on for a track pair at frame t when
its score reaches the threshold of the name. Consecutive frames that are on
for one track pair and one name give one instance [t_b, t_e). Its score is
the mean of its frame scores.

**Clips.** `split_info` of the training run gives the test and validation
clips. The test clips give the headline row. The test and validation clips
together give the second row. The readouts and the prior fit on the train
clips only.

    .venv-local/bin/python planner/triplet_eval.py --dataset se \\
        --out workbench/eval/triplet/2026-10-25

numpy and the standard library, Python 3.6. matplotlib only for --figures.
"""

import argparse
import contextlib
import importlib
import io
import json
import os
import subprocess
import sys
import types
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from planner.predicate_probe import ridge_probe_multi   # noqa: E402
from planner.relation_oracle import slot_of_tid          # noqa: E402

EVAL_SRC = os.path.join(ROOT, "data", "deps", "eval-src")
OFFICIAL = {
    "repo": "NExTplusplus/VidVRD-helper",
    "dir": "NExTplusplus_VidVRD-helper",
    "module": "video_relation_detection",
    "commit": "b556513ba4b1a0d46301bc1eda0546cceef2716c",
}
CROSSCHECK = {
    "repo": "xdshang/VidVRD-helper",
    "dir": "xdshang_VidVRD-helper",
    "module": "visual_relation_detection",
    "commit": "1b4de175ce6e7a103d5feaae66b68d32a306a877",
}

EXPORTS = os.path.join("workbench", "eval", "exports", "2026-10-11")
RUNS = os.path.join("out", "learnability", "zs0", "rep2", "video", "vidvrd")
DATASETS = {
    "se": ("se_latents.npz",
           "data/video/something_else/vidvrd_format/annotations"),
    "kit": ("kit_latents.npz",
            "data/video/kit_bimanual/vidvrd_format_3d/annotations_roles"),
    "havid": ("havid_latents.npz",
              "data/video/ha_vid/vidvrd_format/annotations_roles"),
    "vidvrd": ("vidvrd_latents.npz", "data/video/vidvrd/annotations/train"),
}
LADDER = ("se", "kit", "havid", "vidvrd")
LABELS = {"se": "SE", "kit": "KIT", "havid": "HAV", "vidvrd": "VRD"}
PREDICTORS = ("perfect", "frequency", "ridge", "onetoone")
CLIPSETS = ("test", "testval")
NUM_SLOTS = 3
TOP_K = 10
TOP_RETURNS = 200
RIDGE_ALPHA = 1.0
CANVAS_W, CANVAS_H = 300, 200
PUBLISHED = (("VidVRD 2017", 8.58), ("VrdONE 2024", 31.33),
             ("VrdONE oracle", 43.15))


# ------------------------------------------------------------ the evaluator

def install_tqdm_stub():
    """Put a do-nothing `tqdm` into `sys.modules` when the package is absent.

    The official evaluator imports `tqdm` at line 2 and uses it only for a
    progress bar. Returns True when the stub goes in.
    """
    try:
        import tqdm  # noqa: F401
        return False
    except ImportError:
        stub = types.ModuleType("tqdm")
        stub.tqdm = lambda iterable=None, *args, **kwargs: iterable
        sys.modules["tqdm"] = stub
        return True


def load_evaluator(spec, eval_src=EVAL_SRC):
    """The `evaluate` function of one clone, imported without `__init__.py`.

    Each clone calls its package `evaluation`, so each clone gets its own
    alias in `sys.modules`. The `__init__.py` of a clone imports modules that
    this run does not use, so the alias is an empty package with the right
    path. No byte code goes into the clone.
    """
    folder = os.path.join(eval_src, spec["dir"], "evaluation")
    if not os.path.isfile(os.path.join(folder, spec["module"] + ".py")):
        raise SystemExit("no evaluator at %s" % folder)
    install_tqdm_stub()
    alias = "_triplet_eval_" + spec["dir"].replace("-", "_")
    if alias not in sys.modules:
        package = types.ModuleType(alias)
        package.__path__ = [folder]
        sys.modules[alias] = package
    before = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module = importlib.import_module(alias + "." + spec["module"])
    finally:
        sys.dont_write_bytecode = before
    return module.evaluate


def clone_state(spec, eval_src=EVAL_SRC):
    """`{"commit": HEAD, "clean": True when no tracked file is modified}`.

    `cwd` and not `git -C`, because an srun step can see only the system
    git 1.8.3, and `-C` starts at git 1.8.5.
    """
    repo = os.path.join(eval_src, spec["dir"])
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
        changed = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "clean": None}
    return {"commit": head, "clean": changed == ""}


def pinned_evaluator(spec, eval_src=EVAL_SRC):
    """`(evaluate, record)` of one clone at its pinned commit.

    A clone at another commit, or with a modified tracked file, stops the
    run. `record` goes into the metrics file.
    """
    state = clone_state(spec, eval_src)
    if state["commit"] != spec["commit"] or state["clean"] is not True:
        raise SystemExit("%s: expected commit %s and no modified file, "
                         "found %s" % (spec["repo"], spec["commit"], state))
    record = dict(spec)
    record.update(state)
    return load_evaluator(spec, eval_src), record


def score(evaluate, groundtruth, prediction, **kwargs):
    """The six numbers of one evaluator call.

    The evaluator prints its numbers. The print goes nowhere, and the
    returned values are the record.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        mean_ap, recall, precision = evaluate(groundtruth, prediction,
                                              **kwargs)
    return {"mAP": float(mean_ap), "R@50": float(recall[50]),
            "R@100": float(recall[100]), "P@1": float(precision[1]),
            "P@5": float(precision[5]), "P@10": float(precision[10])}


def clip_aps(evaluate, groundtruth, prediction, **kwargs):
    """AP of each clip with truth: the evaluator's mAP on that clip alone."""
    return dict(
        (vid, score(evaluate, {vid: gt}, {vid: prediction.get(vid, [])},
                    **kwargs)["mAP"])
        for vid, gt in sorted(groundtruth.items()) if gt)


# ------------------------------------------- copied from m7_map.py at fc39ea3
#
# The truth reader and the perfect and frequency predictors of
# `planner/m7_map.py`, lines 247-334 and 478-557, byte for byte. That
# module goes to the archive after one cross-check on Something-Else.

def _boxes_per_frame(doc):
    """Frame index -> {tid: (xmin, ymin, xmax, ymax)}."""
    out = []
    for frame in doc.get("trajectories", []):
        row = {}
        for roi in (frame or []):
            b = roi["bbox"]
            row[roi["tid"]] = (b["xmin"], b["ymin"], b["xmax"], b["ymax"])
        out.append(row)
    return out


def relation_instances(doc):
    """The ground-truth relation instances of one annotation document.

    This is `Dataset.get_relation_insts` from the reference toolkit: one
    instance per entry of `relation_instances`, the triplet given as
    **category names** rather than track ids, the duration half open, and the
    two trajectories sliced out of the per-frame boxes.

    An instance is dropped where a box is missing on some frame of its
    interval. Interpolating one would change the volume the metric divides
    by, and dropping is the only choice that cannot inflate a score.
    """
    category = dict((so["tid"], so["category"])
                    for so in doc.get("subject/objects", []))
    frames = _boxes_per_frame(doc)

    out = []
    for rel in doc.get("relation_instances", []):
        begin = int(rel["begin_fid"])
        end = int(rel["end_fid"])
        s_tid = rel["subject_tid"]
        o_tid = rel["object_tid"]
        if begin < 0 or end > len(frames) or end <= begin:
            continue
        window = frames[begin:end]
        if any(s_tid not in f or o_tid not in f for f in window):
            continue
        out.append({
            "triplet": (category[s_tid], rel["predicate"], category[o_tid]),
            "subject_tid": s_tid,
            "object_tid": o_tid,
            "duration": (begin, end),
            "sub_traj": [f[s_tid] for f in window],
            "obj_traj": [f[o_tid] for f in window],
        })
    return out


def tubelets(doc):
    """One tubelet per track: its category, its span and its boxes.

    The span is the **longest run of consecutive frames** the track is
    annotated on. A track that vanishes and returns would otherwise get a
    trajectory shorter than the duration it claims, and `viou` would compare
    the wrong frames.
    """
    category = dict((so["tid"], so["category"])
                    for so in doc.get("subject/objects", []))
    frames = _boxes_per_frame(doc)

    runs = {}
    current = {}
    for f, row in enumerate(frames):
        for tid in list(current):
            if tid not in row:
                del current[tid]
        for tid in row:
            if tid not in current:
                current[tid] = f
            begin = current[tid]
            best = runs.get(tid)
            if best is None or (f + 1 - begin) > (best[1] - best[0]):
                runs[tid] = (begin, f + 1)

    out = []
    for tid in sorted(runs):
        if tid not in category:
            continue
        begin, end = runs[tid]
        out.append({
            "tid": tid,
            "category": category[tid],
            "duration": (begin, end),
            "traj": [frames[f][tid] for f in range(begin, end)],
        })
    return out


def perfect_predictions(gt_relations):
    """The ground truth returned as its own prediction, every score 1.0."""
    out = []
    for gt in gt_relations:
        pred = dict(gt)
        pred["score"] = 1.0
        out.append(pred)
    return out


class TripletPrior(object):
    """How often each `(subject, predicate, object)` appears in a split.

    The no-vision predictor. It answers "what relation usually holds between
    a dog and a frisbee", which needs no pixels at all, so whatever it scores
    is the part of the metric that carries no information about the model.
    """

    def __init__(self):
        self.by_pair = defaultdict(lambda: defaultdict(int))
        self.overall = defaultdict(int)
        self.n_videos = 0

    def count(self, doc):
        category = dict((so["tid"], so["category"])
                        for so in doc.get("subject/objects", []))
        self.n_videos += 1
        for rel in doc.get("relation_instances", []):
            s = category.get(rel["subject_tid"])
            o = category.get(rel["object_tid"])
            if s is None or o is None:
                continue
            self.by_pair[(s, o)][rel["predicate"]] += 1
            self.overall[rel["predicate"]] += 1

    def top(self, subject, obj, k):
        """The k most frequent predicates for this category pair.

        An unseen pair falls back to the dataset-wide predicate frequency,
        because returning nothing would give the pair a free zero rather than
        the honest guess a frequency model would make.
        """
        counts = self.by_pair.get((subject, obj)) or self.overall
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]


def frequency_predictions(doc, prior, top_k=10, max_per_video=200):
    """The frequency prior applied to every ordered pair of ground-truth tubelets.

    The duration of a prediction is the span the two tubelets share, so the
    prediction is as well localised as the tubelets allow. Ordering is fully
    deterministic, including the ties, which are many: a frequency model
    gives whole groups of triplets the same score, and an arbitrary order
    among them would make the mAP irreproducible.
    """
    tubs = tubelets(doc)
    out = []
    for sub in tubs:
        for obj in tubs:
            if sub["tid"] == obj["tid"]:
                continue
            begin = max(sub["duration"][0], obj["duration"][0])
            end = min(sub["duration"][1], obj["duration"][1])
            if end <= begin:
                continue
            s_off = begin - sub["duration"][0]
            o_off = begin - obj["duration"][0]
            n = end - begin
            for predicate, count in prior.top(sub["category"],
                                              obj["category"], top_k):
                out.append({
                    "triplet": (sub["category"], predicate, obj["category"]),
                    "duration": (begin, end),
                    "sub_traj": sub["traj"][s_off:s_off + n],
                    "obj_traj": obj["traj"][o_off:o_off + n],
                    "score": float(count),
                })
    out.sort(key=lambda r: (-r["score"], r["triplet"], r["duration"]))
    return out[:max_per_video]


# ------------------------------------------------- split, export and slots

def read_split(path):
    """`{"train", "val", "test"}` clip lists of one training run.

    `split_info` names the test clips and the validation clips. The train
    clips are the other entries of `video_ids`. A train count that differs
    from `split_info.clips.train` stops the run.
    """
    with open(path) as f:
        doc = json.load(f)
    info = doc["split_info"]
    test = sorted(info["test_clips"])
    val = sorted(info["val_clips"])
    held = set(test) | set(val)
    train = sorted(c for c in doc["video_ids"] if c not in held)
    if len(train) != info["clips"]["train"]:
        raise SystemExit("%s: %d train clips, split_info says %d"
                         % (path, len(train), info["clips"]["train"]))
    return {"train": train, "val": val, "test": test}


def read_export(path):
    """One export, cut into clips.

    Returns `{"model_name", "P", "clips"}`. `clips[c]` holds `frames`, the
    frame index t of each row, and the rows of `latents`, `bindings` and
    `gt_boxes`, in frame order. A row id has the form `<clip>/<t:06d>`.
    """
    z = np.load(path)
    rows = defaultdict(list)
    for row, fid in enumerate(z["frame_ids"]):
        clip, frame = str(fid).rsplit("/", 1)
        rows[clip].append((int(frame), row))
    latents, bindings, boxes = z["latents"], z["bindings"], z["gt_boxes"]
    clips = {}
    for clip, pairs in rows.items():
        pairs.sort()
        index = np.array([r for _, r in pairs], dtype=int)
        clips[clip] = {"frames": np.array([t for t, _ in pairs], dtype=int),
                       "latents": latents[index],
                       "bindings": bindings[index],
                       "gt_boxes": boxes[index]}
    return {"model_name": str(z["model_name"]), "P": int(z["P"]),
            "clips": clips}


def dataset_spec(name, root=ROOT):
    """The export, truth directory and split file of one rung (report B.4)."""
    export, truth = DATASETS[name]
    export = os.path.join(root, EXPORTS, export)
    model_name = str(np.load(export)["model_name"])
    return {"name": name, "export": export,
            "truth": os.path.join(root, truth),
            "split": os.path.join(root, RUNS, model_name,
                                  "loaded_videos.json")}


def track_of_slot(doc, num_slots=NUM_SLOTS):
    """`{slot: tid}`: the inverse of the slot map of the bake."""
    return dict((slot, tid)
                for tid, slot in slot_of_tid(doc, num_slots).items())


def letterbox(box, width, height):
    """A source box on the 300 by 200 canvas, as the bake places it.

    A copy of `_scale_bbox_to_canvas` in
    `model/puzzles/puzzle_labeled_objects.py`: one scale for both axes,
    centred, rounded half up. A copy, because that module is part of the
    training package.
    """
    s = min(CANVAS_W / float(width), CANVAS_H / float(height))
    off_x = (CANVAS_W - width * s) / 2.0
    off_y = (CANVAS_H - height * s) / 2.0

    def px(value, limit):
        return int(np.clip(np.floor(value + 0.5), 0, limit - 1))

    return (px(box[0] * s + off_x, CANVAS_W), px(box[1] * s + off_y, CANVAS_H),
            px(box[2] * s + off_x, CANVAS_W), px(box[3] * s + off_y, CANVAS_H))


def slot_box_mismatches(doc, clip):
    """`(checked, off)` for one clip.

    `checked` counts the slot boxes of the export that the slot map fills.
    `off` counts those that differ by more than 1 canvas pixel from the
    letterboxed truth box of their track. `off` is 0 when the slot map of
    this module is the slot map of the bake.
    """
    slots = slot_of_tid(doc, NUM_SLOTS)
    frames = _boxes_per_frame(doc)
    checked = off = 0
    for i, t in enumerate(clip["frames"]):
        if t >= len(frames):
            continue
        for tid, box in frames[t].items():
            if tid not in slots:
                continue
            want = np.array(letterbox(box, doc["width"], doc["height"]))
            got = np.asarray(clip["gt_boxes"][i, slots[tid]])
            checked += 1
            off += int(np.abs(want - got).max() > 1)
    return checked, off


# ------------------------------------------------------------ the probe

def relation_counts(latents, bindings, P, num_slots=NUM_SLOTS):
    """`F[r, a, b, p]`: units of row r that bind slots (a, b) with bit (u, p) on.

    The model relation set of the thesis model, section A.4, as counts. A
    bit is on when it rounds to 1, as in `export_latents.relations_of_state`.
    Returns an array of shape (rows, num_slots, num_slots, P).
    """
    z = np.round(np.asarray(latents, dtype=np.float64))
    z = z.reshape(len(z), -1, P) > 0
    b = np.asarray(bindings)
    out = np.zeros((len(z), num_slots, num_slots, P))
    for sa in range(num_slots):
        for sb in range(num_slots):
            unit = (b[:, :, 0] == sa) & (b[:, :, 1] == sb)
            out[:, sa, sb, :] = (z & unit[:, :, None]).sum(axis=1)
    return out


def probe_rows(doc, clip, P, vocabulary=None):
    """The rows of the readouts for one clip.

    One row for each frame t of the export and each ordered pair of two
    different slots whose tracks both have a box at t. Returns
    `(keys, X, Y)`. `keys[i]` is `(t, subject tid, object tid)`. `X[i]`
    holds the P relation counts of the pair. `Y[i, v]` is 1 when the truth
    holds name `vocabulary[v]` on the pair at t. `Y` is None when
    `vocabulary` is None.
    """
    track = track_of_slot(doc)
    frames = _boxes_per_frame(doc)
    counts = relation_counts(clip["latents"], clip["bindings"], P)
    keys, rows = [], []
    for i, t in enumerate(clip["frames"]):
        t = int(t)
        if t >= len(frames):
            continue
        for sa in sorted(track):
            for sb in sorted(track):
                if sa == sb:
                    continue
                if track[sa] not in frames[t] or track[sb] not in frames[t]:
                    continue
                keys.append((t, track[sa], track[sb]))
                rows.append(counts[i, sa, sb])
    X = np.array(rows, dtype=np.float64).reshape(len(keys), P)
    if vocabulary is None:
        return keys, X, None
    column = dict((name, v) for v, name in enumerate(vocabulary))
    where = dict((key, i) for i, key in enumerate(keys))
    Y = np.zeros((len(keys), len(vocabulary)))
    for rel in doc.get("relation_instances", []):
        v = column.get(rel["predicate"])
        if v is None:
            continue
        for t in range(int(rel["begin_fid"]), int(rel["end_fid"])):
            i = where.get((t, rel["subject_tid"], rel["object_tid"]))
            if i is not None:
                Y[i, v] = 1.0
    return keys, X, Y


def ridge_readout(X_train, Y_train, X_eval, alpha=RIDGE_ALPHA):
    """`(scores, thresholds)` of the ridge readout.

    `predicate_probe.ridge_probe_multi` fits on the train rows and scores
    the train rows and the eval rows. The threshold of name v is the k-th
    largest train score of v, where k is the count of train rows that hold
    v. So on the train clips the readout switches name v on at least as
    often as the truth holds it. A name with no train row gets an infinite
    threshold, so it is never on.

    The scores are rounded to 9 decimals. Without the rounding, the BLAS
    thread count moves the last bits, and a frame whose score equals a
    threshold can flip. Measured on KIT on 2026-10-25: 4 instances changed
    between 8 threads and 1 thread.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_eval = np.asarray(X_eval, dtype=np.float64).reshape(-1, X_train.shape[1])
    both = np.round(ridge_probe_multi(
        X_train, Y_train, np.vstack([X_train, X_eval]), alpha=alpha), 9)
    train, scores = both[:len(X_train)], both[len(X_train):]
    thresholds = np.full(Y_train.shape[1], np.inf)
    for v in range(Y_train.shape[1]):
        k = int(np.asarray(Y_train)[:, v].sum())
        if k > 0:
            thresholds[v] = np.sort(train[:, v])[::-1][k - 1]
    return scores, thresholds


def one_to_one(X_train, Y_train):
    """`(names, agreement)`: one name for each FOSAE predicate.

    `names[p]` is the name that holds on the most train rows on which p
    holds. A tie goes to the first name in vocabulary order. `agreement[p]`
    is the share of the rows of p that hold `names[p]`. A predicate that
    never meets a true name gets -1 and agreement 0.
    """
    on = (np.asarray(X_train) > 0).astype(np.float64)
    together = on.T.dot(np.asarray(Y_train, dtype=np.float64))
    names = np.full(on.shape[1], -1, dtype=int)
    agreement = np.zeros(on.shape[1])
    for p in range(on.shape[1]):
        if together[p].max() > 0:
            names[p] = int(np.argmax(together[p]))
            agreement[p] = together[p, names[p]] / on[:, p].sum()
    return names, agreement


def one_to_one_scores(X_eval, names, agreement, n_names):
    """`(scores, thresholds)` of the one-to-one map on the eval rows.

    The score of name v on a row is the largest agreement of a predicate
    that holds on the row and carries name v, and 0 when there is none. A
    name is on when its score is above 0.
    """
    X_eval = np.asarray(X_eval)
    scores = np.zeros((len(X_eval), n_names))
    for p, v in enumerate(names):
        if v < 0:
            continue
        hold = X_eval[:, p] > 0
        scores[hold, v] = np.maximum(scores[hold, v], agreement[p])
    return scores, np.full(n_names, 1e-12)


def frames_to_instances(doc, keys, scores, thresholds, vocabulary,
                        max_per_video=TOP_RETURNS):
    """The named relations of each frame, joined into relation instances.

    Name v is on for row i when `scores[i, v] >= thresholds[v]`. A run is a
    maximal set of consecutive frames t, t+1, ... on which one name is on
    for one track pair. One run gives one instance [t_b, t_e) with the true
    boxes of the two tracks on those frames. Its score is the mean of its
    frame scores. The best `max_per_video` instances stay, high score first.
    """
    category = dict((so["tid"], so["category"])
                    for so in doc.get("subject/objects", []))
    frames = _boxes_per_frame(doc)
    scores = np.asarray(scores)
    on = scores >= np.asarray(thresholds)[None, :]
    cells = defaultdict(list)
    for i, v in zip(*np.nonzero(on)):
        t, s_tid, o_tid = keys[i]
        cells[(s_tid, o_tid, int(v))].append((t, float(scores[i, v])))
    out = []
    for key in sorted(cells, key=lambda k: (str(k[0]), str(k[1]), k[2])):
        s_tid, o_tid, v = key
        points = sorted(cells[key])
        start = 0
        for j in range(1, len(points) + 1):
            if j < len(points) and points[j][0] == points[j - 1][0] + 1:
                continue
            run = points[start:j]
            begin, end = run[0][0], run[-1][0] + 1
            out.append({
                "triplet": (category[s_tid], vocabulary[v], category[o_tid]),
                "duration": (begin, end),
                "sub_traj": [frames[t][s_tid] for t in range(begin, end)],
                "obj_traj": [frames[t][o_tid] for t in range(begin, end)],
                "score": float(np.mean([p[1] for p in run])),
            })
            start = j
    out.sort(key=lambda r: (-r["score"], r["triplet"], r["duration"]))
    return out[:max_per_video]


def probe_predictions(docs, export, split, vocabulary, route,
                      alpha=RIDGE_ALPHA):
    """`(predictions by clip, diagnostics)` of one naming route.

    `route` is `ridge` or `onetoone`. The readout fits on the rows of the
    train clips. It predicts on the test clips and the validation clips.
    """
    P = export["P"]
    X_parts, Y_parts = [], []
    for clip in split["train"]:
        _, X, Y = probe_rows(docs[clip], export["clips"][clip], P, vocabulary)
        X_parts.append(X)
        Y_parts.append(Y)
    X_train = np.vstack(X_parts)
    Y_train = np.vstack(Y_parts)
    held = split["test"] + split["val"]
    rows = dict((clip, probe_rows(docs[clip], export["clips"][clip], P))
                for clip in held)
    X_eval = np.vstack([rows[c][1] for c in held] + [np.zeros((0, P))])
    diagnostics = {
        "route": route,
        "train_rows": int(len(X_train)),
        "train_positives": dict((name, int(Y_train[:, v].sum()))
                                for v, name in enumerate(vocabulary)),
    }
    if route == "ridge":
        scores, thresholds = ridge_readout(X_train, Y_train, X_eval, alpha)
        diagnostics["alpha"] = alpha
        diagnostics["thresholds"] = dict(
            (name, float(thresholds[v]) if np.isfinite(thresholds[v])
             else None)
            for v, name in enumerate(vocabulary))
    elif route == "onetoone":
        names, agreement = one_to_one(X_train, Y_train)
        scores, thresholds = one_to_one_scores(X_eval, names, agreement,
                                               len(vocabulary))
        diagnostics["map"] = dict(
            (str(p), vocabulary[v] if v >= 0 else None)
            for p, v in enumerate(names))
        diagnostics["agreement"] = dict(
            (str(p), float(a)) for p, a in enumerate(agreement))
    else:
        raise ValueError("unknown route %r" % (route,))
    predictions = {}
    lo = 0
    for clip in held:
        keys = rows[clip][0]
        hi = lo + len(keys)
        predictions[clip] = frames_to_instances(
            docs[clip], keys, scores[lo:hi], thresholds, vocabulary)
        lo = hi
    return predictions, diagnostics


# ------------------------------------------------------ files and the run

def _jsonable(pred):
    """One prediction in the official form, with plain JSON types."""
    return {"triplet": [str(x) for x in pred["triplet"]],
            "score": float(pred["score"]),
            "duration": [int(pred["duration"][0]), int(pred["duration"][1])],
            "sub_traj": [list(box) for box in pred["sub_traj"]],
            "obj_traj": [list(box) for box in pred["obj_traj"]]}


def write_predictions(path, predictions, details):
    """Write one prediction file in the official JSON form and read it back.

    The form is the one of `docs/mm21-gdc/assets/submission-task1.json` in
    the official clone. The return value is the `results` part as read
    from disk, so the evaluator scores the file and not a copy in memory.
    """
    payload = {"version": "VERSION 1.0",
               "results": dict((vid, [_jsonable(p) for p in preds])
                               for vid, preds in sorted(predictions.items())),
               "external_data": {"used": False, "details": details}}
    with open(path, "w") as f:
        json.dump(payload, f)
    with open(path) as f:
        return json.load(f)["results"]


def _rel(path):
    """A path relative to the repository root, for the metrics file."""
    return os.path.relpath(os.path.abspath(path), ROOT)


def run_dataset(spec, out_dir, official, crosscheck=None):
    """Score the four predictors on one dataset and write the files.

    `spec` holds `name`, `export`, `truth` and `split`. `official` and
    `crosscheck` are `(evaluate, record)` pairs from `pinned_evaluator`.
    Writes `predictions_<name>_<predictor>.json` and `metrics_<name>.json`
    into `out_dir`, and returns the metrics.
    """
    name = spec["name"]
    split = read_split(spec["split"])
    export = read_export(spec["export"])
    clips = split["train"] + split["val"] + split["test"]
    if set(export["clips"]) != set(clips):
        raise SystemExit("%s: the export clips and the split clips differ"
                         % name)
    docs = {}
    for clip in clips:
        with open(os.path.join(spec["truth"], clip + ".json")) as f:
            docs[clip] = json.load(f)
    held = split["test"] + split["val"]
    truth = dict((clip, relation_instances(docs[clip])) for clip in held)
    dropped = sum(len(docs[c].get("relation_instances", [])) - len(truth[c])
                  for c in held)

    prior = TripletPrior()
    for clip in split["train"]:
        prior.count(docs[clip])
    vocabulary = sorted(prior.overall)
    if not vocabulary:
        raise SystemExit("%s: the train clips hold no relation" % name)

    predictions = {
        "perfect": dict((c, perfect_predictions(truth[c])) for c in held),
        "frequency": dict(
            (c, frequency_predictions(docs[c], prior, TOP_K, TOP_RETURNS))
            for c in held),
    }
    probe = {}
    for route in ("ridge", "onetoone"):
        predictions[route], probe[route] = probe_predictions(
            docs, export, split, vocabulary, route)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    files = {}
    for predictor in PREDICTORS:
        path = os.path.join(out_dir,
                            "predictions_%s_%s.json" % (name, predictor))
        files[predictor] = write_predictions(
            path, predictions[predictor],
            "%s predictor on true box tracks, export %s"
            % (predictor, _rel(spec["export"])))

    evaluate = official[0]
    rows = []
    self_check = {}
    for clipset in CLIPSETS:
        members = split["test"] if clipset == "test" else held
        gt = dict((c, truth[c]) for c in members)
        scored = [c for c in members if gt[c]]
        if not scored:
            raise SystemExit("%s %s: no clip holds a true instance"
                             % (name, clipset))
        uncut = score(evaluate, gt, files["perfect"], top_returns=None)
        expected = float(np.mean([min(1.0, float(TOP_RETURNS) / len(gt[c]))
                                  for c in scored]))
        self_check[clipset] = {
            "perfect_uncut_mAP": uncut["mAP"],
            "perfect_cut_expected": round(expected, 4),
            "clips_over_200": sum(1 for c in scored
                                  if len(gt[c]) > TOP_RETURNS),
            "pass": uncut["mAP"] == 1.0,
        }
        for predictor in PREDICTORS:
            pred = dict((c, files[predictor].get(c, [])) for c in members)
            row = {"clips": clipset, "predictor": predictor,
                   "n_clips": len(scored),
                   "n_instances": sum(len(gt[c]) for c in members),
                   "official": score(evaluate, gt, pred,
                                     top_returns=TOP_RETURNS),
                   "clip_ap": clip_aps(evaluate, gt, pred,
                                       top_returns=TOP_RETURNS)}
            if crosscheck is not None:
                try:
                    row["crosscheck"] = score(crosscheck[0], gt, pred)
                except (IndexError, ValueError) as exc:
                    row["crosscheck"] = {"error": "%s: %s"
                                         % (type(exc).__name__, exc)}
            rows.append(row)

    checked = off = 0
    for clip in clips:
        n_checked, n_off = slot_box_mismatches(docs[clip],
                                               export["clips"][clip])
        checked += n_checked
        off += n_off
    stub = getattr(sys.modules.get("tqdm"), "__file__", None) is None
    metrics = {
        "dataset": name,
        "export": _rel(spec["export"]),
        "model_name": export["model_name"],
        "truth": _rel(spec["truth"]),
        "split": _rel(spec["split"]),
        "evaluator": official[1],
        "crosscheck": crosscheck[1] if crosscheck is not None else None,
        "tqdm": "stub" if stub else "package",
        "parameters": {"top_returns": TOP_RETURNS, "viou_threshold": 0.5,
                       "top_k": TOP_K, "num_slots": NUM_SLOTS,
                       "ridge_alpha": RIDGE_ALPHA},
        "clips": dict((k, len(v)) for k, v in split.items()),
        "test_clips": split["test"],
        "val_clips": split["val"],
        "vocabulary": vocabulary,
        "truth_dropped": dropped,
        "slot_boxes": {"checked": checked, "off_by_more_than_1px": off},
        "self_check": self_check,
        "probe": probe,
        "rows": rows,
    }
    with open(os.path.join(out_dir, "metrics_%s.json" % name), "w") as f:
        json.dump(metrics, f, indent=1, sort_keys=True)
    return metrics


# ------------------------------------------------------- report tables

def load_metrics(out_dir):
    """`{dataset: metrics}` of every metrics file in `out_dir`, ladder order."""
    found = {}
    for name in LADDER:
        path = os.path.join(out_dir, "metrics_%s.json" % name)
        if os.path.exists(path):
            with open(path) as f:
                found[name] = json.load(f)
    if not found:
        raise SystemExit("no metrics_<dataset>.json in %s" % out_dir)
    return found


def _row(metrics, clipset, predictor):
    return [r for r in metrics["rows"]
            if r["clips"] == clipset and r["predictor"] == predictor][0]


def markdown_tables(out_dir):
    """Tables C.1 to C.3 of the report and the reading rules, applied."""
    metrics = load_metrics(out_dir)
    keys = ("mAP", "R@50", "R@100", "P@1", "P@5", "P@10")
    out = ["**C.1 Self-check.**", "",
           "| dataset | clips | perfect, no cut | perfect, cut 200 "
           "| expected with cut | clips over 200 |",
           "|---|---|---|---|---|---|"]
    for name, m in metrics.items():
        for clipset in CLIPSETS:
            check = m["self_check"][clipset]
            out.append("| %s | %s | %.4f | %.4f | %.4f | %d |" % (
                LABELS[name], clipset, check["perfect_uncut_mAP"],
                _row(m, clipset, "perfect")["official"]["mAP"],
                check["perfect_cut_expected"], check["clips_over_200"]))
    out += ["", "**C.2 Triplet mAP, official.**", "",
            "| dataset | clips | predictor | " + " | ".join(keys)
            + " | scored clips | instances |",
            "|---|---|---|" + "---|" * len(keys) + "---|---|"]
    for clipset in CLIPSETS:
        for name, m in metrics.items():
            for predictor in PREDICTORS:
                r = _row(m, clipset, predictor)
                out.append("| %s | %s | %s | %s | %d | %d |" % (
                    LABELS[name], clipset, predictor,
                    " | ".join("%.4f" % r["official"][k] for k in keys),
                    r["n_clips"], r["n_instances"]))
    out += ["", "**C.3 Cross-check.** xdshang 1b4de17 minus NExTplusplus "
            "b556513, on the same files. The code of the two differs in two "
            "points only: xdshang has no cut at 200, and it does not round "
            "to 4 decimals (A1:26-36). A row with more than 200 predictions "
            "in a clip can differ for the first reason.", "",
            "| dataset | clips | predictor | " + " | ".join(keys) + " |",
            "|---|---|---|" + "---|" * len(keys)]
    for clipset in CLIPSETS:
        for name, m in metrics.items():
            for predictor in PREDICTORS:
                r = _row(m, clipset, predictor)
                cross = r.get("crosscheck") or {"error": "not run"}
                if "error" in cross:
                    cells = " | ".join([cross["error"]] + [""] * 5)
                else:
                    cells = " | ".join(
                        "%+.4f" % (cross[k] - r["official"][k]) for k in keys)
                out.append("| %s | %s | %s | %s |" % (
                    LABELS[name], clipset, predictor, cells))
    out += ["", "**Reading, by the rules fixed before the run.** Test "
            "clips, official numbers.", ""]
    for name, m in metrics.items():
        label = LABELS[name]
        check = m["self_check"]["test"]
        if not check["pass"]:
            out.append("- %s: perfect with no cut gives %.4f. The harness is "
                       "broken, and no other %s number counts."
                       % (label, check["perfect_uncut_mAP"], label))
            continue
        floor = _row(m, "test", "frequency")["official"]["mAP"]
        for route in ("ridge", "onetoone"):
            got = _row(m, "test", route)["official"]["mAP"]
            if got <= floor:
                out.append("- %s, %s: mAP %.4f, at or below frequency %.4f. "
                           "The latent adds nothing that this metric can "
                           "see." % (label, route, got, floor))
            else:
                out.append("- %s, %s: mAP %.4f, above frequency %.4f. The "
                           "latent carries relation information that this "
                           "metric can see, with true box tracks."
                           % (label, route, got, floor))
    return "\n".join(out)


# ------------------------------------------------------------- figures

COLORS = {"perfect": "#52514e", "frequency": "#2a78d6", "ridge": "#eb6834",
          "onetoone": "#1baf7a"}
MARKERS = {"perfect": "D", "frequency": "s", "ridge": "o", "onetoone": "^"}
NAMES = {"perfect": "perfect, cut 200", "frequency": "frequency, no vision",
         "ridge": "probe, ridge readout", "onetoone": "probe, one-to-one map"}


def _frame(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)


def figures(out_dir, fig_dir):
    """Draw the PNG figures of report part C.5 and return their paths.

    1. mAP of each rung on its test clips, in ladder order, one mark for
       each predictor.
    2. AP of each test clip for frequency and the two probe routes, one
       panel for each rung.
    3. VRD beside the published ladder, on the 0 to 100 scale.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = load_metrics(out_dir)
    names = list(metrics)
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    paths = []

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=150)
    for k, predictor in enumerate(PREDICTORS):
        xs = [i + (k - 1.5) * 0.14 for i in range(len(names))]
        ys = [_row(metrics[n], "test", predictor)["official"]["mAP"]
              for n in names]
        ax.plot(xs, ys, linestyle="none", marker=MARKERS[predictor],
                markersize=8, color=COLORS[predictor], label=NAMES[predictor])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([LABELS[n] for n in names])
    ax.set_ylim(-0.03, 1.05)
    _frame(ax, "triplet mAP, test clips")
    ax.set_title("Official evaluator, top 200, true box tracks", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="center left",
              bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    paths.append(os.path.join(fig_dir, "triplet-map-ladder.png"))
    fig.savefig(paths[-1])
    plt.close(fig)

    shown = ("frequency", "ridge", "onetoone")
    fig, axes = plt.subplots(1, len(names), figsize=(2.3 * len(names), 3.4),
                             dpi=150, sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        for k, predictor in enumerate(shown):
            aps = sorted(_row(metrics[name], "test",
                              predictor)["clip_ap"].values())
            ax.plot([k] * len(aps), aps, linestyle="none",
                    marker=MARKERS[predictor], markersize=8,
                    color=COLORS[predictor], alpha=0.75)
        ax.set_xticks(range(len(shown)))
        ax.set_xticklabels(["freq", "ridge", "1:1"])
        ax.set_xlim(-0.6, len(shown) - 0.4)
        ax.set_ylim(-0.03, 1.05)
        n_clips = _row(metrics[name], "test", "ridge")["n_clips"]
        ax.set_title("%s, %d clips" % (LABELS[name], n_clips), fontsize=9)
        _frame(ax, "AP of one test clip" if ax is axes[0][0] else "")
    fig.tight_layout()
    paths.append(os.path.join(fig_dir, "triplet-clip-ap.png"))
    fig.savefig(paths[-1])
    plt.close(fig)

    if "vidvrd" in metrics:
        fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
        for label, value in PUBLISHED:
            ax.axhline(value, color="#9a9893", linewidth=1.0,
                       linestyle="--")
            ax.text(len(PREDICTORS) - 0.45, value + 1.0,
                    "%s %.2f" % (label, value), fontsize=7,
                    color="#52514e", ha="right")
        for k, predictor in enumerate(PREDICTORS):
            y = 100.0 * _row(metrics["vidvrd"], "test",
                             predictor)["official"]["mAP"]
            ax.plot([k], [y], linestyle="none", marker=MARKERS[predictor],
                    markersize=9, color=COLORS[predictor])
            ax.text(k + 0.12, y, "%.2f" % y, fontsize=7, color="#0b0b0b",
                    va="center")
        ax.set_xticks(range(len(PREDICTORS)))
        ax.set_xticklabels(["perfect", "frequency", "ridge", "1:1"])
        ax.set_xlim(-0.5, len(PREDICTORS) - 0.4)
        ax.set_ylim(0, 105)
        _frame(ax, "mAP, 0 to 100")
        ax.set_title("VRD: 7 test clips from the VidVRD train split.\n"
                     "The published rows use the 200 official test clips.",
                     fontsize=8)
        fig.tight_layout()
        paths.append(os.path.join(fig_dir, "triplet-vidvrd-ladder.png"))
        fig.savefig(paths[-1])
        plt.close(fig)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=LADDER,
                    help="the rung to score")
    ap.add_argument("--out", required=True,
                    help="the directory of the prediction and metrics files")
    ap.add_argument("--tables", action="store_true",
                    help="print tables C.1 to C.3 from the metrics in --out")
    ap.add_argument("--figures", metavar="DIR",
                    help="draw the PNG figures from the metrics in --out")
    args = ap.parse_args(argv)
    if args.figures:
        for path in figures(args.out, args.figures):
            print("wrote %s" % path)
        return 0
    if args.tables:
        print(markdown_tables(args.out))
        return 0
    if not args.dataset:
        ap.error("--dataset is necessary to score a rung")
    official = pinned_evaluator(OFFICIAL)
    crosscheck = pinned_evaluator(CROSSCHECK)
    metrics = run_dataset(dataset_spec(args.dataset), args.out, official,
                          crosscheck)
    for row in metrics["rows"]:
        o = row["official"]
        print("%-4s %-8s %-10s mAP %.4f  R@50 %.4f  R@100 %.4f  P@1 %.4f  "
              "P@5 %.4f  P@10 %.4f  clips %d  instances %d" % (
                  LABELS[args.dataset], row["clips"], row["predictor"],
                  o["mAP"], o["R@50"], o["R@100"], o["P@1"], o["P@5"],
                  o["P@10"], row["n_clips"], row["n_instances"]))
    print("slot boxes off by more than 1 px: %d of %d"
          % (metrics["slot_boxes"]["off_by_more_than_1px"],
             metrics["slot_boxes"]["checked"]))
    if not all(c["pass"] for c in metrics["self_check"].values()):
        print("SELF-CHECK FAILED: perfect with no cut is not 1.0. The "
              "harness is broken. Report nothing else.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
