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

    .venv-local/bin/python tools/planner/triplet_eval.py --dataset se \\
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from tools.planner.predicate_probe import ridge_probe_multi   # noqa: E402
from tools.planner.relation_oracle import slot_of_tid          # noqa: E402

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
# `tools/planner/m7_map.py`, lines 247-334 and 478-557, byte for byte. That
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
    `latplan/puzzles/puzzle_labeled_objects.py`: one scale for both axes,
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
