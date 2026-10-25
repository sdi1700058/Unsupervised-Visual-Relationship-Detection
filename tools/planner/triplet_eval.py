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
