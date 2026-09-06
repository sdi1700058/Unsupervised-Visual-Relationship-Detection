#!/usr/bin/env python3
"""The analytic floor of the reconstruction loss, and whether a run sits on it.

**What this answers.** The overnight sweep produced sixteen runs between 0.488
and 0.597 whatever `ZEROSUPPRESS`, `MAX_TEMPERATURE` or `U/A/P` were set to.
When no knob moves the number, the varied knobs are not the cause. They were
not: `val_loss = 0.5245` is `H(0.2182)`, the entropy of the mean of the data,
which is what a decoder emitting **one constant** costs.

The loss is BCE averaged over the whole flattened feature vector
(`latplan/util/distances.py`), so its floor is the density of the data and
carries no dependence on `U`, `A` or `P`. Different architectures converging to
the same number is then not a coincidence, and not a plateau either. It is the
analytic solution of a model that has stopped using its latent.

Measured on CPU over 12 VidVRD videos at 30fps, `mo=5 patch=32` gives a
scalar-mean floor of 0.5235 against `H(0.2182) = 0.5246`.

**How to use it.** Point it at a baked dataset and, if you have one, a run
directory::

    python3 tools/planner/collapse_floor.py data/npz/video/vidvrd/overfit/x.npz
    python3 tools/planner/collapse_floor.py x.npz --run out/video/vidvrd/...

It reads only `images` and `bboxes`, so it needs numpy and nothing else. No
TensorFlow, no GPU, no training. Checking a configuration costs a second and
tells you whether the run that would follow it can say anything at all.

**The related check.** `tools/planner/liveness.py` asks the same question from
the other end: it counts distinct codes in an export and reports one distinct
code as a dead encoder. The two agree where both can be run -- 4 of the 21
exports on disk are dead by that measure -- and they are independent, because
one reads the latent and this one reads the data.

Standard library and numpy only, Python 3.6 clean.
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

# The bbox block is four one-hot vectors: x1, y1, x2, y2 over a 60x40 grid.
# `latplan/puzzles/puzzle_labeled_objects.py` fixes the grid at PICSIZE // 5,
# so the block is 2*60 + 2*40 = 200 dims however large the patch is. That is
# why the patch size decides the balance of the loss.
BBOX_DIMS = 200
BBOX_ONES = 4

# How close to the floor counts as sitting on it. The measured pair was 0.5235
# against 0.5246, a gap of 0.0011, so anything under a hundredth is the same
# answer to the precision the training curve is recorded at.
TOLERANCE = 0.01


def entropy(p):
    """Binary entropy in **nats**, because Keras BCE uses the natural log.

    This is the loss of the best constant predictor: a decoder that ignores
    its input and emits the mean of the data pays exactly `H(mean)`.
    """
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1.0 - p) * math.log1p(-p))


def box_share(patch_size):
    """The fraction of the feature vector that carries position.

    The patch and the box block are concatenated with no weighting and the
    loss averages over the result, so this **is** the share of the gradient
    the position signal receives. It moves by a factor of eight across the
    patch sizes that have been swept:

        patch 8 -> 51.0 percent      patch 32 -> 6.1 percent
        patch 16 -> 20.7 percent     patch 48 -> 2.8 percent

    Which means a patch-size sweep is also a loss-reweighting sweep, and its
    arms cannot be compared to each other as though only the input changed.
    """
    patch_dim = patch_size * patch_size * 3
    return float(BBOX_DIMS) / (patch_dim + BBOX_DIMS)


def feature_mean(images, bboxes):
    """`p`, the mean of the feature vector the model actually reconstructs.

    Computed from the two arrays a bake writes rather than by building the
    concatenation, which would need `strips.bboxes_to_onehot` and therefore
    TensorFlow. The arithmetic is exact: every real object slot contributes
    its patch pixels plus exactly `BBOX_ONES` ones out of `BBOX_DIMS`, and
    every padded slot contributes an all-zero patch and an all-zero box.

    A padded slot is one whose box is all zeros. A real object with a
    degenerate box is indistinguishable from padding here, which is a known
    property of the loader rather than of this function.
    """
    images = np.asarray(images)
    bboxes = np.asarray(bboxes)
    if images.ndim < 3 or bboxes.ndim != 3 or bboxes.shape[-1] != 4:
        raise ValueError(
            "expected images (frames, objects, h, w, c) and bboxes "
            "(frames, objects, 4); got %r and %r"
            % (images.shape, bboxes.shape))

    n_frames, n_objs = bboxes.shape[0], bboxes.shape[1]
    patch_dim = int(np.prod(images.shape[2:]))

    real = int(np.count_nonzero(np.any(bboxes != 0, axis=-1)))
    total_ones = float(images.sum()) + BBOX_ONES * real
    total_dims = float(n_frames * n_objs * (patch_dim + BBOX_DIMS))
    return total_ones / total_dims if total_dims else 0.0


def floor_of(images, bboxes):
    """`(p, H(p))` for one baked dataset."""
    p = feature_mean(images, bboxes)
    return p, entropy(p)


def val_bce_min(path):
    """The lowest `val_BCE` in a `training_history.csv`, or None.

    None rather than 0.0 when there is nothing to read: a zero here would
    read as the best run ever recorded.
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as handle:
            rows = list(csv.reader(handle))
    except OSError:
        return None
    if len(rows) < 2:
        return None
    header = [c.strip() for c in rows[0]]
    if "val_BCE" not in header:
        return None
    idx = header.index("val_BCE")
    values = []
    for row in rows[1:]:
        if idx >= len(row):
            continue
        try:
            v = float(row[idx])
        except ValueError:
            continue
        if v == v:                      # skip nan
            values.append(v)
    return min(values) if values else None


def verdict(val_loss, floor, tolerance=TOLERANCE):
    """Where a run sits relative to the constant-output solution."""
    out = {"val_loss": val_loss, "floor": floor, "tolerance": tolerance,
           "gap": None, "state": None, "reading": ""}
    if val_loss is None or floor is None:
        out["reading"] = ("no validation loss was found, so nothing is "
                          "claimed about this run")
        return out

    gap = float(val_loss) - float(floor)
    out["gap"] = gap
    if abs(gap) <= tolerance:
        out["state"] = "at_the_floor"
        out["reading"] = (
            "at the floor: this is what a decoder emitting one constant "
            "costs, so the latent is carrying nothing")
    elif gap < 0:
        out["state"] = "below_the_floor"
        out["reading"] = (
            "below the floor by %.4f, so the model is using its latent"
            % -gap)
    else:
        out["state"] = "above_the_floor"
        out["reading"] = (
            "above the floor by %.4f, which is worse than predicting the "
            "mean of the data" % gap)
    return out


def read_bake(path):
    """`(images, bboxes)` from a baked npz, whatever it was written by.

    `allow_pickle` is needed for the object-dtype `names` and `meta` fields
    `save_cache` writes, and the files read here are ones this repository
    baked. Do not point it at an npz from anywhere else.
    """
    with np.load(path, allow_pickle=True) as data:
        if "images" not in data.files or "bboxes" not in data.files:
            raise SystemExit(
                "%s holds %s. This reads a baked dataset written by "
                "`latplan.util.cache.save_cache`, which stores `images` and "
                "`bboxes`. An export from `export_latents.py` holds latents "
                "instead; use tools/planner/liveness.py for those."
                % (path, ", ".join(data.files) or "nothing"))
        return data["images"], data["bboxes"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("npz", nargs="+", help="baked dataset(s) to read")
    ap.add_argument("--run", default=None,
                    help="a run directory holding training_history.csv, to "
                         "compare the achieved loss against the floor")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    a = ap.parse_args(argv)

    achieved = None
    if a.run:
        achieved = val_bce_min(os.path.join(a.run, "training_history.csv"))
        if achieved is None:
            sys.stderr.write(
                "no val_BCE in %s/training_history.csv; reporting the floor "
                "only\n" % a.run)

    print("%-44s %8s %8s %s" % ("dataset", "mean", "floor", "reading"))
    print("%-44s %8s %8s %s" % ("-" * 44, "-" * 8, "-" * 8, "-" * 7))
    worst = 0
    for path in a.npz:
        try:
            images, bboxes = read_bake(path)
        except SystemExit as exc:
            print("%-44s %8s %8s %s" % (os.path.basename(path), "-", "-", exc))
            worst = 1
            continue
        p, floor = floor_of(images, bboxes)
        v = verdict(achieved, floor, a.tolerance)
        print("%-44s %8.4f %8.4f %s"
              % (os.path.basename(path), p, floor, v["reading"]))
        if v["state"] == "at_the_floor":
            worst = 1

    if achieved is not None:
        print("\nachieved val_BCE %.4f from %s" % (achieved, a.run))
        print("A run at the floor has not learned. Reverting `zerosuppress` "
              "to upstream's 0.0 and `max_temperature` to 5.0, one at a "
              "time, is the first thing to try.")
    return worst


if __name__ == "__main__":
    sys.exit(main())
