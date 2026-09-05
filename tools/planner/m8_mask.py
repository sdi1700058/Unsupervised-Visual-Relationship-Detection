#!/usr/bin/env python3
"""M8 — score a plan against masks rather than bounding boxes.

How a corpus draws an object boundary is a property of the **corpus**, and the
evaluation has to adapt to it. The reverse rule, where a corpus must fit
whatever evaluation happens to exist already, is why usable datasets were set
aside: PVSG carries 150,000 labelled frames over 400 videos with temporal
scene graphs, and it was passed over only because its objects are masks.

This module removes that reason. It provides both halves, and they are not
alternatives:

1. **A mask metric.** `bbox_mse` becomes a mask distance, and the mask
   representation gets the same round-trip floor the box representation has,
   so a mask number is comparable with a box number.
2. **A mask-to-box conversion.** A tight box from a mask is a few lines, and
   with it a mask corpus is directly comparable with VidVRD, VidOR and Action
   Genome under every metric already written.

Two metrics, because one is not enough
--------------------------------------

`mask_iou` is the region measure. It is bounded in [0, 1], scale invariant,
and it is what the segmentation literature reports, so a reader calibrated to
that literature can read it. **It is blind to where the error sits.** A mask
eroded evenly by a pixel and a mask with one thick slab missing score the same
whenever the two losses have the same area. It also loses resolution on small
objects, where a few pixels swing it a long way.

`boundary_f` is the contour measure, the DAVIS-style F-measure between the two
outlines at a pixel tolerance. It sees exactly what IoU cannot: an outline in
the wrong place is penalised even when the area is right. **It is blind to
errors below its tolerance** — at `tol=2` a mask eroded by one pixel scores a
perfect 1.0 — and it is blind to how far a wrong outline is once it passes the
tolerance, since 3 pixels out and 300 pixels out both score nothing.

Report both. Neither one is the answer on its own, and the pair of them fails
in different directions.

The mask quantisation floor
---------------------------

Every box result in this project is reported against its own floor, the error
that the box-to-latent-to-box round trip costs before a planner exists
(`oracle.round_trip_error`). A mask number without the same treatment cannot
be compared with any of them.

The mask analogue of the box's one-hot coordinate code is a **grid occupancy
code**: split the canvas into `bins_y` by `bins_x` cells and spend one bit per
cell per object. That keeps the property the box code was chosen for, that
Hamming distance means something — moving a mask by one cell flips a bounded
number of bits — and it is the smallest code that can represent a shape at
all.

It costs far more than a box. At the resolution the real decoder uses, 60 by
40, a box is 200 bits per object and a grid mask is 2400. That is a fact about
the representation rather than about any model, and it is the first thing to
know before asking FOSAE to carry masks.

The decode threshold is deliberate. A cell turns on when the mask covers at
least `threshold` of it, and 0.5 is the default because it is unbiased: a
lower value makes every decoded mask contain the true one and reports a floor
that is optimistic about area and pessimistic about outline. The box side
learned this lesson the hard way, where the choice between the bin centre and
the bin's left edge moved the measured floor by a factor of 4.1
(`oracle.dequantise`). The threshold is therefore an argument, and the run
reports the floor at more than one value of it.
"""

import argparse
import json
import os
import sys

import numpy as np

# Running this file as a script puts `tools/planner` on the path, not the
# repository root, so the package imports below would fail. The tests import
# it as `tools.planner.m8_mask`, where the root is already there and this is a
# no-op.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.planner.common.metrics import _assign, bbox_iou   # noqa: E402
from tools.planner.oracle import (CANVAS_H, CANVAS_W,        # noqa: E402
                                  DEFAULT_BINS_X, DEFAULT_BINS_Y)

DEFAULT_TOL = 2


# --------------------------------------------------------------------------
# the conversion: masks to boxes and back
# --------------------------------------------------------------------------

def masks_to_boxes(masks):
    """Tight boxes `(..., 4)` enclosing boolean masks `(..., H, W)`.

    The box is half-open in the same sense the corpora use it, so `x2 - x1` is
    the number of covered columns. An empty mask returns an all-zero box,
    which is how every loader in this project marks "no object here" — not a
    box at the origin.

    This is the whole reason a mask corpus does not need a separate pipeline.
    With it, PVSG or any other mask corpus is scored by `bbox_mse`,
    `bbox_iou`, `oracle.round_trip_error` and the planner exactly as VidVRD is.
    """
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim < 2:
        raise ValueError("masks must be at least (H, W); got %r"
                         % (masks.shape,))
    height, width = masks.shape[-2], masks.shape[-1]
    lead = masks.shape[:-2]
    flat = masks.reshape((-1, height, width))

    rows_any = flat.any(axis=2)                       # (N, H)
    cols_any = flat.any(axis=1)                       # (N, W)
    occupied = rows_any.any(axis=1)                   # (N,)

    r = np.arange(height)
    c = np.arange(width)
    y1 = np.where(rows_any, r[None, :], height).min(axis=1)
    y2 = np.where(rows_any, r[None, :], -1).max(axis=1) + 1
    x1 = np.where(cols_any, c[None, :], width).min(axis=1)
    x2 = np.where(cols_any, c[None, :], -1).max(axis=1) + 1

    out = np.zeros((flat.shape[0], 4), dtype=np.float64)
    out[occupied, 0] = x1[occupied]
    out[occupied, 1] = y1[occupied]
    out[occupied, 2] = x2[occupied]
    out[occupied, 3] = y2[occupied]
    return out.reshape(lead + (4,))


def boxes_to_masks(boxes, height=CANVAS_H, width=CANVAS_W):
    """Rasterise boxes `(..., 4)` into rectangular masks `(..., H, W)`.

    The degenerate direction, and it exists on purpose. A box is a mask whose
    shape happens to be a rectangle, so this runs the mask code path over the
    corpora already on disk and lets the mask metric be checked against the
    box metric it has to agree with.

    A pixel is covered when its **centre** falls inside the half-open box, so
    an integer box round-trips through `masks_to_boxes` exactly. A box from a
    real corpus lands between pixels after the canvas rescale, and there the
    two representations differ by the area the rasteriser rounds;
    `agreement_with_boxes` measures that difference rather than assuming it
    away.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.shape[-1] != 4:
        raise ValueError("boxes must be (..., 4); got %r" % (boxes.shape,))
    lead = boxes.shape[:-1]
    flat = boxes.reshape((-1, 4))

    cx = np.arange(width, dtype=np.float64) + 0.5
    cy = np.arange(height, dtype=np.float64) + 0.5
    cols = (cx[None, :] >= flat[:, 0:1]) & (cx[None, :] < flat[:, 2:3])
    rows = (cy[None, :] >= flat[:, 1:2]) & (cy[None, :] < flat[:, 3:4])
    masks = rows[:, :, None] & cols[:, None, :]
    return masks.reshape(lead + (height, width))


# --------------------------------------------------------------------------
# the two metrics
# --------------------------------------------------------------------------

def mask_iou(pred_mask, gt_mask):
    """Intersection over union of two boolean masks, or None.

    None when the union is empty. Two empty masks are two absent objects and
    there is nothing to score, which is not the same as a perfect overlap —
    `bbox_iou` refuses the same case for the same reason, and returning 1.0
    would put a window with no data into the reported median as its best row.
    """
    a = np.asarray(pred_mask, dtype=bool)
    b = np.asarray(gt_mask, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("shape mismatch: pred %r vs gt %r"
                         % (a.shape, b.shape))
    union = int((a | b).sum())
    if union == 0:
        return None
    return float((a & b).sum()) / union


def dice_from_iou(iou):
    """Dice from Jaccard: `2J / (1 + J)`.

    Written as a transform rather than as a second formula, because that is
    the honest shape of the relationship and it makes the dependence
    impossible to miss at the call site.
    """
    if iou is None:
        return None
    return 2.0 * float(iou) / (1.0 + float(iou))


def mask_dice(pred_mask, gt_mask):
    """Dice coefficient — the SAME measure as `mask_iou`, under another name.

    Dice is `2|A and B| / (|A| + |B|)`, and it equals `2J / (1 + J)` where J
    is the Jaccard index that `mask_iou` returns. That transform is strictly
    increasing, so the two agree on the ordering of any set of predictions and
    **neither can ever contradict the other**.

    It is here because the segmentation literature reports Dice and a reader
    calibrated to that literature should not have to convert. It is not a
    second evaluation method. Anything counting distinct methods must count
    this pair once, exactly as `mask_iou` and `boundary_f` must be counted
    twice — those two do disagree, and the pair with equal IoU and different
    contour scores in the tests is the demonstration.

    None when both masks are empty, matching `mask_iou`.
    """
    return dice_from_iou(mask_iou(pred_mask, gt_mask))


def _shift(mask, dy, dx):
    """`mask` moved by (dy, dx), with False shifted in from outside."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys, ye = max(0, dy), h + min(0, dy)
    xs, xe = max(0, dx), w + min(0, dx)
    if ys >= ye or xs >= xe:
        return out
    out[ys:ye, xs:xe] = mask[ys - dy:ye - dy, xs - dx:xe - dx]
    return out


def _erode(mask):
    """Drop every pixel with a 4-neighbour outside the mask.

    Outside the image counts as outside the mask, so an object clipped by the
    frame edge has a contour there. Both sides of every comparison use the
    same rule, so the choice does not favour either.
    """
    out = mask.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        out &= _shift(mask, dy, dx)
    return out


def mask_boundary(mask):
    """The one-pixel contour of a mask: the mask minus its erosion."""
    mask = np.asarray(mask, dtype=bool)
    return mask & ~_erode(mask)


def _dilate(mask, radius):
    """`mask` grown by a disk of the given radius, by shifts alone.

    A disk of radius 2 is 13 shifts, so this stays cheap and needs nothing
    beyond numpy. scipy would do it faster and is not assumed present.
    """
    radius = int(radius)
    if radius <= 0:
        return mask.copy()
    out = np.zeros_like(mask)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                out |= _shift(mask, dy, dx)
    return out


def boundary_f(pred_mask, gt_mask, tol=DEFAULT_TOL):
    """Contour F-measure between two masks at a pixel tolerance, or None.

    Precision is the share of the predicted contour lying within `tol` of the
    real contour, recall is the share of the real contour lying within `tol`
    of the predicted one, and the score is their harmonic mean. This is the
    DAVIS convention, and `tol=2` is about 0.75% of the 300x200 canvas
    diagonal, which is the tolerance that convention uses.

    Read it next to `mask_iou`, never instead of it. It is blind to error
    below `tol`, blind to how far past `tol` a wrong contour has gone, and
    blind to which side of a contour is filled.

    None when neither mask has a contour, which means both are empty and there
    is nothing to score. 0.0 when exactly one of them is empty, which is a
    real and total miss.
    """
    pb = mask_boundary(pred_mask)
    gb = mask_boundary(gt_mask)
    n_p, n_g = int(pb.sum()), int(gb.sum())
    if n_p == 0 and n_g == 0:
        return None
    if n_p == 0 or n_g == 0:
        return 0.0
    precision = float((pb & _dilate(gb, tol)).sum()) / n_p
    recall = float((gb & _dilate(pb, tol)).sum()) / n_g
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------
# the mask code, and the floor it costs
# --------------------------------------------------------------------------

def mask_bits_per_object(bins_x, bins_y):
    """Latent bits one object occupies under the grid occupancy code.

    One bit per cell. Compare `oracle.bits_per_object`, which is
    `2*bins_x + 2*bins_y` for the same grid: at 60 by 40 a box costs 200 bits
    and a mask costs 2400, a factor of 12.
    """
    return int(bins_x) * int(bins_y)


def _cell_starts(n, bins):
    """First pixel index of each grid cell along one axis.

    Cells are as equal as integer division allows, so the grid does not have
    to divide the canvas. At 200 pixels over 40 cells every cell is 5 pixels;
    at 20 pixels over 6 cells the widths are 3, 3, 4, 3, 3, 4.
    """
    n, bins = int(n), int(bins)
    if bins < 1:
        raise ValueError("bins must be at least 1; got %d" % bins)
    if bins > n:
        raise ValueError(
            "%d cells over %d pixels leaves empty cells, and an empty cell "
            "can never turn on" % (bins, n))
    return (np.arange(bins) * n) // bins


def _cell_sizes(n, bins):
    starts = _cell_starts(n, bins)
    return np.diff(np.append(starts, int(n)))


def masks_to_latents(masks, bins_x=DEFAULT_BINS_X, bins_y=DEFAULT_BINS_Y,
                     threshold=0.5):
    """Encode masks `(N, n_objs, H, W)` as `(N, n_objs*bins_x*bins_y)` bits.

    A cell turns on when the mask covers at least `threshold` of its area.
    The bits of one object run row-major over the grid, so neighbouring cells
    along x are neighbouring bits, which keeps a sideways move a small number
    of bit flips.

    An absent object — an all-zero mask, the padding convention every loader
    here uses — encodes to an all-zero block rather than to the top-left cell,
    exactly as `oracle.boxes_to_latents` does. The cost of that convention is
    that an object too small to fill half a cell also encodes to all zeros and
    is read back as absent. `round_trip_masks` counts those rather than
    letting them disappear.
    """
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 4:
        raise ValueError("masks must be (N, n_objs, H, W); got %r"
                         % (masks.shape,))
    n_states, n_objs, height, width = masks.shape

    row_starts, col_starts = _cell_starts(height, bins_y), _cell_starts(width,
                                                                        bins_x)
    row_sizes, col_sizes = _cell_sizes(height, bins_y), _cell_sizes(width,
                                                                    bins_x)

    counts = np.add.reduceat(masks.astype(np.int64), row_starts, axis=2)
    counts = np.add.reduceat(counts, col_starts, axis=3)
    cell_area = row_sizes[:, None] * col_sizes[None, :]
    on = counts >= (np.asarray(threshold, dtype=np.float64) * cell_area)
    # A cell with no covered pixel is off whatever the threshold, so a
    # threshold of 0 does not paint the whole canvas.
    on &= counts > 0

    return on.reshape((n_states, n_objs * bins_y * bins_x)).astype(np.int8)


def latents_to_masks(latents, n_objs, bins_x=DEFAULT_BINS_X,
                     bins_y=DEFAULT_BINS_Y, height=CANVAS_H, width=CANVAS_W):
    """Decode grid occupancy bits back to masks `(N, n_objs, H, W)`.

    A cell that is on paints all of its pixels. An all-zero block comes back
    as an empty mask, so padding round-trips.
    """
    latents = np.asarray(latents)
    if latents.ndim == 1:
        latents = latents[None, :]
    per_obj = mask_bits_per_object(bins_x, bins_y)
    expected = n_objs * per_obj
    if latents.shape[1] != expected:
        raise ValueError(
            "latent width %d does not match %d objects at %d bits each (%d)"
            % (latents.shape[1], n_objs, per_obj, expected))

    grid = latents.reshape((latents.shape[0], n_objs, bins_y, bins_x))
    grid = grid.astype(bool)
    out = np.repeat(grid, _cell_sizes(height, bins_y), axis=2)
    out = np.repeat(out, _cell_sizes(width, bins_x), axis=3)
    return out


def round_trip_masks(masks, bins_x=DEFAULT_BINS_X, bins_y=DEFAULT_BINS_Y,
                     threshold=0.5, tol=DEFAULT_TOL):
    """The mask quantisation floor: what the code costs before any planner.

    This is the mask counterpart of `oracle.round_trip_error`, and it is the
    number that makes a mask result comparable with a box result. Every box
    figure in this project is reported against its own floor, so a mask figure
    without one cannot be placed beside them.

    It differs from the box floor in its units, and that is unavoidable: a
    box floor is squared pixels of corner displacement and a mask floor is a
    region overlap. What carries across is the **ratio** — how close a planner
    gets to the best its representation permits — which is what
    `metrics.floor_ratio` reports on the box side.

    Averaged over **present** object-frames only, matching the denominator
    `bbox_mse` and `oracle.round_trip_error` use. An object that vanishes
    entirely stays in that average with a score of 0 and is also counted
    separately, because a deleted object is a different failure from a
    blurred one.
    """
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 4:
        raise ValueError("masks must be (N, n_objs, H, W); got %r"
                         % (masks.shape,))
    n_states, n_objs, height, width = masks.shape

    z = masks_to_latents(masks, bins_x, bins_y, threshold)
    back = latents_to_masks(z, n_objs, bins_x, bins_y, height, width)

    present = masks.any(axis=(2, 3))
    vanished = int((present & ~back.any(axis=(2, 3))).sum())

    ious, bfs = [], []
    for t in range(n_states):
        for o in range(n_objs):
            if not present[t, o]:
                continue
            ious.append(mask_iou(back[t, o], masks[t, o]))
            f = boundary_f(back[t, o], masks[t, o], tol=tol)
            bfs.append(0.0 if f is None else f)

    n_present = len(ious)
    return {
        "mean_iou": float(np.mean(ious)) if n_present else None,
        # Derived per pair and then averaged, never recomputed, so the two
        # columns cannot drift into looking like independent evidence.
        "mean_dice": (float(np.mean([dice_from_iou(v) for v in ious]))
                      if n_present else None),
        "mean_boundary_f": float(np.mean(bfs)) if n_present else None,
        "n_present": n_present,
        "vanished": vanished,
        "bits_per_object": mask_bits_per_object(bins_x, bins_y),
        "bins": [int(bins_x), int(bins_y)],
        "threshold": float(threshold),
        "tol": int(tol),
    }


# --------------------------------------------------------------------------
# scoring a window
# --------------------------------------------------------------------------

def match_mask_slots(pred_masks, gt_masks, gt_present=None):
    """Pair decoded object slots with annotated objects, by mask overlap.

    The mask counterpart of `metrics.match_slots`, and it follows the same two
    rules that measurement forced on that function: the pairing is solved once
    over the whole window rather than per frame, and the cost is accumulated
    only over frames where the ground-truth object is actually there. An
    all-zero mask is an absent object, not an object of zero size, and costing
    against it made absence a strong attractor on the box side.

    The cost is `1 - IoU` rather than a box distance, so an object with an odd
    shape is paired on the evidence the mask metric will be scored with.
    `_assign` is shared with the box pipeline so both have one solver.
    """
    pred = np.asarray(pred_masks, dtype=bool)
    gt = np.asarray(gt_masks, dtype=bool)
    if pred.ndim == 3:
        pred = pred[None]
    if gt.ndim == 3:
        gt = gt[None]

    n_pred, n_gt = pred.shape[1], gt.shape[1]
    n = max(n_pred, n_gt)
    present = gt.any(axis=(2, 3))                     # (T, n_gt)
    if gt_present is not None:
        present = present & np.asarray(gt_present, dtype=bool)[None, :]

    cost = np.full((n, n), 1e6)
    for j in range(n_gt):
        rows = np.nonzero(present[:, j])[0]
        if rows.size == 0:
            continue
        for i in range(n_pred):
            total = 0.0
            for t in rows:
                v = mask_iou(pred[t, i], gt[t, j])
                total += 1.0 - (0.0 if v is None else v)
            cost[i, j] = total / rows.size

    rows, cols = _assign(cost)
    mapping = np.full(n_pred, -1, dtype=np.int64)
    for r, c in zip(rows, cols):
        if r < n_pred and c < n_gt and cost[r, c] < 1e6:
            mapping[r] = c
    return mapping


def mask_scores(pred_masks, gt_masks, mapping=None, matching="hungarian",
                scoreable=None, tol=DEFAULT_TOL):
    """Mask IoU and contour F over a window, the way `bbox_mse` scores boxes.

    `pred_masks` and `gt_masks` are both `(T, n_objs, H, W)` and cover the same
    T frames.

    Absent ground-truth objects are excluded from both numbers rather than
    scored as misses, and the exclusion is shared, which is the correction
    `bbox_iou` needed on 2026-08-30: an object that leaves the scene once
    scored zero error and 0.625 IoU at the same time, because the two metrics
    were counting different frames.

    `mean_iou` is None, never 0.0, when nothing was scoreable. Zero is the
    worst score and would be read as a result; None says there was no data.
    """
    pred = np.asarray(pred_masks, dtype=bool)
    gt = np.asarray(gt_masks, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError("shape mismatch: pred %r vs gt %r"
                         % (pred.shape, gt.shape))
    n_frames, n_objs = pred.shape[0], pred.shape[1]

    if mapping is None:
        if matching == "hungarian":
            mapping = match_mask_slots(pred, gt, gt_present=scoreable)
        elif matching == "fixed":
            mapping = np.arange(n_objs, dtype=np.int64)
        else:
            raise ValueError("unknown matching mode: %s" % matching)
    mapping = np.asarray(mapping)

    ious = np.zeros((n_frames, n_objs))
    bfs = np.zeros((n_frames, n_objs))
    scored = np.zeros((n_frames, n_objs), dtype=bool)

    for i in range(n_objs):
        j = int(mapping[i])
        if j < 0:
            continue
        present = gt[:, j].any(axis=(1, 2))
        if scoreable is not None:
            present = present & bool(scoreable[j])
        for t in range(n_frames):
            if not present[t]:
                continue
            v = mask_iou(pred[t, i], gt[t, j])
            f = boundary_f(pred[t, i], gt[t, j], tol=tol)
            ious[t, i] = 0.0 if v is None else v
            bfs[t, i] = 0.0 if f is None else f
            scored[t, i] = True

    per_frame_count = scored.sum(axis=1)
    per_frame_iou = np.divide(ious.sum(axis=1), per_frame_count,
                              out=np.zeros(n_frames),
                              where=per_frame_count > 0)
    n_scored = int(scored.sum())

    return {
        "mean_iou": float(ious[scored].mean()) if n_scored else None,
        # See `dice_from_iou`: one measure, two names. Reported because the
        # segmentation literature reports it, derived so that nothing can
        # mistake it for a second opinion.
        "mean_dice": (float(np.mean([dice_from_iou(v) for v in ious[scored]]))
                      if n_scored else None),
        "mean_boundary_f": float(bfs[scored].mean()) if n_scored else None,
        "per_frame_iou": [float(v) for v in per_frame_iou],
        "frames_above_0.5": int((per_frame_iou[per_frame_count > 0]
                                 >= 0.5).sum()),
        "mapping": [int(v) for v in mapping],
        "matching_mode": matching,
        "skipped_absent": int(scored.size - n_scored),
        "n_scored": n_scored,
    }


def score_masks(pred_masks, gt_masks, baseline_masks=None,
                matching="hungarian", scoreable=None, tol=DEFAULT_TOL,
                floor=None):
    """Score one interpolation window against mask ground truth.

    The mask counterpart of `metrics.score_window`. `baseline_masks` is the
    straight-line prediction, and a planner that does not beat it has not
    learned the dynamics — the same bar the box pipeline sets.

    `beats_baseline` compares IoU, where **higher is better**, so it is a
    difference and not the `mse_ratio` division used on the box side. Dividing
    two overlaps would report the smaller one whenever a window is easy.

    `floor` is a `round_trip_masks` result. When given, `iou_vs_floor` reports
    the planner's overlap as a share of the best this representation permits,
    which is the mask counterpart of `metrics.floor_ratio` and approaches 1
    when the planner has taken everything the code allows.
    """
    result = {"planner": mask_scores(pred_masks, gt_masks, matching=matching,
                                     scoreable=scoreable, tol=tol)}

    if baseline_masks is not None:
        result["baseline"] = mask_scores(baseline_masks, gt_masks,
                                         matching=matching,
                                         scoreable=scoreable, tol=tol)
        p, b = result["planner"]["mean_iou"], result["baseline"]["mean_iou"]
        result["iou_gain"] = None if (p is None or b is None) else float(p - b)
        result["beats_baseline"] = bool(result["iou_gain"] is not None
                                        and result["iou_gain"] > 0.0)

    if floor is not None and floor.get("mean_iou"):
        p = result["planner"]["mean_iou"]
        result["iou_vs_floor"] = (None if p is None
                                  else float(p) / float(floor["mean_iou"]))
    return result


# --------------------------------------------------------------------------
# the agreement check
# --------------------------------------------------------------------------

def agreement_with_boxes(pred_boxes, gt_boxes, height=CANVAS_H,
                         width=CANVAS_W):
    """Do the mask metric and the box metric agree on rectangles?

    The correctness test that can be run today, with no mask corpus on disk. A
    box is a mask whose shape is a rectangle, so rasterising both sides and
    scoring them with `mask_iou` must reproduce what `bbox_iou` returns. If it
    does not, the new metric is measuring something else and no mask result
    from it would mean anything.

    `bbox_iou` is **called**, not reimplemented, so the comparison is against
    the real box metric rather than against a second copy of it that could
    drift.

    Exact agreement is expected only for integer boxes. A corpus box lands
    between pixels after the canvas rescale, and a mask cannot hold half a
    pixel, so `max_abs_deviation` is the rasterisation error and it is
    reported rather than assumed small.
    """
    pred = np.asarray(pred_boxes, dtype=np.float64)
    gt = np.asarray(gt_boxes, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError("shape mismatch: pred %r vs gt %r"
                         % (pred.shape, gt.shape))

    pm = boxes_to_masks(pred, height, width)
    gm = boxes_to_masks(gt, height, width)

    box_ious, mask_ious = [], []
    for t in range(gt.shape[0]):
        for o in range(gt.shape[1]):
            if float(np.abs(gt[t, o]).sum()) <= 0:
                continue
            b = bbox_iou(pred[t:t + 1, o:o + 1], gt[t:t + 1, o:o + 1],
                         matching="fixed")["mean_iou"]
            m = mask_iou(pm[t, o], gm[t, o])
            if b is None or m is None:
                continue
            box_ious.append(float(b))
            mask_ious.append(float(m))

    if not box_ious:
        return {"max_abs_deviation": None, "mean_abs_deviation": None,
                "n_pairs": 0, "box_ious": [], "mask_ious": []}

    d = np.abs(np.asarray(box_ious) - np.asarray(mask_ious))
    return {
        "max_abs_deviation": float(d.max()),
        "mean_abs_deviation": float(d.mean()),
        "n_pairs": len(box_ious),
        "box_ious": box_ious,
        "mask_ious": mask_ious,
    }


# --------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------

def _esc(text):
    """XML-escape a label.

    A raw angle bracket in SVG text is invalid XML and the file will not open.
    This project has shipped an unopenable figure that way three times.
    """
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _wrap(text, limit=128):
    """Break a caption into lines that fit the figure.

    SVG text does not wrap. A long caption runs off the right edge and the
    reader never sees the end of it, which is how a caveat gets lost.
    """
    words, lines, current = str(text).split(), [], ""
    for w in words:
        if current and len(current) + 1 + len(w) > limit:
            lines.append(current)
            current = w
        else:
            current = w if not current else current + " " + w
    if current:
        lines.append(current)
    return lines


def _caption(out, x, y, text, size=11, fill="#444", line_h=15, limit=128):
    """Emit a wrapped, escaped caption and return the y below it."""
    for line in _wrap(text, limit):
        out.append('<text x="%d" y="%d" font-size="%d" fill="%s">%s</text>'
                   % (x, y, size, fill, _esc(line)))
        y += line_h
    return y


def write_figure(report, path, width=820):
    """Three panels: the floors, the blind spot, and the agreement."""
    floors = report.get("floors", [])
    blind = report.get("blindness", {})
    agree = report.get("agreement", {})

    dev = agree.get("max_abs_deviation")
    if dev is None:
        headline = "no pair could be compared"
    else:
        headline = ("largest gap between mask IoU and bbox IoU over %d "
                    "object-frames: %.2e  (source: %s)"
                    % (agree.get("n_pairs", 0), dev,
                       agree.get("source", "unknown")))
    footer = ("No mask corpus is on disk. Every number here comes from "
              "synthetic masks or from boxes rasterised as rectangles.")
    tail = [l for l in (headline, report.get("agreement_note"), footer) if l]

    row_h = 24
    top = 68 + len(_wrap(report.get("subtitle", ""))) * 15
    # Sized from the text that will actually be written, every wrapped line of
    # it. Budgeting for one line each put the last caveat below the bottom
    # edge, where the reader never sees it.
    height = (top + len(floors) * row_h + 34 + 18 + 2 * row_h + 46
              + sum(len(_wrap(t)) for t in tail) * 15 + 30)
    left, span = 300, float(width - 300 - 130)

    out = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" '
           'font-family="sans-serif">' % (width, height)]
    out.append('<rect width="%d" height="%d" fill="#ffffff"/>'
               % (width, height))
    out.append('<text x="10" y="22" font-size="15" fill="#222">M8 '
               'mask scoring: what the representation costs before any '
               'planner</text>')
    _caption(out, 10, 40, report.get("subtitle", ""), size=11, fill="#666")

    out.append('<text x="10" y="%d" font-size="12" fill="#333">'
               'round-trip floor, per present object-frame '
               '(blue IoU, orange contour F)</text>' % (top - 24))
    out.append('<text x="%d" y="%d" font-size="11" fill="#666">'
               'bits/object</text>' % (left + span + 20, top - 24))

    for i, f in enumerate(floors):
        y = top + i * row_h
        out.append('<text x="10" y="%d" font-size="11" fill="#333">%s</text>'
                   % (y + 13, _esc(f["name"])))
        out.append('<rect x="%d" y="%d" width="%.1f" height="8" '
                   'fill="#1f6feb"/>' % (left, y, span * float(f["iou"])))
        out.append('<rect x="%d" y="%d" width="%.1f" height="8" '
                   'fill="#e08a1e"/>'
                   % (left, y + 9, span * float(f["boundary_f"])))
        out.append('<text x="%.1f" y="%d" font-size="10" fill="#666">'
                   '%.3f / %.3f</text>'
                   % (left + span + 6, y + 13, f["iou"], f["boundary_f"]))
        out.append('<text x="%d" y="%d" font-size="10" fill="#444">%d</text>'
                   % (left + span + 78, y + 13, int(f["bits"])))

    y = top + len(floors) * row_h + 34
    out.append('<text x="10" y="%d" font-size="12" fill="#333">what IoU '
               'cannot see: two predictions losing the same area</text>' % y)
    pairs = (("thin strip at each end", blind.get("iou_thin"),
              blind.get("bf_thin")),
             ("one thick slab at one end", blind.get("iou_thick"),
              blind.get("bf_thick")))
    for k, (name, iou, bf) in enumerate(pairs):
        yy = y + 18 + k * row_h
        out.append('<text x="10" y="%d" font-size="11" fill="#333">%s</text>'
                   % (yy + 13, _esc(name)))
        if iou is None or bf is None:
            continue
        out.append('<rect x="%d" y="%d" width="%.1f" height="8" '
                   'fill="#1f6feb"/>' % (left, yy, span * float(iou)))
        out.append('<rect x="%d" y="%d" width="%.1f" height="8" '
                   'fill="#e08a1e"/>' % (left, yy + 9, span * float(bf)))
        out.append('<text x="%.1f" y="%d" font-size="10" fill="#666">'
                   '%.3f / %.3f</text>'
                   % (left + span + 6, yy + 13, iou, bf))

    y = y + 18 + len(pairs) * row_h + 26
    out.append('<text x="10" y="%d" font-size="12" fill="#333">agreement '
               'with the box metric on rectangular masks</text>' % y)
    y += 20
    for t in tail[:-1]:
        y = _caption(out, 10, y, t, size=11, fill="#444") + 4
    _caption(out, 10, y + 2, tail[-1], size=10, fill="#888")
    out.append('</svg>')

    with open(path, "w") as f:
        f.write("".join(out))


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def blindness_demo():
    """Two predictions with identical IoU and different contour scores.

    Kept in the module rather than in the run script so the figure's claim
    about the two metrics is produced by the same code the tests check.
    """
    def sq(r0, r1, c0, c1):
        m = np.zeros((50, 50), dtype=bool)
        m[r0:r1, c0:c1] = True
        return m

    gt = sq(10, 30, 10, 30)
    thin = sq(12, 28, 10, 30)
    thick = sq(14, 30, 10, 30)
    return {"iou_thin": mask_iou(thin, gt), "iou_thick": mask_iou(thick, gt),
            "bf_thin": boundary_f(thin, gt), "bf_thick": boundary_f(thick, gt)}


def box_code_floor(boxes, bins_x, bins_y, height, width, tol=DEFAULT_TOL):
    """The box code's floor, expressed in the mask metrics, for comparison.

    Puts the two representations on one axis. The box code is round-tripped by
    `oracle`, both sides are rasterised, and the same two mask metrics are
    applied — so the only difference between this row and the mask row is the
    code itself.

    One asymmetry has to be stated rather than hidden. The box code decodes to
    the bin's **left edge** because that is what a trained decoder emits
    (`oracle.dequantise`), which is a deliberate half-bin bias, and the grid
    code decodes by majority, which has none. The box row is therefore the
    floor of the representation *a model actually learns*, and the mask row is
    the floor of a representation *no model has learned yet*.
    """
    from tools.planner.oracle import boxes_to_latents, latents_to_boxes

    z = boxes_to_latents(boxes, bins_x, bins_y, width, height)
    back = latents_to_boxes(z, boxes.shape[1], bins_x, bins_y, width, height)

    gm = boxes_to_masks(boxes, height, width)
    pm = boxes_to_masks(back, height, width)
    present = gm.any(axis=(2, 3))

    ious, bfs = [], []
    for t in range(gm.shape[0]):
        for o in range(gm.shape[1]):
            if not present[t, o]:
                continue
            v = mask_iou(pm[t, o], gm[t, o])
            f = boundary_f(pm[t, o], gm[t, o], tol=tol)
            ious.append(0.0 if v is None else v)
            bfs.append(0.0 if f is None else f)

    n = len(ious)
    return {
        "mean_iou": float(np.mean(ious)) if n else None,
        "mean_dice": (float(np.mean([dice_from_iou(v) for v in ious]))
                      if n else None),
        "mean_boundary_f": float(np.mean(bfs)) if n else None,
        "n_present": n,
        "vanished": int((present & ~pm.any(axis=(2, 3))).sum()),
        "bits_per_object": 2 * int(bins_x) + 2 * int(bins_y),
        "bins": [int(bins_x), int(bins_y)],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Score against masks, and convert masks to boxes. "
                    "Validates on synthetic masks and on box corpora read as "
                    "rectangular masks, because no mask corpus is on disk.")
    ap.add_argument("annotation", nargs="?", default=None,
                    help="a VidVRD annotation json. Omit it to run the "
                         "synthetic half only.")
    ap.add_argument("--limit", type=int, default=40,
                    help="frames to read. Masks are big: 40 frames of 3 "
                         "objects on the 300x200 canvas is about 7 MB.")
    ap.add_argument("--max-objects", type=int, default=3)
    ap.add_argument("--bins", type=int, default=None,
                    help="cells per axis. Default matches the real decoder, "
                         "%d in x and %d in y." % (DEFAULT_BINS_X,
                                                   DEFAULT_BINS_Y))
    ap.add_argument("--tol", type=int, default=DEFAULT_TOL,
                    help="contour tolerance in pixels")
    ap.add_argument("--out-dir", default="eval/M8")
    a = ap.parse_args(argv)

    bins_x = a.bins if a.bins else DEFAULT_BINS_X
    bins_y = a.bins if a.bins else DEFAULT_BINS_Y

    report = {"bins": [bins_x, bins_y], "tol": a.tol,
              "blindness": blindness_demo(), "floors": []}

    print("what IoU cannot see")
    print("  thin strip at each end     iou %.3f   contour F %.3f"
          % (report["blindness"]["iou_thin"], report["blindness"]["bf_thin"]))
    print("  one thick slab at one end  iou %.3f   contour F %.3f"
          % (report["blindness"]["iou_thick"],
             report["blindness"]["bf_thick"]))

    if a.annotation is None:
        report["subtitle"] = "synthetic masks only; no corpus was given"
        report["agreement"] = {"max_abs_deviation": None, "n_pairs": 0,
                               "source": "none"}
    else:
        from tools.planner.oracle import boxes_from_vidvrd

        boxes, meta = boxes_from_vidvrd(a.annotation,
                                        num_objs=a.max_objects)
        boxes = boxes[:a.limit]
        name = os.path.basename(a.annotation)
        if name.endswith(".json"):
            name = name[:-5]
        print("\nread %d of %d annotated frames from %s"
              % (len(boxes), meta["frames"], name))

        # Two agreement rows, because they answer different questions. On
        # integer boxes the two metrics must agree EXACTLY, and any gap is a
        # bug in the mask metric. On the raw canvas boxes, which land between
        # pixels after the rescale, the gap is the price of rasterising and it
        # is a property of the conversion rather than of the metric.
        rounded = np.round(boxes)
        agree = agreement_with_boxes(rounded + 4.0, rounded, CANVAS_H,
                                     CANVAS_W)
        agree_raw = agreement_with_boxes(boxes + 4.0, boxes, CANVAS_H,
                                         CANVAS_W)
        for r in (agree, agree_raw):
            # The full lists are for a caller, not for a json anyone reads.
            r.pop("box_ious", None)
            r.pop("mask_ious", None)
        report["agreement"] = dict(agree)
        report["agreement"]["source"] = ("%s, integer boxes shifted 4 px and "
                                         "read as rectangular masks" % name)
        report["agreement_raw"] = dict(agree_raw)
        already_integer = bool(np.all(boxes == np.round(boxes)))
        report["boxes_already_integer"] = already_integer
        if already_integer:
            report["agreement_note"] = (
                "the raw canvas boxes of this corpus are already whole "
                "pixels, because the canvas scaler rounds, so the rasteriser "
                "costs nothing here and this row repeats the one above. The "
                "between-pixel case is exercised only by the test suite.")
        else:
            report["agreement_note"] = (
                "on the raw canvas boxes, which sit between pixels, the gap "
                "is %.4f at worst and %.4f on average over %d object-frames: "
                "the cost of rasterising, not of the metric"
                % (agree_raw["max_abs_deviation"],
                   agree_raw["mean_abs_deviation"], agree_raw["n_pairs"]))
        print("agreement with bbox_iou over %d object-frames"
              % agree["n_pairs"])
        print("  integer boxes    max gap %.3e   mean gap %.3e"
              % (agree["max_abs_deviation"], agree["mean_abs_deviation"]))
        print("  raw canvas boxes max gap %.3e   mean gap %.3e"
              % (agree_raw["max_abs_deviation"],
                 agree_raw["mean_abs_deviation"]))

        masks = boxes_to_masks(boxes, CANVAS_H, CANVAS_W)
        box_floor = box_code_floor(boxes, bins_x, bins_y, CANVAS_H, CANVAS_W,
                                   tol=a.tol)
        report["box_floor"] = box_floor
        report["floors"].append(
            {"name": "box one-hot %dx%d (left-edge decode)"
                     % (bins_x, bins_y),
             "bits": box_floor["bits_per_object"],
             "iou": box_floor["mean_iou"],
             "boundary_f": box_floor["mean_boundary_f"]})

        report["mask_floors"] = {}
        for thr, label in ((0.5, "majority"), (0.01, "any coverage")):
            f = round_trip_masks(masks, bins_x, bins_y, threshold=thr,
                                 tol=a.tol)
            report["mask_floors"]["%.2f" % thr] = f
            report["floors"].append(
                {"name": "mask grid %dx%d (%s)" % (bins_x, bins_y, label),
                 "bits": f["bits_per_object"],
                 "iou": f["mean_iou"],
                 "boundary_f": f["mean_boundary_f"]})

        report["subtitle"] = (
            "%s, %d frames, %d objects. Masks are rectangles here, so this is "
            "the code's cost on boxes, not on real segmentation shapes."
            % (name, len(boxes), a.max_objects))

        print("\nround-trip floor, per present object-frame")
        print("  %-42s %8s %8s %8s %8s %9s"
              % ("representation", "bits/obj", "IoU", "Dice", "contourF",
                 "vanished"))
        for row, src in ((report["floors"][0], box_floor),
                         (report["floors"][1],
                          report["mask_floors"]["0.50"]),
                         (report["floors"][2],
                          report["mask_floors"]["0.01"])):
            print("  %-42s %8d %8.4f %8.4f %8.4f %9d"
                  % (row["name"], row["bits"], row["iou"], src["mean_dice"],
                     row["boundary_f"], src["vanished"]))
        print("  Dice is IoU restated, 2J/(1+J). It ranks every row the same "
              "way and is not a second method.")

    if not os.path.isdir(a.out_dir):
        os.makedirs(a.out_dir)
    with open(os.path.join(a.out_dir, "m8_mask.json"), "w") as f:
        json.dump(report, f, indent=2)
    fig = os.path.join(a.out_dir, "m8_mask.svg")
    write_figure(report, fig)
    print("\nwrote %s/m8_mask.json and m8_mask.svg" % a.out_dir)
    print("PVSG is not on disk, so no number here comes from a real mask "
          "corpus. Read experiments/M8_mask_metric/README.md before deciding "
          "what these mean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
