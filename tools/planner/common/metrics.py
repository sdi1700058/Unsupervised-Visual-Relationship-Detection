#!/usr/bin/env python3
"""Score a plan against the real video frames.

The planner receives frame i and frame i+k-1. It must reconstruct the k-2
frames in between. We decode each intermediate latent to bounding boxes and
measure how far they sit from the annotated boxes.

A raw error number means nothing on its own, so every run also scores the
linear-interpolation baseline: draw a straight line from the init boxes to
the goal boxes. A planner that does not beat that line has not learned the
dynamics of the video.
"""

import itertools


def _assign(cost):
    """Solve the assignment problem. scipy if present, brute force if not.

    MAX_OBJECTS is 10 (SPEC V4), so 10! = 3.6M permutations is the worst
    case for the fallback. That is slow but survivable, and the video
    overfit runs use --max-objects 3.
    """
    import numpy as np

    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost)
    except ImportError:
        pass

    n = cost.shape[0]
    if n > 10:
        raise RuntimeError(
            f"no scipy and n={n} exceeds the brute-force cap of 10. "
            "pip install scipy")

    best, best_cost = None, float("inf")
    for perm in itertools.permutations(range(n)):
        c = sum(cost[i, perm[i]] for i in range(n))
        if c < best_cost:
            best, best_cost = perm, c
    return np.arange(n), np.array(best, dtype=np.int64)


def match_slots(pred_boxes, gt_boxes, gt_present=None):
    """Pair decoded object slots with annotated objects.

    The decoder has no reason to keep slot order stable, so the pairing is
    solved once for the whole window and reused for every frame. Solving per
    frame would let the mapping drift and would flatter the score.

    Accepts either a single frame, `(n_objs, 4)`, or a whole window,
    `(T, n_objs, 4)`. **Prefer the window.** Solving on one frame means
    solving on frame 0 whether or not anything is annotated there, and an
    all-zero ground-truth box is not a box at the origin — it means the object
    is not in that frame. Costing against it made the zero box a strong
    attractor, and the review of 2026-08-30 measured the consequence: the
    planner's pairing was wrong on 32.8% of realistic windows, and because a
    wrong pairing raises the *baseline*'s error 10 times more often than it
    lowers it, `mse_ratio` moved about 6% in this project's own favour.

    So the cost is accumulated over the frames where the ground-truth object
    is actually present, and objects excluded by `gt_present` take no part in
    the assignment at all.

    Returns an array where pred slot i corresponds to gt object mapping[i],
    or -1 when the slot has no partner.
    """
    import numpy as np

    pred = np.asarray(pred_boxes, dtype=np.float64)
    gt = np.asarray(gt_boxes, dtype=np.float64)
    if pred.ndim == 2:
        pred = pred[None, :, :]
    if gt.ndim == 2:
        gt = gt[None, :, :]

    n_pred, n_gt = pred.shape[1], gt.shape[1]
    n = max(n_pred, n_gt)

    # An object is present in a frame when its box is not all zeros.
    present = np.abs(gt).sum(axis=-1) > 0            # (T, n_gt)
    if gt_present is not None:
        present = present & np.asarray(gt_present, dtype=bool)[None, :]

    cost = np.full((n, n), 1e6)
    for j in range(n_gt):
        rows = np.nonzero(present[:, j])[0]
        if rows.size == 0:
            # Never present, or excluded. Leave the column at the sentinel so
            # the assignment treats it as unpairable rather than as a box.
            continue
        for i in range(n_pred):
            d = pred[rows, i, :] - gt[rows, j, :]
            cost[i, j] = float((d * d).sum() / rows.size)

    rows, cols = _assign(cost)
    mapping = np.full(n_pred, -1, dtype=np.int64)
    for r, c in zip(rows, cols):
        if r < n_pred and c < n_gt and cost[r, c] < 1e6:
            mapping[r] = c
    return mapping


def bbox_mse(pred_trace, gt_boxes, matching="hungarian", scoreable=None):
    """Mean squared bbox error over a window, in canvas pixels.

    pred_trace and gt_boxes both have shape (T, num_objs, 4) and cover the
    same T frames.
    """
    import numpy as np

    pred = np.asarray(pred_trace, dtype=np.float64)
    gt = np.asarray(gt_boxes, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    n_frames, n_objs, _ = pred.shape

    if matching == "hungarian":
        # The whole window, not frame 0: see `match_slots`.
        mapping = match_slots(pred, gt, gt_present=scoreable)
    elif matching == "fixed":
        mapping = np.arange(n_objs, dtype=np.int64)
    else:
        raise ValueError(f"unknown matching mode: {matching}")

    if not (mapping >= 0).any():
        # Every object is absent or excluded. That is a window with no data,
        # which the caller has to be able to tell apart from a perfect score.
        return {
            "mean_mse": None,
            "per_frame_mse": [0.0] * n_frames,
            "per_object_mse": [0.0] * n_objs,
            "mapping": [int(v) for v in mapping],
            "matching_mode": matching,
            "skipped_absent": int(n_frames * n_objs),
        }

    # A ground-truth box of all zeros means the object is not in that frame,
    # not that it sits at the origin. Squaring the distance to it would score
    # a prediction against nothing, which is what `bbox_iou` already refuses
    # to do. On 23% of the winnable VidVRD clips at least one object does not
    # span every frame, so this is not a corner case.
    sq = np.zeros((n_frames, n_objs))
    scored = np.zeros((n_frames, n_objs), dtype=bool)
    for i in range(n_objs):
        j = int(mapping[i])
        if j < 0:
            continue
        present = np.abs(gt[:, j, :]).sum(axis=-1) > 0
        if scoreable is not None:
            present = present & bool(scoreable[j])
        d = pred[:, i, :] - gt[:, j, :]
        sq[:, i] = np.where(present, np.sum(d * d, axis=-1), 0.0)
        scored[:, i] = present

    n_scored = int(scored.sum())
    per_frame = np.divide(sq.sum(axis=1), scored.sum(axis=1),
                          out=np.zeros(n_frames),
                          where=scored.sum(axis=1) > 0)
    per_object = np.divide(sq.sum(axis=0), scored.sum(axis=0),
                           out=np.zeros(n_objs),
                           where=scored.sum(axis=0) > 0)

    return {
        # None, not 0.0. Zero is the BEST possible score, so returning it for
        # "nothing was scoreable" put a perfect window into the reported
        # median. Measured on 2026-08-30: a window with every object absent
        # and a planner 400 px wrong on every frame scored 0.0 and passed the
        # report's motion filter.
        "mean_mse": float(sq.sum() / n_scored) if n_scored else None,
        "per_frame_mse": [float(v) for v in per_frame],
        "per_object_mse": [float(v) for v in per_object],
        "mapping": [int(v) for v in mapping],
        "matching_mode": matching,
        "skipped_absent": int(scored.size - n_scored),
    }


def bbox_iou(pred_trace, gt_boxes, mapping=None, matching="hungarian",
             scoreable=None):
    """Mean intersection-over-union between predicted and annotated boxes.

    Reported alongside `bbox_mse` because squared pixel error is not
    comparable across clips. A large box displaced by ten pixels and a small
    box displaced by ten pixels score the same under MSE, and the same
    absolute error means something different on a dog filling the frame than
    on a bird crossing it. IoU is scale-invariant and bounded in [0, 1].

    It is also the quantity the video relation literature localises with: the
    standard protocol thresholds volumetric IoU over a box trajectory at 0.5
    (see `.claude/docs/EVAL_CONSIDERED.md`). Reporting it makes the result
    legible to readers calibrated to that convention rather than to pixel MSE.

    `mapping` reuses the slot assignment already solved by `bbox_mse`, so the
    two metrics describe the same pairing. `scoreable` must be the same mask
    `bbox_mse` was given, so they also describe the same **frames**. Before
    2026-08-30 it was not passed, and the two metrics disagreed by
    construction: an object that left the scene scored `bbox_mse` 0.0 (frames
    correctly excluded) and `bbox_iou` 0.625 (the same frames scored as
    misses).
    """
    import numpy as np

    pred = np.asarray(pred_trace, dtype=np.float64)
    gt = np.asarray(gt_boxes, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    n_frames, n_objs, _ = pred.shape
    if mapping is None:
        if matching == "hungarian":
            mapping = match_slots(pred, gt, gt_present=scoreable)
        else:
            mapping = np.arange(n_objs, dtype=np.int64)
    mapping = np.asarray(mapping)

    per_frame = np.zeros(n_frames)
    ious = np.zeros((n_frames, n_objs))
    # Per frame, not per window: an object can be present in some frames of
    # the window and absent in others, and dividing by a window-wide count
    # spread its absent frames across the ones where it was there.
    counted_per_frame = np.zeros(n_frames)
    for i in range(n_objs):
        j = int(mapping[i])
        if j < 0:
            continue
        present = np.abs(gt[:, j, :]).sum(axis=-1) > 0
        if scoreable is not None:
            present = present & bool(scoreable[j])
        counted_per_frame += present
        p, g = pred[:, i, :], gt[:, j, :]
        x1 = np.maximum(p[:, 0], g[:, 0])
        y1 = np.maximum(p[:, 1], g[:, 1])
        x2 = np.minimum(p[:, 2], g[:, 2])
        y2 = np.minimum(p[:, 3], g[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_p = np.clip(p[:, 2] - p[:, 0], 0, None) * np.clip(p[:, 3] - p[:, 1], 0, None)
        area_g = np.clip(g[:, 2] - g[:, 0], 0, None) * np.clip(g[:, 3] - g[:, 1], 0, None)
        union = area_p + area_g - inter
        # A padded slot is an all-zero box on both sides. Union zero means
        # there is nothing to score, not a perfect overlap. `present` carries
        # the same exclusion `bbox_mse` applies, so an absent object is not
        # scored as a miss.
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        ious[:, i] = np.where(present, iou, 0.0)

    per_frame = np.divide(ious.sum(axis=1), counted_per_frame,
                          out=np.zeros(n_frames),
                          where=counted_per_frame > 0)
    scored_frames = counted_per_frame > 0

    return {
        "mean_iou": (float(per_frame[scored_frames].mean())
                     if scored_frames.any() else None),
        "per_frame_iou": [float(v) for v in per_frame],
        "frames_above_0.5": int((per_frame[scored_frames] >= 0.5).sum()),
        "n_frames": int(scored_frames.sum()),
    }


def action_effect_consistency(pre, suc, labels):
    """Does the same named action flip the same bits wherever it applies?

    **A second evaluation method, independent of frame interpolation.** The
    planning literature is unanimous that this is what makes a learned
    representation usable: a STRIPS operator has one add list and one delete
    list, so an action's effect must be constant. `RELATED_WORK.md` A6
    (Cube-Space AE) adds a training prior to enforce it; A4 rejects
    auto-encoding objectives precisely because they do not; A7 (DeepSym)
    grounds its symbols in action effects rather than reconstruction.

    Nothing in this pipeline measured it. A model could reconstruct every
    frame, score well on `val_loss`, and still give every transition its own
    idiosyncratic effect — in which case `pddl/planner.py` is not recovering an
    action model but memorising a list of observed jumps.

    The measure is the Jaccard overlap of the XOR deltas:

    - `within`  — mean overlap between deltas carrying the **same** label
    - `between` — mean overlap between deltas carrying **different** labels
    - `consistency` — `within - between`, clipped at 0

    1.0 means every instance of an action has an identical effect and different
    actions do not overlap: a clean operator set. 0.0 means the label predicts
    nothing about the effect, so there are no operators to find.

    `consistency` is None when no label repeats, since there is then nothing to
    compare within an action.
    """
    import numpy as np
    from itertools import combinations

    pre = np.asarray(pre, dtype=np.int8)
    suc = np.asarray(suc, dtype=np.int8)
    labels = list(labels)
    if len(pre) != len(suc) or len(pre) != len(labels):
        raise ValueError("pre, suc and labels must be the same length")

    deltas = (pre ^ suc).astype(bool)
    keep = [i for i in range(len(deltas)) if deltas[i].any()]
    if len(keep) < 2:
        return {"within": None, "between": None, "consistency": None,
                "n_transitions": len(keep), "n_actions": 0}

    def jaccard(a, b):
        union = int((a | b).sum())
        return float((a & b).sum()) / union if union else 0.0

    within, between = [], []
    for i, j in combinations(keep, 2):
        score = jaccard(deltas[i], deltas[j])
        (within if labels[i] == labels[j] else between).append(score)

    w = float(np.mean(within)) if within else None
    b = float(np.mean(between)) if between else 0.0
    n_actions = len({labels[i] for i in keep})

    # With one label there is no `between` set, so `within - between` reduces
    # to `within` and the measure cannot fail: eight unrelated random deltas
    # all labelled "a" scored 0.3159. A comparison needs two things to
    # compare, so this is undefined rather than low.
    if n_actions < 2 or w is None:
        consistency = None
    else:
        consistency = max(0.0, w - b)

    return {"within": w, "between": b, "consistency": consistency,
            "n_transitions": len(keep),
            "n_actions": n_actions}


def _rank(a):
    import numpy as np
    order = a.argsort()
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(len(a))
    return r


def latent_geometry(latents, gt_boxes, max_pairs=4000, seed=0):
    """Does Hamming distance in the latent track distance in the world?

    This decides whether a plan decodes to sensible boxes, and it is
    measurable from an export alone — no planner, no search, no budget.

    The reason it matters is `Export.boxes_for`. A plan's intermediate states
    are almost never latents the model actually produced, so they are decoded
    by falling back to the **Hamming-nearest observed latent**. If Hamming
    distance tracks position, that fallback lands near the truth. If it does
    not, the decoded box is arbitrary and no amount of search quality helps.

    Measured on `ILSVRC2015_train_00005005`, where the oracle's planner error
    is 13.67 and the trained model's is 101.44:

        oracle       spearman +0.651   nearest-neighbour box error  2.57 px
        trained P10  spearman +0.356                                6.10 px
        trained P20  spearman +0.343                               10.16 px

    **This is a screen, not a ranking**, and the distinction was measured
    rather than assumed. Across six oracle variants differing only in bin
    count, `spearman` correlates +0.934 with planner error *in the wrong
    direction*: coarse bins make the code a coarse position, which raises the
    correlation and raises the quantisation floor together. Within one
    encoding family the floor predicts error at +0.985 and this does not.

    - **Use it to catch a code that does not encode position at all.** The
      trained models sit at +0.34; no oracle variant falls below +0.53.
    - **Do not use it to choose between two codes of the same kind.** Higher
      there usually means coarser. Use the quantisation floor instead.

    Returns `spearman` (None when no pair of latents differs, since the
    correlation is then undefined rather than zero) and `nearest_box_error`,
    the mean positional error incurred by decoding a frame as its
    Hamming-nearest neighbour.
    """
    import numpy as np

    z = np.asarray(latents, dtype=np.int16)
    b = np.asarray(gt_boxes, dtype=np.float64)
    n = len(z)
    if n < 2:
        return {"spearman": None, "nearest_box_error": None, "n_frames": n}

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        rng = np.random.RandomState(seed)
        pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs,
                                              replace=False)]

    ham = np.array([int((z[i] ^ z[j]).sum()) for i, j in pairs],
                   dtype=np.float64)
    pos = np.array([float(np.sqrt(((b[i] - b[j]) ** 2).sum()))
                    for i, j in pairs])

    # A code where every frame is identical carries no information, and a
    # correlation over a constant is undefined, not zero.
    if ham.std() == 0 or pos.std() == 0:
        rho = None
    else:
        rh, rp = _rank(ham), _rank(pos)
        rh -= rh.mean()
        rp -= rp.mean()
        denom = np.sqrt((rh ** 2).sum() * (rp ** 2).sum())
        rho = float((rh * rp).sum() / denom) if denom > 0 else None

    big = float(10 ** 9)
    total = 0.0
    for i in range(n):
        h = np.array([int((z[i] ^ z[j]).sum()) if j != i else big
                      for j in range(n)])
        total += float(np.sqrt(((b[i] - b[int(h.argmin())]) ** 2).sum()))

    return {"spearman": rho,
            "nearest_box_error": total / n,
            "n_frames": n,
            "n_pairs": len(pairs)}


def floor_ratio(planner_mse, quantisation_floor):
    """`planner_error / quantisation_floor` — how close to the best possible.

    **A measure of the representation, where `mse_ratio` is a measure of the
    clip.** `mse_ratio` divides by the linear-interpolation baseline, and
    measured over 22 screened clips that baseline spans **1575x** (2.0 to
    3084.5) while the oracle's planner error spans **5x** (20.4 to 100.4). A
    ratio reports whichever of its parts varies most, so `mse_ratio` reports
    how non-linear a clip happens to be (`SPEC.md` V38).

    The quantisation floor spans only **1.9x** across the same clips, so
    dividing by it leaves the numerator in charge.

    Validated on two independent samples before being written, because two
    measures were already narrowed after shipping on small evidence this week::

        n=10, paired    floor_ratio CV 0.51   mse_ratio CV 0.79
        n=22, oracle    floor_ratio CV 0.52   mse_ratio CV 1.87   3.6x steadier

    For one fixed representation across 22 clips, `mse_ratio` ranged 0.01 to
    25.19 and `floor_ratio` 0.91 to 5.51.

    **Report both.** They answer different questions:

    - `mse_ratio < 1` — does this beat the trivial alternative? Practical
      utility, and the question the thesis's task statement asks.
    - `floor_ratio -> 1` — how close is this to the best any representation
      could do at this bin resolution? Representation quality.

    A value below 1 is possible rather than an error: slot matching can pair
    boxes more favourably than the identity the floor assumes.

    None when either input is missing or the floor is not positive (V29).
    """
    if planner_mse is None or quantisation_floor is None:
        return None
    if quantisation_floor <= 0:
        return None
    return float(planner_mse) / float(quantisation_floor)


def moving_gt_steps(gt_boxes, tol=1e-3):
    """Frame steps in the window where the annotated boxes actually move.

    A window whose objects stand still measures nothing. Linear interpolation
    is exact there, so `mse_ratio` divides by a number near zero and reports a
    ratio in the millions that says only that the denominator was small. Clip
    `00005005` opens with sixty consecutive motionless frame pairs, which is
    eight windows of pure noise at window 8.

    This counts from the **annotations**, not from the latents, which is the
    whole point of having it beside `moving_steps`: a model can change its code
    while the objects stand still, and that is a fact about the model rather
    than about the window being measurable.

    An object appearing or disappearing is **not** motion. Differencing raw
    boxes made a present-to-absent step look like a large jump, so a window
    where nothing moved but one object left scored as moving — and that was
    the gate that let a window with no scoreable data at all into the report.
    Measured and fixed 2026-08-30. The step is therefore computed only over
    objects present in **both** frames of a pair.
    """
    import numpy as np

    gt = np.asarray(gt_boxes, dtype=np.float64)
    if len(gt) < 2:
        return 0
    if gt.ndim == 2:                     # (T, 4): a single object
        gt = gt[:, None, :]

    present = np.abs(gt).sum(axis=-1) > 0            # (T, n_objs)
    both = present[:-1] & present[1:]                # (T-1, n_objs)
    delta = np.abs(np.diff(gt, axis=0)).sum(axis=-1)  # (T-1, n_objs)
    step = np.where(both, delta, 0.0).sum(axis=1)
    return int((step > tol).sum())


def mse_ratio(planner_mse, baseline_mse, floor=1e-6):
    """`planner_mse / baseline_mse`, or None when the ratio means nothing.

    The headline number of this project, so its degenerate cases matter more
    than its arithmetic. It is None when either side is missing, and None when
    the baseline is at or below `floor`: linear interpolation is exact on a
    motionless window, so dividing by its error reports how small the
    denominator was and nothing else. A baseline of 4.1e-10 previously
    produced a ratio of 243874974014.05, written verbatim into the CSV and the
    per-window table.
    """
    if planner_mse is None or baseline_mse is None:
        return None
    if baseline_mse <= floor:
        return None
    return float(planner_mse) / float(baseline_mse)


def temporal_order(pred_trace, gt_boxes, mapping=None, scoreable=None):
    """Spearman correlation between plan step and closest real frame.

    A plan can hit the right set of positions in the wrong order. This
    catches that. 1.0 means the plan walks the video forwards, 0 means the
    order carries no information, negative means it runs backwards.

    `mapping` and `scoreable` are the ones `bbox_mse` solved, and both are
    needed for this to describe the same thing the error does. Without the
    mapping it compared slot-for-slot, so a perfect forward plan with two
    slots swapped scored 0.0 while `bbox_mse` scored 0.0 error and IoU 1.0.
    """
    import numpy as np

    pred = np.asarray(pred_trace, dtype=np.float64)
    gt = np.asarray(gt_boxes, dtype=np.float64)
    n_steps = len(pred)
    if n_steps < 2:
        return None

    # Reorder the ground truth into the predictor's slot order, and keep only
    # the objects the error was computed over.
    if mapping is not None:
        mapping = np.asarray(mapping)
        keep = [i for i in range(len(mapping))
                if int(mapping[i]) >= 0
                and (scoreable is None or bool(scoreable[int(mapping[i])]))]
        if not keep:
            return None
        pred = pred[:, keep, :]
        gt = gt[:, [int(mapping[i]) for i in keep], :]

    # For each plan step, which real frame does it resemble most? Frames where
    # the object is not annotated carry no position, so they are not eligible.
    present = np.abs(gt).sum(axis=-1).sum(axis=-1) > 0
    if present.sum() < 2:
        return None

    nearest = []
    for t in range(n_steps):
        d = ((gt - pred[t]) ** 2).sum(axis=(1, 2))
        d = np.where(present, d, np.inf)
        nearest.append(int(np.argmin(d)))

    steps = np.arange(n_steps, dtype=np.float64)
    nearest = np.asarray(nearest, dtype=np.float64)
    if len(set(nearest.tolist())) < 2:
        return 0.0

    def rank(a):
        order = a.argsort()
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(a))
        return r

    rs, rn = rank(steps), rank(nearest)
    rs -= rs.mean()
    rn -= rn.mean()
    denom = np.sqrt((rs ** 2).sum() * (rn ** 2).sum())
    return float((rs * rn).sum() / denom) if denom > 0 else 0.0


def score_window(pred_trace, gt_boxes, baseline_trace=None,
                 matching="hungarian", endpoints=None):
    """Score one interpolation window against the real frames.

    baseline_trace is the linear-interpolation prediction. When given, the
    result carries the baseline error and the ratio between the two. A ratio
    below 1 means the planner beat the straight line.

    `endpoints` is the (init, goal) ground-truth box pair. When supplied, any
    object absent at either endpoint is dropped from **both** metrics.
    `linear_interp_bboxes` interpolates raw boxes, so an object missing at an
    endpoint has its straight line drawn from the origin, and the resulting
    baseline error is enormous — which flatters the planner. The exclusion has
    to be shared, or the two sides are measured on different frames.
    """
    import numpy as np

    scoreable = None
    if endpoints is not None:
        init, goal = (np.asarray(e, dtype=np.float64) for e in endpoints)
        scoreable = ((np.abs(init).sum(axis=-1) > 0)
                     & (np.abs(goal).sum(axis=-1) > 0))

    result = {"planner": bbox_mse(pred_trace, gt_boxes, matching, scoreable)}

    # Every metric below gets the SAME pairing and the SAME mask as the error.
    # Before 2026-08-30 `bbox_iou` and `temporal_order` got neither, so the
    # three numbers in one row described three different object-frame sets.
    mapping = result["planner"]["mapping"]
    result["temporal_order"] = temporal_order(pred_trace, gt_boxes,
                                              mapping=mapping,
                                              scoreable=scoreable)
    result["planner_iou"] = bbox_iou(pred_trace, gt_boxes, mapping=mapping,
                                     scoreable=scoreable)

    if baseline_trace is not None:
        base = bbox_mse(baseline_trace, gt_boxes, matching, scoreable)
        result["baseline_linear"] = base
        result["baseline_iou"] = bbox_iou(baseline_trace, gt_boxes,
                                          mapping=base["mapping"],
                                          scoreable=scoreable)
        planner_mse = result["planner"]["mean_mse"]
        base_mse = base["mean_mse"]
        result["mse_ratio"] = mse_ratio(planner_mse, base_mse)
        result["beats_baseline"] = bool(
            result["mse_ratio"] is not None and result["mse_ratio"] < 1.0)

    return result


def summarize(found, plan_length, wall_s, window=None, scores=None, extra=None):
    """Flatten a run into the metrics.json payload."""
    out = {
        "reachability": bool(found),
        "plan_length": int(plan_length),
        "wall_s": float(wall_s),
    }

    if window is not None:
        out["init_frame"] = window["init"]
        out["goal_frame"] = window["goal"]
        out["n_intermediate"] = len(window["intermediate"])
        out["expected_plan_length"] = window["plan_length"]
        out["plan_length_match"] = (plan_length == window["plan_length"])

    if scores is not None:
        out["bbox_mse_mean"] = scores["planner"]["mean_mse"]
        out["skipped_absent"] = scores["planner"].get("skipped_absent")
        out["bbox_mse_per_frame"] = scores["planner"]["per_frame_mse"]
        out["hungarian_mapping"] = scores["planner"]["mapping"]
        out["matching_mode"] = scores["planner"]["matching_mode"]
        out["temporal_order"] = scores.get("temporal_order")
        # SPEC V38: reported beside mse_ratio, because mse_ratio measures the
        # clip and this measures the representation.
        out["floor_ratio"] = scores.get("floor_ratio")
        out["quantisation_floor"] = scores.get("quantisation_floor")
        if "planner_iou" in scores:
            out["bbox_iou_mean"] = scores["planner_iou"]["mean_iou"]
            out["frames_iou_above_half"] = scores["planner_iou"]["frames_above_0.5"]
        if "baseline_iou" in scores:
            out["baseline_iou_mean"] = scores["baseline_iou"]["mean_iou"]
        if "baseline_linear" in scores:
            out["baseline_mse_mean"] = scores["baseline_linear"]["mean_mse"]
            out["mse_ratio"] = scores.get("mse_ratio")
            out["beats_baseline"] = scores.get("beats_baseline")

    if extra:
        out.update(extra)
    return out
