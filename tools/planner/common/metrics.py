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


def match_slots(pred_boxes, gt_boxes):
    """Pair decoded object slots with annotated objects.

    The decoder has no reason to keep slot order stable, so we solve the
    pairing once on the first scored frame and reuse it for the rest of the
    window. Solving per frame would let the mapping drift and would flatter
    the score.

    Returns an array where pred slot i corresponds to gt object mapping[i],
    or -1 when the slot has no partner.
    """
    import numpy as np

    n_pred, n_gt = pred_boxes.shape[0], gt_boxes.shape[0]
    n = max(n_pred, n_gt)

    cost = np.full((n, n), 1e6)
    for i in range(n_pred):
        for j in range(n_gt):
            d = pred_boxes[i] - gt_boxes[j]
            cost[i, j] = float(d @ d)

    rows, cols = _assign(cost)
    mapping = np.full(n_pred, -1, dtype=np.int64)
    for r, c in zip(rows, cols):
        if r < n_pred and c < n_gt:
            mapping[r] = c
    return mapping


def bbox_mse(pred_trace, gt_boxes, matching="hungarian"):
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
        mapping = match_slots(pred[0], gt[0])
    elif matching == "fixed":
        mapping = np.arange(n_objs, dtype=np.int64)
    else:
        raise ValueError(f"unknown matching mode: {matching}")

    if not (mapping >= 0).any():
        raise RuntimeError("no slot could be paired with an annotated object")

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
        "mean_mse": float(sq.sum() / n_scored) if n_scored else 0.0,
        "per_frame_mse": [float(v) for v in per_frame],
        "per_object_mse": [float(v) for v in per_object],
        "mapping": [int(v) for v in mapping],
        "matching_mode": matching,
        "skipped_absent": int(scored.size - n_scored),
    }


def bbox_iou(pred_trace, gt_boxes, mapping=None, matching="hungarian"):
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
    two metrics describe the same pairing.
    """
    import numpy as np

    pred = np.asarray(pred_trace, dtype=np.float64)
    gt = np.asarray(gt_boxes, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    n_frames, n_objs, _ = pred.shape
    if mapping is None:
        if matching == "hungarian":
            mapping = match_slots(pred[0], gt[0])
        else:
            mapping = np.arange(n_objs, dtype=np.int64)
    mapping = np.asarray(mapping)

    per_frame = np.zeros(n_frames)
    counted = 0
    ious = np.zeros((n_frames, n_objs))
    for i in range(n_objs):
        j = int(mapping[i])
        if j < 0:
            continue
        counted += 1
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
        # there is nothing to score, not a perfect overlap.
        ious[:, i] = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)

    if counted:
        per_frame = ious.sum(axis=1) / counted

    return {
        "mean_iou": float(per_frame.mean()) if n_frames else 0.0,
        "per_frame_iou": [float(v) for v in per_frame],
        "frames_above_0.5": int((per_frame >= 0.5).sum()),
        "n_frames": int(n_frames),
    }


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
    """
    import numpy as np

    gt = np.asarray(gt_boxes, dtype=np.float64)
    if len(gt) < 2:
        return 0
    step = np.abs(np.diff(gt, axis=0)).sum(axis=tuple(range(1, gt.ndim)))
    return int((step > tol).sum())


def temporal_order(pred_trace, gt_boxes):
    """Spearman correlation between plan step and closest real frame.

    A plan can hit the right set of positions in the wrong order. This
    catches that. 1.0 means the plan walks the video forwards, 0 means the
    order carries no information, negative means it runs backwards.
    """
    import numpy as np

    pred = np.asarray(pred_trace, dtype=np.float64)
    gt = np.asarray(gt_boxes, dtype=np.float64)
    n_steps = len(pred)
    if n_steps < 2:
        return None

    # For each plan step, which real frame does it resemble most?
    nearest = []
    for t in range(n_steps):
        d = ((gt - pred[t]) ** 2).sum(axis=(1, 2))
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
                 matching="hungarian"):
    """Score one interpolation window against the real frames.

    baseline_trace is the linear-interpolation prediction. When given, the
    result carries the baseline error and the ratio between the two. A ratio
    below 1 means the planner beat the straight line.
    """
    result = {"planner": bbox_mse(pred_trace, gt_boxes, matching)}
    result["temporal_order"] = temporal_order(pred_trace, gt_boxes)

    # Same slot pairing as the error, so the two describe one assignment.
    mapping = result["planner"]["mapping"]
    result["planner_iou"] = bbox_iou(pred_trace, gt_boxes, mapping=mapping)

    if baseline_trace is not None:
        base = bbox_mse(baseline_trace, gt_boxes, matching)
        result["baseline_linear"] = base
        result["baseline_iou"] = bbox_iou(baseline_trace, gt_boxes,
                                          mapping=base["mapping"])
        planner_mse = result["planner"]["mean_mse"]
        base_mse = base["mean_mse"]
        result["mse_ratio"] = (planner_mse / base_mse) if base_mse > 0 else None
        result["beats_baseline"] = bool(base_mse > 0 and planner_mse < base_mse)

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
        out["bbox_mse_per_frame"] = scores["planner"]["per_frame_mse"]
        out["hungarian_mapping"] = scores["planner"]["mapping"]
        out["matching_mode"] = scores["planner"]["matching_mode"]
        out["temporal_order"] = scores.get("temporal_order")
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
