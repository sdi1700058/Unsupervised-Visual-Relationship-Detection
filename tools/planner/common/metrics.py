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

    sq = np.zeros((n_frames, n_objs))
    for i in range(n_objs):
        j = int(mapping[i])
        if j < 0:
            continue
        d = pred[:, i, :] - gt[:, j, :]
        sq[:, i] = np.sum(d * d, axis=-1)

    return {
        "mean_mse": float(sq.mean()),
        "per_frame_mse": [float(v) for v in sq.mean(axis=1)],
        "per_object_mse": [float(v) for v in sq.mean(axis=0)],
        "mapping": [int(v) for v in mapping],
        "matching_mode": matching,
    }


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

    if baseline_trace is not None:
        base = bbox_mse(baseline_trace, gt_boxes, matching)
        result["baseline_linear"] = base
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
        if "baseline_linear" in scores:
            out["baseline_mse_mean"] = scores["baseline_linear"]["mean_mse"]
            out["mse_ratio"] = scores.get("mse_ratio")
            out["beats_baseline"] = scores.get("beats_baseline")

    if extra:
        out.update(extra)
    return out
