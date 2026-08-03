#!/usr/bin/env python3
"""Shared run path for every planner method.

Each method only has to answer one question: given the init latent and the
goal latent, what is the sequence of states between them? Everything either
side of that -- loading the model, encoding the frames, decoding the plan,
scoring against the annotations, writing metrics.json -- lives here so the
three methods stay comparable.
"""

import json
from pathlib import Path


def run_window(model_dir, npz_path, init_idx, goal_idx, out_dir, solve,
               time_budget_s=60, matching="hungarian", plan_only=False,
               method="unknown", solve_kwargs=None):
    """Plan across one window and score it.

    `solve` is called as solve(z_init, z_goal, z_all, time_budget_s, out_dir,
    **solve_kwargs) and returns (found, trace, wall_s, extra) where trace is
    an (L+1, U*P) array holding the init state, the intermediate states and
    the goal state.
    """
    import numpy as np

    from tools.planner.common.encode import load_model, load_npz_states, encode_all
    from tools.planner.common.decode import decode_trace_to_bboxes
    from tools.planner.common.metrics import score_window, summarize
    from tools.planner.common.windows import (
        extract_intermediate_states, linear_interp_bboxes)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    net = load_model(model_dir)
    states, gt_boxes, _names, _frame_ids = load_npz_states(model_dir, npz_path)
    n_frames = len(states)

    init_idx = init_idx if init_idx >= 0 else n_frames + init_idx
    goal_idx = goal_idx if goal_idx >= 0 else n_frames + goal_idx
    if not (0 <= init_idx < n_frames and 0 <= goal_idx < n_frames):
        raise IndexError(f"window {init_idx}..{goal_idx} outside [0,{n_frames})")
    if goal_idx <= init_idx:
        raise ValueError(f"goal frame {goal_idx} must come after init {init_idx}")

    window = {
        "init": init_idx,
        "goal": goal_idx,
        "intermediate": list(range(init_idx + 1, goal_idx)),
        "plan_length": goal_idx - init_idx,
    }
    n_mid = len(window["intermediate"])

    z_all = encode_all(net, states)
    z_init, z_goal = z_all[init_idx], z_all[goal_idx]
    hamming = int((z_init ^ z_goal).sum())
    print(f"latents differ in {hamming} of {z_init.size} bits; "
          f"{n_mid} frames to reconstruct")

    found, trace, wall, extra = solve(
        z_init=z_init, z_goal=z_goal, z_all=z_all,
        time_budget_s=time_budget_s, out_dir=out_dir,
        **(solve_kwargs or {}))

    plan_length = max(0, len(trace) - 1) if found else 0
    print(f"plan length {plan_length} (expected {window['plan_length']}) "
          f"in {wall:.2f}s")

    scores = None
    if found and not plan_only and n_mid > 0 and gt_boxes is not None:
        mid_latents, exact = extract_intermediate_states(trace, n_mid)
        pred_boxes = decode_trace_to_bboxes(net, mid_latents)
        gt_window = gt_boxes[window["intermediate"]].astype(np.float32)
        baseline = linear_interp_bboxes(
            gt_boxes[init_idx], gt_boxes[goal_idx], n_mid)

        scores = score_window(pred_boxes, gt_window, baseline, matching)
        scores["resample_free"] = exact

        (out_dir / "plan_trace.json").write_text(json.dumps({
            "window": window,
            "latents": trace.astype(int).tolist(),
            "intermediate_bboxes": pred_boxes.tolist(),
        }, indent=2))

    metrics = summarize(found, plan_length, wall, window, scores,
                        extra={"method": method,
                               "hamming_init_goal": hamming, **(extra or {})})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if scores:
        print(f"bbox mse {scores['planner']['mean_mse']:.2f} vs "
              f"baseline {scores['baseline_linear']['mean_mse']:.2f} "
              f"(ratio {scores['mse_ratio']:.3f})")
    return metrics
