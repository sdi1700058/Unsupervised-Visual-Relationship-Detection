"""Shared run path for every planner method.

Each method answers one question: given the init latent and the goal latent,
what states lie between them? Everything either side of that lives here so
the three methods stay comparable.

The frames come from a planner export rather than from the model, so this
runs with numpy alone. See tools/planner/export_latents.py for how an export
is made and why the split exists.
"""

import json
from pathlib import Path


def run_window(export_path, init_idx, goal_idx, out_dir, solve,
               time_budget_s=60, matching="hungarian", plan_only=False,
               method="unknown", solve_kwargs=None):
    """Plan across one window and score it.

    `solve` is called as solve(z_init, z_goal, z_all, time_budget_s, out_dir,
    **solve_kwargs) and returns (found, trace, wall_s, extra). The trace holds
    the init state, the states between, and the goal state.
    """
    import numpy as np

    from tools.planner.common.export import load as load_export
    from tools.planner.common.metrics import score_window, summarize
    from tools.planner.common.windows import (
        extract_intermediate_states, linear_interp_bboxes)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    export = load_export(export_path)
    n_frames = len(export)

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

    z_all = export.latents
    z_init, z_goal = z_all[init_idx], z_all[goal_idx]
    hamming = int((z_init ^ z_goal).sum())
    print(f"latents differ in {hamming} of {export.n_bits} bits; "
          f"{n_mid} frames to reconstruct")

    # plan_length goes to the method because the interpolation task wants a
    # trajectory of exactly this many steps, not the cheapest plan between
    # the two ends. A method free to return its shortest plan will collapse a
    # three-frame window into one action whenever the two latents are close.
    found, trace, wall, extra = solve(
        z_init=z_init, z_goal=z_goal, z_all=z_all, export=export,
        time_budget_s=time_budget_s, out_dir=out_dir,
        plan_length=window["plan_length"],
        **(solve_kwargs or {}))

    plan_length = max(0, len(trace) - 1) if found else 0
    print(f"plan length {plan_length} (expected {window['plan_length']}) "
          f"in {wall:.2f}s")

    scores = None
    if found and not plan_only and n_mid > 0:
        mid_latents, exact = extract_intermediate_states(trace, n_mid)
        pred_boxes = export.boxes_for_trace(mid_latents)
        gt_window = export.gt_boxes[window["intermediate"]]
        baseline = linear_interp_bboxes(
            export.gt_boxes[init_idx], export.gt_boxes[goal_idx], n_mid)

        scores = score_window(pred_boxes, gt_window, baseline, matching)
        scores["resample_free"] = exact

        (out_dir / "plan_trace.json").write_text(json.dumps({
            "window": window,
            "latents": trace.astype(int).tolist(),
            "intermediate_bboxes": pred_boxes.tolist(),
        }, indent=2))

    metrics = summarize(found, plan_length, wall, window, scores,
                        extra={"method": method,
                               "hamming_init_goal": hamming,
                               "decode_fallbacks": export.fallback_count,
                               **(extra or {})})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if scores:
        planner_mse = scores["planner"]["mean_mse"]
        base_mse = scores["baseline_linear"]["mean_mse"]
        ratio = scores.get("mse_ratio")
        line = f"bbox mse {planner_mse:.2f} vs baseline {base_mse:.2f}"
        if ratio is None:
            # The straight line was exact, so the objects moved linearly and
            # this window cannot separate a good model from a lucky one.
            line += " (baseline is exact; window carries no signal)"
        else:
            line += f" (ratio {ratio:.3f})"
        print(line)
    if export.fallback_count:
        print(f"note: {export.fallback_count} latents were not in the export "
              "and fell back to the nearest one")
    return metrics
