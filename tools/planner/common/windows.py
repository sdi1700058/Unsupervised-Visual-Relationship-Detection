#!/usr/bin/env python3
"""tools/planner/common/windows.py — sliding-window interpolation task.

SPEC.md §C17 (revised 2026-08-03). The evaluation task is frame interpolation
by classical planning:

    Given frame f_i (init) and frame f_{i+k-1} (goal), the planner must
    reconstruct the k-2 intermediate frames f_{i+1} ... f_{i+k-2}.

Why this task and not `frame 0 -> frame N`:

- Horizon. A 135-frame video needs a 134-step plan for the whole clip.
  Fast Downward does not solve that on a 800-bit propositional state space.
  A window of k=5 needs a 4-step plan. That is seconds of CPU.
- Sample count. One video gives `N - k + 1` windows, not 1 sample. A
  135-frame clip at k=5 gives 131 windows. This gives statistical power.
- Plan length. The number of transitions between k frames is exactly k-1.
  The plan length target is therefore known, and a length mismatch is
  itself a measurable failure mode.
- Metric clarity. The bbox error is measured on the k-2 intermediate
  frames only. The planner receives the endpoints, so scoring them is
  free credit.

The window stride is a knob. `stride=1` gives maximum samples with heavy
overlap. `stride=k-1` gives disjoint windows.
"""

from __future__ import annotations


def make_windows(n_frames, k, stride=1, max_windows=None):
    """Return the list of (init_idx, goal_idx, intermediate_indices) tuples.

    Parameters
    ----------
    n_frames    : int   number of frames in the video.
    k           : int   window size in frames. Must be >= 3 (need at least
                        one intermediate frame to score).
    stride      : int   step between window starts.
    max_windows : int   optional cap (useful for smoke runs).

    Returns
    -------
    windows : list of dict with keys
        'init'          int  index of the init frame
        'goal'          int  index of the goal frame
        'intermediate'  list[int] indices of the frames to reconstruct
        'plan_length'   int  expected plan length = k - 1
    """
    if k < 3:
        raise ValueError(f"k must be >= 3 to have an intermediate frame; got {k}")
    if n_frames < k:
        raise ValueError(f"n_frames {n_frames} < window size {k}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1; got {stride}")

    windows = []
    for i in range(0, n_frames - k + 1, stride):
        goal = i + k - 1
        windows.append({
            "init": i,
            "goal": goal,
            "intermediate": list(range(i + 1, goal)),
            "plan_length": k - 1,
        })
        if max_windows is not None and len(windows) >= max_windows:
            break
    return windows


def extract_intermediate_states(plan_trace, expected_count):
    """Return the intermediate latents from a plan trace.

    A plan trace holds `init, s_1, ..., s_{L-1}, goal` for a plan of length L.
    The intermediate states are the entries between the endpoints.

    When the plan length does not equal the expected transition count, the
    trace is resampled to `expected_count` intermediate steps by nearest
    index. The caller records the mismatch through `plan_length_match`.

    Parameters
    ----------
    plan_trace     : np.ndarray (L+1, U*P)
    expected_count : int  number of intermediate frames to reconstruct (k-2)

    Returns
    -------
    intermediates : np.ndarray (expected_count, U*P)
    exact         : bool  True when no resampling occurred
    """
    import numpy as np

    plan_trace = np.asarray(plan_trace)
    if plan_trace.ndim != 2:
        raise ValueError(f"plan_trace must be 2-D; got shape {plan_trace.shape}")

    n_states = len(plan_trace)
    if n_states < 2:
        # A degenerate plan cannot supply an intermediate state. Repeat init.
        return np.repeat(plan_trace[:1], expected_count, axis=0), False

    interior = plan_trace[1:-1]
    if len(interior) == expected_count:
        return interior, True

    if expected_count == 0:
        return interior[:0], len(interior) == 0

    if len(interior) == 0:
        # Plan jumped straight from init to goal. Repeat the init state.
        return np.repeat(plan_trace[:1], expected_count, axis=0), False

    idx = np.linspace(0, len(interior) - 1, expected_count).round().astype(int)
    return interior[idx], False


def linear_interp_bboxes(bbox_init, bbox_goal, n_intermediate):
    """Return the linear-interpolation baseline for the intermediate frames.

    This is the trivial predictor. A planner that does not beat it has not
    learned usable dynamics from the video.

    Parameters
    ----------
    bbox_init, bbox_goal : np.ndarray (num_objs, 4)
    n_intermediate       : int

    Returns
    -------
    bboxes : np.ndarray (n_intermediate, num_objs, 4)
    """
    import numpy as np

    bbox_init = np.asarray(bbox_init, dtype=np.float32)
    bbox_goal = np.asarray(bbox_goal, dtype=np.float32)
    if n_intermediate <= 0:
        return np.zeros((0,) + bbox_init.shape, dtype=np.float32)

    # Weights strictly inside the open interval (0, 1).
    alphas = np.linspace(0.0, 1.0, n_intermediate + 2)[1:-1]
    return np.stack([
        (1.0 - a) * bbox_init + a * bbox_goal for a in alphas
    ]).astype(np.float32)
