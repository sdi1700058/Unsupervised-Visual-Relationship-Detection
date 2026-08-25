#!/usr/bin/env python3
"""Breadth-first search over the latent space.

The cheap method. No PDDL, no Fast Downward, no lisp. It mines the distinct
latent deltas from the training transitions and searches over them.

The deltas carry no preconditions, so a delta that flips bit 7 applies in any
state, even one where the model would never have produced that transition.
That is the price of skipping PDDL, and it makes this method a lower bound
rather than a faithful reading of the learned schema. It earns its place by
exercising the encode, decode and scoring path with no lisp toolchain.
"""

import time
from collections import deque


def mine_deltas(pre, suc):
    """Collect the distinct XOR deltas seen in the training transitions.

    Sorted lexicographically so the search order stays fixed across runs
    (SPEC V15).
    """
    import numpy as np

    diffs = (np.asarray(suc) ^ np.asarray(pre)).astype(np.int8)

    # Frames that encode to the same latent give an all-zero delta. Keeping it
    # would hand the search a self-loop that can only waste expansions.
    diffs = diffs[diffs.any(axis=1)]
    if len(diffs) == 0:
        return diffs

    packed = np.ascontiguousarray(diffs)
    view = packed.view(np.dtype((np.void, packed.dtype.itemsize * packed.shape[1])))
    _, first = np.unique(view, return_index=True)
    distinct = diffs[np.sort(first)]

    return distinct[np.lexsort(distinct.T[::-1])]


def search(z_init, z_goal, deltas, time_budget_s=60.0):
    """Breadth-first search from z_init to z_goal.

    Returns (found, trace, wall_s). The trace holds the init state, every
    intermediate state and the goal state.
    """
    import numpy as np

    z_init = np.asarray(z_init, dtype=np.int8).reshape(-1)
    z_goal = np.asarray(z_goal, dtype=np.int8).reshape(-1)
    deltas = np.asarray(deltas, dtype=np.int8)

    start, goal = z_init.tobytes(), z_goal.tobytes()
    began = time.time()

    if start == goal:
        return True, np.stack([z_init]), 0.0

    queue = deque([z_init])
    parent = {start: None}

    while queue:
        if time.time() - began > time_budget_s:
            break

        state = queue.popleft()
        for delta in deltas:
            child = (state ^ delta).astype(np.int8)
            key = child.tobytes()
            if key in parent:
                continue
            parent[key] = state.tobytes()

            if key == goal:
                chain, node = [key], key
                while parent[node] is not None:
                    node = parent[node]
                    chain.append(node)
                chain.reverse()
                trace = np.stack([np.frombuffer(k, dtype=np.int8) for k in chain])
                return True, trace, time.time() - began

            queue.append(child)

    empty = np.zeros((0, len(z_init)), dtype=np.int8)
    return False, empty, time.time() - began


def _solve(z_init, z_goal, z_all, time_budget_s, out_dir, export=None, **_):
    pre, suc = export.transitions()
    deltas = mine_deltas(pre=pre, suc=suc)
    print(f"{len(deltas)} distinct deltas from {len(pre)} transitions")
    found, trace, wall = search(z_init, z_goal, deltas, time_budget_s)
    return found, trace, wall, {"n_deltas": int(len(deltas))}


def run(export_path, init_idx, goal_idx, out_dir, **kwargs):
    from tools.planner.common.harness import run_window
    return run_window(export_path, init_idx, goal_idx, out_dir,
                      solve=_solve, method="bfs", **kwargs)
