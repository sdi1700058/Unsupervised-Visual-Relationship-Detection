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

# Why a search reports what it reports. The distinction is load-bearing: three
# of these are claims about the REPRESENTATION and two are claims about our own
# compute, and before 2026-08-30 they were all returned as the same bare
# `False`. `make_report.py` renders that as "the action schema does not connect
# the two frames at all", so a clock running out was being printed as a
# negative result about the thesis.
#
#   solved              a plan was found
#   exhausted           the frontier emptied; no plan exists within the bound
#   endpoints_identical the two frames encode the same latent; the empty plan
#                       is valid but this is not a planned window
#   timeout             the wall-clock budget expired
#   node_cap            the node ceiling was hit
OUTCOMES = ("solved", "exhausted", "endpoints_identical", "timeout",
            "node_cap")


def _outcome(stats, value):
    if stats is not None:
        stats["outcome"] = value


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


def search(z_init, z_goal, deltas, time_budget_s=60.0, exact_length=None,
           max_length=None, max_nodes=1000000, stats=None):
    """Breadth-first search from z_init to z_goal.

    Returns (found, trace, wall_s). The trace holds the init state, every
    intermediate state and the goal state.

    `exact_length` asks for a plan of exactly that many actions rather than
    the shortest one. The interpolation task appears to need this:
    reconstructing the k-2 frames between frame i and frame i+k-1 looks like a
    trajectory of exactly k-1 steps, and scoring a one-action plan as a
    three-frame trajectory measures nothing.

    `max_length` asks for the shortest plan that fits inside k-1 actions, and
    it is the right question for a trained model. `exact_length` assumes the
    latent moves once per frame. A trained encoder does not oblige: it maps
    runs of consecutive frames to one code, so a k-frame window holds fewer
    than k-1 real transitions and no k-1-step plan exists except by padding,
    which the search cannot afford to find. Measured on the first two trained
    exports, four of seven steps in a window changed nothing at all.

    Fixed-depth search cannot key `visited` on the state alone, because a
    state reachable at depth 2 may also be needed at depth 3 on the way to a
    longer trajectory. There the key is (state, depth). Depth-capped search
    keys on the state, which is what keeps it affordable.
    """
    import numpy as np

    if exact_length is not None and max_length is not None:
        raise ValueError("pass exact_length or max_length, not both")

    z_init = np.asarray(z_init, dtype=np.int8).reshape(-1)
    z_goal = np.asarray(z_goal, dtype=np.int8).reshape(-1)
    deltas = np.asarray(deltas, dtype=np.int8)

    start, goal = z_init.tobytes(), z_goal.tobytes()
    began = time.time()

    if exact_length is None:
        if max_length is not None and max_length < 0:
            raise ValueError(f"max_length must be >= 0; got {max_length}")
        if start == goal:
            # The empty plan IS the shortest plan within k-1 actions, so this
            # is correct -- but it must not be counted as a solved window.
            # 108 of 385 recorded rows were `reachability=True` with
            # `plan_length=0`, and one run reported "6 of 12 solved" where the
            # true count of planned windows was zero.
            _outcome(stats, "endpoints_identical")
            return True, np.stack([z_init]), 0.0
        if max_length == 0:
            _outcome(stats, "exhausted")
            return False, np.zeros((0, len(z_init)), dtype=np.int8), 0.0

        queue = deque([(z_init, 0)])
        parent = {start: None}
        outcome = "exhausted"

        while queue:
            if time.time() - began > time_budget_s:
                outcome = "timeout"
                break
            if len(parent) >= max_nodes:
                outcome = "node_cap"
                break

            state, depth = queue.popleft()
            if max_length is not None and depth >= max_length:
                continue

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
                    trace = np.stack([np.frombuffer(k, dtype=np.int8)
                                      for k in chain])
                    _outcome(stats, "solved")
                    return True, trace, time.time() - began

                queue.append((child, depth + 1))

        _outcome(stats, outcome)
        empty = np.zeros((0, len(z_init)), dtype=np.int8)
        return False, empty, time.time() - began

    if exact_length < 0:
        raise ValueError(f"exact_length must be >= 0; got {exact_length}")
    if exact_length == 0:
        # A zero-step plan is only valid when the two ends already agree.
        if start == goal:
            _outcome(stats, "endpoints_identical")
            return True, np.stack([z_init]), 0.0
        _outcome(stats, "exhausted")
        return False, np.zeros((0, len(z_init)), dtype=np.int8), 0.0

    queue = deque([(z_init, 0)])
    parent = {(start, 0): None}
    outcome = "exhausted"

    while queue:
        if time.time() - began > time_budget_s:
            outcome = "timeout"
            break
        if len(parent) >= max_nodes:
            outcome = "node_cap"
            break

        state, depth = queue.popleft()
        if depth >= exact_length:
            continue

        for delta in deltas:
            child = (state ^ delta).astype(np.int8)
            key = (child.tobytes(), depth + 1)
            if key in parent:
                continue
            parent[key] = (state.tobytes(), depth)

            if key[0] == goal and depth + 1 == exact_length:
                chain, node = [key], key
                while parent[node] is not None:
                    node = parent[node]
                    chain.append(node)
                chain.reverse()
                trace = np.stack([np.frombuffer(k[0], dtype=np.int8)
                                  for k in chain])
                _outcome(stats, "solved")
                return True, trace, time.time() - began

            queue.append((child, depth + 1))

    _outcome(stats, outcome)
    empty = np.zeros((0, len(z_init)), dtype=np.int8)
    return False, empty, time.time() - began


def _solve(z_init, z_goal, z_all, time_budget_s, out_dir, export=None,
           plan_length=None, length_mode="max", **_):
    pre, suc = export.transitions()
    deltas = mine_deltas(pre=pre, suc=suc)

    exact = plan_length if (length_mode == "exact" and plan_length) else None
    cap = plan_length if (length_mode == "max" and plan_length) else None
    if length_mode == "exact":
        how = f"; searching at exactly {exact} steps"
    elif length_mode == "max":
        how = f"; searching within {cap} steps"
    else:
        how = "; unbounded search"
    print(f"{len(deltas)} distinct deltas from {len(pre)} transitions" + how)

    found, trace, wall = search(z_init, z_goal, deltas, time_budget_s,
                                exact_length=exact, max_length=cap)
    return found, trace, wall, {"n_deltas": int(len(deltas))}


def run(export_path, init_idx, goal_idx, out_dir, **kwargs):
    from tools.planner.common.harness import run_window
    return run_window(export_path, init_idx, goal_idx, out_dir,
                      solve=_solve, method="bfs", **kwargs)
