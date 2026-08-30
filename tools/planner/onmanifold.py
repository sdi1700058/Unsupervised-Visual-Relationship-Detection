#!/usr/bin/env python3
"""Plan without leaving the support, to test whether that is what breaks H14.

**The claim under test.** H14 measured that a trained FOSAE plans badly
(`mse_ratio` 3.16 against the oracle's 0.086) while its latent geometry is
nearly as good as the oracle's, and that the one thing that differs is
`decode_fallbacks`: 5.5 states per window are latents the encoder never
emitted, against the oracle's 0.0. `RELATED_WORK.md` O1 names the phenomenon —
leaving the support of the data.

That is a **correlation**. This turns it into an ablation.

The ordinary planner composes mined XOR deltas and may land anywhere in
`2^n_bits`. This one searches a graph whose **nodes are only latents the
encoder actually produced** and whose **edges are only transitions actually
observed**, so a plan cannot leave the support by construction and no decode
can fall back.

Same model, same windows, same scoring. The difference between the two is
attributable to leaving the support and to nothing else.

    python3 tools/planner/onmanifold.py eval/exports/H14-P10-150010.npz \\
        --window 8 --max-windows 14

Reading the result
------------------

===========================  =============================================
error drops a lot           Leaving the support is the cause. The
                            representation is usable and the SEARCH needs
                            constraining, which is a much cheaper fix than
                            retraining.
error barely moves          Leaving the support is a symptom. The latents
                            themselves do not carry a plannable structure,
                            and no search discipline rescues them.
===========================  =============================================

Both readings are written down before the run, and both are informative.

numpy and the standard library only, so it runs on Sherlock's Python 3.6.
"""

import argparse
import json
import os
import sys
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def observed_graph(latents):
    """Nodes = distinct observed latents. Edges = observed transitions.

    Returns `(index, adjacency)` where `index` maps a latent's bytes to a node
    id and `adjacency[i]` is the set of nodes reachable in one observed step.

    A self-loop is dropped: consecutive frames encoding to the same latent are
    a no-op, and keeping them would let a plan pad its length for free.
    """
    z = np.asarray(latents, dtype=np.int8)
    index, nodes = {}, []
    for row in z:
        k = row.tobytes()
        if k not in index:
            index[k] = len(nodes)
            nodes.append(k)

    adj = [set() for _ in nodes]
    for a, b in zip(z[:-1], z[1:]):
        i, j = index[a.tobytes()], index[b.tobytes()]
        if i != j:
            adj[i].add(j)
    return index, adj, nodes


def shortest_observed_path(latents, init_idx, goal_idx, max_length=None):
    """Frame-index path from init to goal through observed states only.

    Breadth-first over the observed transition graph, so the plan is the
    shortest sequence of *real* transitions connecting the two states. Returns
    a list of latents, or None when no observed route exists — which is itself
    a finding rather than a failure.
    """
    z = np.asarray(latents, dtype=np.int8)
    index, adj, nodes = observed_graph(z)
    start = index[z[init_idx].tobytes()]
    goal = index[z[goal_idx].tobytes()]

    if start == goal:
        return [z[init_idx]]

    parent = {start: None}
    queue = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if max_length is not None and depth >= max_length:
            continue
        for nxt in adj[node]:
            if nxt in parent:
                continue
            parent[nxt] = node
            if nxt == goal:
                chain, cur = [nxt], nxt
                while parent[cur] is not None:
                    cur = parent[cur]
                    chain.append(cur)
                chain.reverse()
                return [np.frombuffer(nodes[n], dtype=np.int8) for n in chain]
            queue.append((nxt, depth + 1))
    return None


def main(argv=None):
    from tools.planner.common.windows import (make_windows,
                                              extract_intermediate_states,
                                              linear_interp_bboxes)
    from tools.planner.common.metrics import score_window

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("export")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--max-windows", type=int, default=14)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)

    d = np.load(a.export)
    z = np.asarray(d["latents"], dtype=np.int8)
    gt = np.asarray(d["gt_boxes"], dtype=np.float64)
    dec = np.asarray(d["decoded_boxes"], dtype=np.float64)

    # Decode by exact lookup only. Every planned state is observed, so this
    # cannot fall back -- which is the entire point of the experiment.
    lookup = {}
    for i, row in enumerate(z):
        lookup.setdefault(row.tobytes(), i)

    windows = make_windows(len(z), a.window)[:a.max_windows]
    rows, solved = [], 0
    for w in windows:
        path = shortest_observed_path(z, w["init"], w["goal"])
        if path is None:
            rows.append({"init": w["init"], "solved": False})
            continue
        solved += 1
        mid, _ = extract_intermediate_states(np.stack(path),
                                             len(w["intermediate"]))
        pred = np.stack([dec[lookup[r.astype(np.int8).tobytes()]] for r in mid])
        base = linear_interp_bboxes(gt[w["init"]], gt[w["goal"]],
                                    len(w["intermediate"]))
        sc = score_window(pred, gt[w["intermediate"]], baseline_trace=base,
                          endpoints=(gt[w["init"]], gt[w["goal"]]))
        rows.append({"init": w["init"], "solved": True,
                     "plan_length": len(path) - 1,
                     "mse": sc["planner"]["mean_mse"],
                     "baseline": sc["baseline_linear"]["mean_mse"],
                     "ratio": sc["mse_ratio"]})

    scored = [r for r in rows if r.get("ratio") is not None]
    ratios = sorted(r["ratio"] for r in scored)
    mses = sorted(r["mse"] for r in scored if r["mse"] is not None)
    med = lambda v: v[len(v) // 2] if v else None

    print("on-manifold planning: %s" % os.path.basename(a.export))
    print("  windows            %d, solved %d" % (len(windows), solved))
    print("  scored             %d" % len(scored))
    print("  median mse_ratio   %s" % ("n/a" if not ratios else "%.3f" % med(ratios)))
    print("  median plan error  %s" % ("n/a" if not mses else "%.1f" % med(mses)))
    print("  decode fallbacks   0 by construction")

    if a.out_dir:
        if not os.path.isdir(a.out_dir):
            os.makedirs(a.out_dir)
        with open(os.path.join(a.out_dir, "onmanifold.json"), "w") as f:
            json.dump({"export": a.export, "window": a.window,
                       "median_ratio": med(ratios), "median_mse": med(mses),
                       "solved": solved, "windows": len(windows),
                       "rows": rows}, f, indent=2)
        print("\nwrote %s/onmanifold.json" % a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
