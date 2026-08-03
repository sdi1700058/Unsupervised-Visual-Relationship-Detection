#!/usr/bin/env python3
"""Emit propositional STRIPS from the learned latents and call Fast Downward.

The model writes actions.csv during training (latplan/model.py:1051): one row
per observed transition, holding the pre-state bits followed by the successor
bits. Distinct (add, delete) pairs from those rows become the action schema.

The encoding is flat propositional. Each latent bit becomes one predicate,
each distinct delta becomes one operator. Preconditions come from the bits
the delta touches, so an operator that sets bit 7 requires bit 7 to be off.
That keeps operators from firing where the model never saw them fire.

This method skips the lisp toolchain that AMA3 needs, at the cost of drifting
from the paper's own encoding.
"""

import os
import subprocess
import time
from pathlib import Path


def read_actions_csv(path):
    """Split actions.csv into the pre-state and successor blocks."""
    import numpy as np

    data = np.loadtxt(str(path), dtype=np.int8)
    if data.ndim == 1:
        data = data[None, :]

    width = data.shape[1]
    if width % 2:
        raise ValueError(
            f"actions.csv rows are {width} wide; expected an even 2*U*P")

    half = width // 2
    return data[:, :half], data[:, half:]


def distinct_effects(pre, suc):
    """Reduce the transitions to the distinct (add, delete) effect pairs.

    Two transitions that flip the same bits the same way are one operator.
    The no-op transition is dropped: it would give the planner a free action
    with no effect and inflate plan lengths.
    """
    import numpy as np

    diff = suc.astype(np.int16) - pre.astype(np.int16)
    packed = np.ascontiguousarray(
        np.concatenate([(diff > 0), (diff < 0)], axis=1).astype(np.int8))

    view = packed.view(np.dtype((np.void, packed.dtype.itemsize * packed.shape[1])))
    _, first, counts = np.unique(view, return_index=True, return_counts=True)

    order = np.argsort(first)
    rows, counts = packed[first[order]], counts[order]

    half = pre.shape[1]
    effects = []
    for row, count in zip(rows, counts):
        add = [int(i) for i in np.where(row[:half] > 0)[0]]
        delete = [int(i) for i in np.where(row[half:] > 0)[0]]
        if add or delete:
            effects.append({"add": add, "del": delete, "count": int(count)})
    return effects


def write_domain(effects, n_bits, path, name="fosae"):
    lines = [
        f"(define (domain {name})",
        "  (:requirements :strips :negative-preconditions)",
        "  (:predicates",
        "    " + " ".join(f"(bit_{i})" for i in range(n_bits)),
        "  )",
    ]

    for k, effect in enumerate(effects):
        pre = ([f"(not (bit_{i}))" for i in effect["add"]]
               + [f"(bit_{j})" for j in effect["del"]])
        post = ([f"(bit_{i})" for i in effect["add"]]
                + [f"(not (bit_{j}))" for j in effect["del"]])
        lines += [
            f"  (:action act_{k}",
            "    :parameters ()",
            f"    :precondition (and {' '.join(pre)})" if pre
            else "    :precondition ()",
            f"    :effect (and {' '.join(post)})",
            "  )",
        ]

    lines.append(")")
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def write_problem(z_init, z_goal, n_bits, path,
                  domain="fosae", name="interpolate"):
    """Write the problem file.

    The goal lists every bit, positive and negative. A partial goal would let
    the planner leave the untouched bits anywhere, which makes the decoded
    intermediate frames meaningless.
    """
    import numpy as np

    z_init = np.asarray(z_init).astype(int).reshape(-1)
    z_goal = np.asarray(z_goal).astype(int).reshape(-1)
    if z_init.size != n_bits or z_goal.size != n_bits:
        raise ValueError(
            f"latent sizes {z_init.size}/{z_goal.size} do not match {n_bits}")

    init = [f"(bit_{i})" for i in range(n_bits) if z_init[i]]
    goal = [f"(bit_{i})" if z_goal[i] else f"(not (bit_{i}))"
            for i in range(n_bits)]

    Path(path).write_text("\n".join([
        f"(define (problem {name})",
        f"  (:domain {domain})",
        "  (:objects)",
        "  (:init " + " ".join(init) + ")",
        "  (:goal (and " + " ".join(goal) + "))",
        ")",
    ]) + "\n")
    return Path(path)


def read_plan(path):
    """Pull the operator indices out of a sas_plan file."""
    indices = []
    for line in Path(path).read_text().splitlines():
        line = line.strip().strip("()").strip()
        if line.startswith("act_"):
            indices.append(int(line.split("_", 1)[1].split()[0]))
    return indices


def replay(z_init, effects, indices):
    """Walk the plan to recover the state after each operator."""
    import numpy as np

    state = np.asarray(z_init, dtype=np.int8).copy()
    trace = [state.copy()]
    for k in indices:
        for i in effects[k]["add"]:
            state[i] = 1
        for j in effects[k]["del"]:
            state[j] = 0
        trace.append(state.copy())
    return np.stack(trace)


def find_fast_downward():
    here = Path(__file__).resolve().parents[1]
    project_root = here.parent.parent

    candidates = [
        os.environ.get("FAST_DOWNWARD"),
        here / "fast-downward.py",
        project_root / "data/deps/fast-downward/fast-downward.py",
    ]
    scratch = os.environ.get("SCRATCH")
    if scratch:
        candidates.append(
            f"{scratch}/panos/sgg-thesis/deps/fast-downward/fast-downward.py")

    for candidate in candidates:
        if candidate and os.access(str(candidate), os.X_OK):
            return str(candidate)

    from shutil import which
    return which("fast-downward.py")


def call_fast_downward(domain, problem, out_dir, time_budget_s=60):
    """Run the planner. Fixed search config so reruns match (SPEC V15)."""
    binary = find_fast_downward()
    if binary is None:
        raise RuntimeError(
            "fast-downward.py not found. Run tools/planner/install_fd.sh")

    plan_file = Path(out_dir) / "sas_plan"
    began = time.time()
    subprocess.run(
        [binary,
         "--search-time-limit", str(time_budget_s),
         "--plan-file", str(plan_file),
         str(domain), str(problem),
         "--search", "astar(lmcut())"],
        capture_output=True, text=True, cwd=str(out_dir))

    return plan_file, time.time() - began


def _solve(z_init, z_goal, z_all, time_budget_s, out_dir, model_dir=None, **_):
    import numpy as np

    n_bits = z_all.shape[1]

    # Prefer the transitions the model itself dumped. Falling back to
    # consecutive encoded frames loses any all_pairs structure.
    actions_csv = Path(model_dir) / "actions.csv" if model_dir else None
    if actions_csv and actions_csv.exists():
        pre, suc = read_actions_csv(actions_csv)
        print(f"read {len(pre)} transitions from {actions_csv.name}")
    else:
        pre, suc = z_all[:-1], z_all[1:]
        print(f"no actions.csv; using {len(pre)} consecutive frame pairs")

    effects = distinct_effects(pre, suc)
    print(f"{len(effects)} distinct operators")

    out_dir = Path(out_dir)
    domain = write_domain(effects, n_bits, out_dir / "domain.pddl")
    problem = write_problem(z_init, z_goal, n_bits, out_dir / "problem.pddl")

    plan_file, wall = call_fast_downward(domain, problem, out_dir,
                                         time_budget_s)
    if not plan_file.exists():
        return False, np.zeros((0, n_bits), dtype=np.int8), wall, {
            "n_operators": len(effects)}

    indices = read_plan(plan_file)
    trace = replay(z_init, effects, indices)
    return True, trace, wall, {"n_operators": len(effects),
                               "plan_operators": indices}


def run(model_dir, npz_path, init_idx, goal_idx, out_dir, **kwargs):
    from tools.planner.common.harness import run_window
    return run_window(model_dir, npz_path, init_idx, goal_idx, out_dir,
                      solve=_solve, method="pddl",
                      solve_kwargs={"model_dir": model_dir}, **kwargs)
