#!/usr/bin/env python3
"""Drive the upstream latplan AMA3 pipeline.

This is the paper-faithful method. The PDDL comes from the author's own lisp
code, not from our reimplementation, so a result here carries more weight
than the pddl method does.

The chain, and where it departs from upstream ama3-planner.py:

    lisp/ama3-domain.bin      effect CSVs -> domain.pddl    (upstream)
    helper/ama3-problem.sh    init + goal -> problem.pddl   (upstream)
    fast-downward.py          both files  -> a plan         (see below)
    replay_plan()             plan        -> latent trace   (see below)

The two PDDL files come from the author's lisp, which is the part that
matters for fidelity. The last two steps do not:

- Upstream starts the planner through helper/fd-latest.sh, which shells out
  to a planner-scripts/ layout we do not have. We call Fast Downward with
  the same search configuration instead.
- Upstream turns the plan into a state trace with `arrival` and a second
  lisp binary. `arrival` hangs here with no output, and the step is
  redundant: Fast Downward guarantees the plan fits the domain, and we
  wrote the add and delete sets, so replaying them is exact.

The lisp binaries live under /home/panoslat/Dev/Thesis/FOSAE/latplan and
stay read-only (SPEC C14). install_roswell.sh builds them.
"""

import os
import subprocess
import time
from pathlib import Path


def _upstream_binary(name):
    from tools.planner.ama3.upstream_bridge import upstream_dir

    path = Path(upstream_dir("lisp")) / name
    if not path.exists():
        raise RuntimeError(
            f"{path} is missing. Run tools/planner/install_roswell.sh, or "
            f"build it with: cd {path.parent} && make")
    if not os.access(str(path), os.X_OK):
        raise RuntimeError(f"{path} is not executable")
    return str(path)


def _upstream_helper(name):
    from tools.planner.ama3.upstream_bridge import upstream_dir

    path = Path(upstream_dir("helper")) / name
    if not path.exists():
        raise RuntimeError(f"{path} is missing from the upstream checkout")
    return str(path)


def write_domain(export, out_dir):
    """Produce domain.pddl through the upstream lisp emitter.

    ama3-domain.bin takes the action list and the add and delete effects as
    three separate files, the shape the action autoencoder dumps. FOSAE's own
    dump_actions writes only pre|suc rows (model.py:1051), so we derive the
    three files here. One action per distinct (add, delete) pair, which is the
    same grouping upstream reaches by way of the action labels.
    """
    import numpy as np

    from tools.planner.pddl.planner import distinct_effects

    out_dir = Path(out_dir)
    pre, suc = export.transitions()
    effects = distinct_effects(pre, suc)
    if not effects:
        raise RuntimeError("no non-trivial transitions to build a domain from")

    n_bits = pre.shape[1]
    add = np.zeros((len(effects), n_bits), dtype=int)
    delete = np.zeros((len(effects), n_bits), dtype=int)
    for i, effect in enumerate(effects):
        add[i, effect["add"]] = 1
        delete[i, effect["del"]] = 1

    actions_csv = out_dir / "available_actions.csv"
    add_csv = out_dir / "action_add.csv"
    del_csv = out_dir / "action_del.csv"
    np.savetxt(str(actions_csv), np.arange(len(effects)), fmt="%d")
    np.savetxt(str(add_csv), add, fmt="%d")
    np.savetxt(str(del_csv), delete, fmt="%d")

    target = out_dir / "domain.pddl"
    binary = _upstream_binary("ama3-domain.bin")
    with target.open("w") as handle:
        subprocess.check_call(
            [binary, str(actions_csv), str(add_csv), str(del_csv)],
            stdout=handle)
    return target, effects


def write_problem(z_init, z_goal, out_dir):
    """Produce problem.pddl through the upstream shell wrapper."""
    import numpy as np

    out_dir = Path(out_dir)
    bits = np.concatenate([
        np.asarray(z_init).reshape(-1),
        np.asarray(z_goal).reshape(-1)]).astype(int)

    bits_file = out_dir / "init_goal.bits"
    np.savetxt(str(bits_file), [bits], fmt="%d")

    target = out_dir / "problem.pddl"
    subprocess.check_call(
        ["bash", _upstream_helper("ama3-problem.sh"),
         str(bits_file), str(target)])
    return target


def call_fast_downward(domain, problem, out_dir, time_budget_s=600):
    """Run Fast Downward on the PDDL the lisp emitter produced.

    Upstream wraps this in helper/fd-latest.sh, which shells out to
    planner-scripts/ relative to its own working directory and expects a
    checkout layout we do not have. What matters for fidelity is that the
    PDDL comes from the author's lisp, not which script starts the planner,
    so we call the binary ourselves with the same search configuration.
    """
    from tools.planner.pddl.planner import find_fast_downward

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
         "--search", "astar(lmcut())"],   # SPEC V15
        capture_output=True, text=True, cwd=str(out_dir))

    return plan_file, time.time() - began


def read_plan(plan_file):
    """Pull the action indices out of the plan the emitter's domain uses.

    The lisp emitter names actions a0, a1, ... in the order they appear in
    available_actions.csv, which we wrote as 0..n-1, so index k in the plan
    is effects[k].
    """
    indices = []
    for line in Path(plan_file).read_text().splitlines():
        token = line.strip().strip("()").strip()
        if token.startswith("a") and token[1:].split()[0].isdigit():
            indices.append(int(token[1:].split()[0]))
    return indices


def replay_plan(z_init, effects, indices):
    """Walk the plan to recover the state after each action.

    Upstream pipes the plan through `arrival` and a lisp reader to get this
    trace. That binary hangs here with no output, and it is not needed: Fast
    Downward already guarantees the plan is valid for the domain it solved,
    and we know each action's add and delete sets because we wrote them. So
    apply them directly, which is both exact and fast.
    """
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


def _solve(z_init, z_goal, z_all, time_budget_s, out_dir, export=None, **_):
    import numpy as np

    from tools.planner.ama3.upstream_bridge import ensure_upstream_on_path
    ensure_upstream_on_path()

    n_bits = z_all.shape[1]
    out_dir = Path(out_dir)

    domain, effects = write_domain(export, out_dir)
    problem = write_problem(z_init, z_goal, out_dir)
    print(f"{len(effects)} operators in the lisp-generated domain")

    plan_file, wall = call_fast_downward(domain, problem, out_dir,
                                         time_budget_s)
    if not plan_file.exists():
        return False, np.zeros((0, n_bits), dtype=np.int8), wall, {
            "n_operators": len(effects)}

    indices = read_plan(plan_file)
    trace = replay_plan(z_init, effects, indices)
    return True, trace, wall, {"n_operators": len(effects),
                               "plan_operators": indices}


def run(export_path, init_idx, goal_idx, out_dir, **kwargs):
    from tools.planner.common.harness import run_window
    return run_window(export_path, init_idx, goal_idx, out_dir,
                      solve=_solve, method="ama3", **kwargs)
