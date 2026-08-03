#!/usr/bin/env python3
"""Drive the upstream latplan AMA3 pipeline.

This is the paper-faithful method. The PDDL comes from the author's own lisp
code, not from our reimplementation, so a result here carries more weight
than the pddl method does.

The chain follows upstream ama3-planner.py:

    lisp/ama3-domain.bin      actions.csv  -> domain.pddl
    helper/ama3-problem.sh    init+goal    -> problem.pddl
    helper/fd-latest.sh       both files   -> a plan
    arrival                   plan         -> a state trace
    lisp/ama3-read-latent-state-traces.bin trace -> latents as CSV

Everything it calls lives under /home/panoslat/Dev/Thesis/FOSAE/latplan and
stays read-only (SPEC C14). install_roswell.sh builds the lisp binaries and
installs arrival.
"""

import os
import shutil
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


def write_domain(model_dir, out_dir):
    """Produce domain.pddl for the model, or reuse one already built.

    The domain depends only on the model, not on the window, so a batch run
    over many windows should build it once and link the rest.
    """
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    cached = model_dir / "domain.pddl"
    target = out_dir / "domain.pddl"

    if cached.exists():
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(cached.resolve())
        return target

    actions_csv = model_dir / "actions.csv"
    if not actions_csv.exists():
        raise RuntimeError(
            f"{actions_csv} is missing. Retrain with mode 'learn+dump' so "
            "that dump_actions writes it.")

    binary = _upstream_binary("ama3-domain.bin")
    with target.open("w") as handle:
        subprocess.check_call([binary, str(actions_csv)], stdout=handle)
    return target


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
    """Run the planner through the upstream wrapper. Fixed search (SPEC V15)."""
    began = time.time()
    subprocess.run(
        ["bash", _upstream_helper("fd-latest.sh"),
         "--search astar(lmcut())", str(problem), str(domain)],
        capture_output=True, text=True, cwd=str(out_dir))

    return Path(problem).with_suffix(".plan"), time.time() - began


def read_trace(plan_file, domain, problem, n_bits, out_dir):
    """Turn a plan into the latent states it passes through.

    arrival replays the plan against the PDDL and records each state. The
    lisp reader then converts that trace back into latent bit vectors.
    """
    import numpy as np

    out_dir = Path(out_dir)
    trace_file = out_dir / "problem.trace"
    csv_file = out_dir / "problem.csv"

    arrival = shutil.which("arrival")
    if arrival is None:
        raise RuntimeError(
            "arrival is not on PATH. Run tools/planner/install_roswell.sh")

    subprocess.check_call(
        [arrival, str(domain), str(problem), str(plan_file), str(trace_file)])

    with csv_file.open("w") as handle:
        subprocess.check_call(
            [_upstream_binary("ama3-read-latent-state-traces.bin"),
             str(trace_file), str(n_bits)],
            stdout=handle)

    latents = np.loadtxt(str(csv_file), dtype=int)
    return latents[None, :] if latents.ndim == 1 else latents


def _solve(z_init, z_goal, z_all, time_budget_s, out_dir, model_dir=None, **_):
    import numpy as np

    from tools.planner.ama3.upstream_bridge import ensure_upstream_on_path
    ensure_upstream_on_path()

    n_bits = z_all.shape[1]
    out_dir = Path(out_dir)

    domain = write_domain(model_dir, out_dir)
    problem = write_problem(z_init, z_goal, out_dir)

    plan_file, wall = call_fast_downward(domain, problem, out_dir,
                                         time_budget_s)
    if not plan_file.exists():
        return False, np.zeros((0, n_bits), dtype=np.int8), wall, {}

    trace = read_trace(plan_file, domain, problem, n_bits, out_dir)
    return True, trace, wall, {}


def run(model_dir, npz_path, init_idx, goal_idx, out_dir, **kwargs):
    from tools.planner.common.harness import run_window
    return run_window(model_dir, npz_path, init_idx, goal_idx, out_dir,
                      solve=_solve, method="ama3",
                      solve_kwargs={"model_dir": model_dir}, **kwargs)
