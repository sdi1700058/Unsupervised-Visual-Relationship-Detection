#!/usr/bin/env python3
"""Run a classical planner over FOSAE latents and score the result.

The evaluation task is frame interpolation. Give the planner frame i as the
initial state and frame i+window_frames-1 as the goal. The planner must
reconstruct the window_frames-2 frames between them. Score = bbox deviation
from the real frames.

Three planner methods share the same encode/decode/score path:

    ama3    upstream latplan AMA3 (Roswell + lisp), paper-faithful
    pddl    python STRIPS emitter + Fast Downward
    bfs     breadth-first search over the latent space, no PDDL

Takes a planner export, not a model directory, so it needs numpy alone.
Make the export once where keras and the model live:

    python3 model/latents/export_latents.py <model_dir> -o latents.npz

Output lands in eval/planner/<export>/<method>/win_<i>_<j>/.
"""

import argparse
import sys
from importlib import import_module
from pathlib import Path

METHODS = {
    "ama3": "planner.ama3.planner",
    "pddl": "planner.pddl.planner",
    "bfs":  "planner.bfs.planner",
}


def output_dir(export_path, method, init, goal):
    project_root = Path(__file__).resolve().parents[1]
    return (project_root / "eval" / "planner" / export_path.stem
            / method / f"win_{init}_{goal}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", type=Path,
                    help="planner export npz from export_latents.py")
    ap.add_argument("--method", choices=sorted(METHODS), required=True)
    ap.add_argument("--init", type=int, default=0,
                    help="index of the frame given as the initial state")
    ap.add_argument("--goal", type=int, default=-1,
                    help="index of the frame given as the goal state")
    ap.add_argument("--time-budget-s", type=int, default=60)
    ap.add_argument("--matching", choices=("hungarian", "fixed"),
                    default="hungarian",
                    help="how to pair decoded object slots with annotations")
    ap.add_argument("--length-mode", choices=("max", "exact", "free"),
                    default="max",
                    help="ask for the shortest plan within window_frames-1 "
                         "actions (max), exactly window_frames-1 (exact), "
                         "or an unbounded search (free)")
    ap.add_argument("--plan-only", action="store_true",
                    help="stop after the plan; skip scoring")
    ap.add_argument("--out-dir", type=Path,
                    help="defaults to eval/planner/<export>/<method>/win_i_j")
    args = ap.parse_args(argv)

    export_path = args.export.resolve()
    if not export_path.exists():
        sys.exit(f"no such export: {export_path}\n"
                 "Make one with model/latents/export_latents.py")

    out_dir = args.out_dir or output_dir(export_path, args.method,
                                         args.init, args.goal)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Direct invocation puts planner on sys.path, not the repo root.
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print(f"method   {args.method}")
    print(f"export   {export_path}")
    print(f"window   frame {args.init} -> frame {args.goal}")
    print(f"out      {out_dir}")

    planner = import_module(METHODS[args.method])
    try:
        planner.run(export_path=export_path,
                    init_idx=args.init,
                    goal_idx=args.goal,
                    out_dir=out_dir,
                    time_budget_s=args.time_budget_s,
                    matching=args.matching,
                    length_mode=args.length_mode,
                    plan_only=args.plan_only)
    except NotImplementedError as exc:
        print(f"method {args.method} is not ready: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
