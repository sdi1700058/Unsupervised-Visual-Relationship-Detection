#!/usr/bin/env python3
"""Run a classical planner over FOSAE latents and score the result.

The evaluation task is frame interpolation. Give the planner frame i as the
initial state and frame i+k-1 as the goal. The planner must reconstruct the
k-2 frames between them. Score = bbox deviation from the real frames.

Three planner methods share the same encode/decode/score path:

    ama3    upstream latplan AMA3 (Roswell + lisp), paper-faithful
    pddl    python STRIPS emitter + Fast Downward
    bfs     breadth-first search over the latent space, no PDDL

Output lands in eval/planner/<model>/<video>/<method>/win_<i>_<j>/.
"""

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

METHODS = {
    "ama3": "tools.planner.ama3.planner",
    "pddl": "tools.planner.pddl.planner",
    "bfs":  "tools.planner.bfs.planner",
}


def resolve_npz(model_dir, cli_npz):
    """Find the npz the model trained on, from the CLI or the run manifest."""
    if cli_npz is not None:
        return Path(cli_npz)

    manifest = model_dir / "loaded_videos.json"
    if not manifest.exists():
        sys.exit(f"no --npz-path and no manifest at {manifest}")

    path = json.loads(manifest.read_text()).get("npz_path")
    if not path:
        sys.exit(f"manifest has no 'npz_path' key: {manifest}")
    return Path(path)


def output_dir(model_dir, npz_path, method, init, goal):
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / "eval" / "planner" / model_dir.name / npz_path.stem
            / method / f"win_{init}_{goal}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path,
                    help="trained FirstOrderSAE directory (holds net0.h5)")
    ap.add_argument("--method", choices=sorted(METHODS), required=True)
    ap.add_argument("--npz-path", type=Path,
                    help="defaults to loaded_videos.json['npz_path']")
    ap.add_argument("--init", type=int, default=0,
                    help="index of the frame given as the initial state")
    ap.add_argument("--goal", type=int, default=-1,
                    help="index of the frame given as the goal state")
    ap.add_argument("--time-budget-s", type=int, default=60)
    ap.add_argument("--matching", choices=("hungarian", "fixed"),
                    default="hungarian",
                    help="how to pair decoded object slots with annotations")
    ap.add_argument("--plan-only", action="store_true",
                    help="stop after the plan; skip decode and scoring")
    args = ap.parse_args(argv)

    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        sys.exit(f"not a directory: {model_dir}")

    npz_path = resolve_npz(model_dir, args.npz_path).resolve()
    out_dir = output_dir(model_dir, npz_path, args.method, args.init, args.goal)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Direct invocation puts tools/planner on sys.path, not the repo root,
    # so the tools.planner.* imports below would fail without this.
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print(f"method   {args.method}")
    print(f"model    {model_dir}")
    print(f"npz      {npz_path}")
    print(f"window   frame {args.init} -> frame {args.goal}")
    print(f"out      {out_dir}")

    planner = import_module(METHODS[args.method])
    try:
        planner.run(model_dir=model_dir,
                    npz_path=npz_path,
                    init_idx=args.init,
                    goal_idx=args.goal,
                    out_dir=out_dir,
                    time_budget_s=args.time_budget_s,
                    matching=args.matching,
                    plan_only=args.plan_only)
    except NotImplementedError as exc:
        print(f"method {args.method} is not ready: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
