#!/usr/bin/env python3
"""Rank every trained model on disk by how well it reconstructed.

tools/list_runs.py answers the same question through sacct, so it only sees
jobs still inside the accounting window and needs their logs. This walks the
output tree instead, which keeps working long after SLURM has forgotten the
job. Reads only training_history.csv, so it needs neither keras nor the
weights.

    python3 tools/rank_models.py
    python3 tools/rank_models.py --domain video --limit 20
    python3 tools/rank_models.py --plannable        # only ones worth planning on
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Below this the reconstruction is too poor for the latents to mean anything,
# so planning over them tells you nothing (SPEC C17).
PLANNABLE_VAL_LOSS = 0.5


def read_history(csv_path):
    """Return (best_val, last_val, last_train, epochs) from a history file."""
    try:
        with open(csv_path, newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return None
    if not rows:
        return None

    # Column names differ by domain, so find the validation loss by pattern.
    val_key = next((k for k in rows[0]
                    if k and k.startswith("val_") and "loss" in k.lower()), None)
    if val_key is None:
        val_key = next((k for k in rows[0] if k and k.startswith("val_")), None)
    if val_key is None:
        return None

    train_key = val_key[4:] if val_key[4:] in rows[0] else None

    def num(row, key):
        try:
            return float(row[key])
        except (TypeError, ValueError, KeyError):
            return None

    vals = [v for v in (num(r, val_key) for r in rows) if v is not None]
    if not vals:
        return None

    trains = ([v for v in (num(r, train_key) for r in rows) if v is not None]
              if train_key else [])

    return {
        "metric": val_key,
        "best_val": min(vals),
        "last_val": vals[-1],
        "last_train": trains[-1] if trains else None,
        "epochs": len(rows),
        # A loss that never moved means the run produced nothing, even if it
        # exited cleanly.
        "moved": (max(vals) - min(vals)) > 1e-9,
    }


def describe(model_dir):
    """Collect what we know about one model directory."""
    history = read_history(model_dir / "training_history.csv")
    if history is None:
        return None

    info = {"dir": model_dir, **history}
    info["has_weights"] = (model_dir / "net0.h5").exists()
    info["has_recon"] = (model_dir / "viz" / "recon_grid.png").exists()
    info["has_actions"] = (model_dir / "actions.csv").exists()

    manifest = model_dir / "loaded_videos.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text())
            info["npz"] = Path(data.get("npz_path", "")).name or None
            info["category"] = data.get("category_filter")
            info["mode"] = data.get("transition_mode")
        except (OSError, ValueError):
            pass
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="out", type=Path,
                    help="directory to walk. Default: out")
    ap.add_argument("--domain", help="only paths containing this string")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--plannable", action="store_true",
                    help=f"only models with val loss below {PLANNABLE_VAL_LOSS}")
    ap.add_argument("--weights-only", action="store_true",
                    help="skip models whose net0.h5 is gone")
    args = ap.parse_args(argv)

    if not args.root.is_dir():
        sys.exit(f"no such directory: {args.root}")

    models = []
    for history in args.root.rglob("training_history.csv"):
        if args.domain and args.domain not in str(history):
            continue
        info = describe(history.parent)
        if info:
            models.append(info)

    if not models:
        sys.exit(f"no training_history.csv found under {args.root}")

    total = len(models)
    if args.plannable:
        models = [m for m in models if m["best_val"] < PLANNABLE_VAL_LOSS
                  and m["moved"]]
    if args.weights_only:
        models = [m for m in models if m["has_weights"]]

    models.sort(key=lambda m: m["best_val"])
    shown = models[:args.limit]

    print(f"{total} models under {args.root}, showing {len(shown)}\n")
    header = f"{'best val':>12}  {'last val':>12}  {'epochs':>6}  w r a  path"
    print(header)
    print("-" * len(header))

    for m in shown:
        flags = "".join([
            "W" if m["has_weights"] else ".",
            "R" if m["has_recon"] else ".",
            "A" if m["has_actions"] else ".",
        ])
        note = "" if m["moved"] else "   <- loss never moved"
        print(f"{m['best_val']:12.6g}  {m['last_val']:12.6g}  "
              f"{m['epochs']:6d}  {flags[0]} {flags[1]} {flags[2]}  "
              f"{m['dir']}{note}")

    print("\nW = net0.h5 present, R = recon_grid.png present, "
          "A = actions.csv present")
    print(f"metric: {shown[0]['metric']}")

    ready = [m for m in models
             if m["has_weights"] and m["moved"]
             and m["best_val"] < PLANNABLE_VAL_LOSS]
    if ready:
        print(f"\n{len(ready)} model(s) look worth planning on. Best:")
        print(f"  python3 tools/planner/export_latents.py {ready[0]['dir']} "
              f"-o eval/exports/best.npz")
    else:
        print(f"\nNothing clears val loss < {PLANNABLE_VAL_LOSS} with weights "
              "on disk. Planning over these would not mean much.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
