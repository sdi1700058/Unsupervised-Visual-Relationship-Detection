#!/usr/bin/env python3
"""Score an export by whether its latent geometry can support planning.

The planner's intermediate states are almost never latents the model actually
produced. `Export.boxes_for` therefore decodes them by falling back to the
**Hamming-nearest observed latent**. That makes one property decisive: does
Hamming distance in the latent track distance in the world?

If it does, a fallback lands near the truth and the plan decodes sensibly. If
it does not, the decoded box is arbitrary and no amount of search quality
rescues it.

The point of having this as a separate tool is cost. It reads an export and
runs in seconds, with no planner, no search budget and no PDDL, so a sweep's
worth of models can be ranked before a single planning run is spent on them.
Reconstruction loss does not measure this and cannot: a code can reconstruct
every frame perfectly while ordering the frames arbitrarily in Hamming space.

    python3 tools/planner/latent_geometry.py eval/exports/*.npz
    python3 tools/planner/latent_geometry.py eval/exports/ --csv eval/geom.csv

Measured on ILSVRC2015_train_00005005, against planner error from the same
windows (see .claude/docs/EVAL.md 4.8):

    oracle       spearman +0.651   nn error  2.57 px   planner mse  13.67
    trained P10  spearman +0.356   nn error  6.10 px   planner mse 101.44
    trained P20  spearman +0.343   nn error 10.16 px   not measurable
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from tools.planner.common.metrics import latent_geometry  # noqa: E402


def score(path):
    import numpy as np

    # No allow_pickle: this reads only the numeric arrays, and an export that
    # needs pickle to load is one this tool should refuse rather than execute.
    data = np.load(path)
    if "latents" not in data.files or "gt_boxes" not in data.files:
        return None
    result = latent_geometry(data["latents"], data["gt_boxes"])
    result["export"] = os.path.basename(path)[:-4]
    z = np.asarray(data["latents"])
    result["n_bits"] = int(z.shape[1])
    result["distinct"] = len({row.tobytes() for row in z.astype("int8")})
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Rank planner exports by latent geometry, without planning.")
    ap.add_argument("paths", nargs="+",
                    help="export npz files, or directories holding them")
    ap.add_argument("--csv", help="also write the rows here")
    args = ap.parse_args(argv)

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.npz"))))
        else:
            files.append(p)
    if not files:
        print("no exports found", file=sys.stderr)
        return 1

    rows = []
    for f in files:
        try:
            r = score(f)
        except Exception as exc:                       # noqa: BLE001
            print(f"skip {os.path.basename(f)}: {exc}", file=sys.stderr)
            continue
        if r:
            rows.append(r)

    if not rows:
        print("no export carried both latents and gt_boxes", file=sys.stderr)
        return 1

    # Best first. A missing correlation sorts last; it means the code is
    # degenerate, which is worse than a weak correlation, not better.
    rows.sort(key=lambda r: (r["spearman"] is None,
                             -(r["spearman"] or 0.0)))

    print(f"{'export':52} {'bits':>6} {'distinct':>9} "
          f"{'spearman':>9} {'nn_err_px':>10}")
    for r in rows:
        rho = "    n/a" if r["spearman"] is None else f"{r['spearman']:+.3f}"
        print(f"{r['export'][:52]:52} {r['n_bits']:>6} {r['distinct']:>9} "
              f"{rho:>9} {r['nearest_box_error']:>10.2f}")

    print("\nHigher spearman and lower nn_err_px both mean the fallback decode "
          "lands\nnearer the truth. Neither is visible in reconstruction loss.")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w") as fh:
            fh.write("export,n_bits,distinct,n_frames,spearman,"
                     "nearest_box_error\n")
            for r in rows:
                rho = "" if r["spearman"] is None else f"{r['spearman']:.4f}"
                fh.write(f"{r['export']},{r['n_bits']},{r['distinct']},"
                         f"{r['n_frames']},{rho},"
                         f"{r['nearest_box_error']:.4f}\n")
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
