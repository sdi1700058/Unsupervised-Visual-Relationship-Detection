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

IT IS A SCREEN, NOT A RANKING. That was tested rather than assumed. Across six
oracle variants differing only in bin count, spearman correlates +0.934 with
planner error in the WRONG direction: coarse bins make the code a coarse
position, which raises the correlation and raises the quantisation floor
together. Within one encoding family the floor predicts error at +0.985 and
this metric does not.

    bins     floor   spearman   planner mse
    20x14    41.06    +0.740        58.85
    30x20    21.43    +0.727        34.22
    45x30    11.01    +0.682        28.24
    60x40     5.40    +0.651        13.67
    90x60     3.08    +0.652        13.85
    120x80    1.53    +0.615         7.28

So use it to catch a code that does not encode position at all, which is what
the trained models do at +0.34. Do not use it to choose between two codes of
the same kind, where a higher score usually just means coarser bins.
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
    result.update({"temporal_" + k: v
                   for k, v in temporal_fidelity(data["latents"]).items()
                   if k in ("steps_per_frame", "n_distinct")})
    result["export"] = os.path.splitext(os.path.basename(path))[0]
    z = np.asarray(data["latents"])
    result["n_bits"] = int(z.shape[1])
    result["distinct"] = len({row.tobytes() for row in z.astype("int8")})
    return result



def temporal_fidelity(latents, ks=(2, 4, 7, 10), samples=12, max_states=4000):
    """Does latent path length track real time? `SPEC.md` V35.

    **NOT VALIDATED AS A PREDICTOR OF PLANNER ERROR. Read this before using
    it.** It was shipped claiming to predict planner error where `spearman`
    does not. Tested across 8 clips of one model, spanning fidelity 0.00 to
    1.00, it correlates **+0.238** with `mse_ratio` — the **wrong sign** — and
    `spearman` does better at **-0.452**. The clip with perfect fidelity 1.00
    had the *worst* ratio of the eight, 265.

    The n=3 evidence that looked convincing compared **different models on one
    clip**; the n=8 test compares **one model across clips**. Whatever it
    separates, it is not clip-level plannability, and the between-model claim
    now rests on three points and should not be leaned on either.

    What it does measure, and this part holds: **whether latent path length
    tracks real time.** It tracks `plan_length` almost exactly (fidelity 1.00
    gives 6-step plans, 0.00 gives 0-step plans), which is close to true by
    construction. Plan length simply does not determine interpolation error.

    Use it to describe a code, not to rank one. This is the second screen in
    this project to be narrowed after shipping; see V26 for the first.

    Measured on identical frames, median observed-graph steps between frames
    *k* apart::

        k              2     4     7    10     steps per frame
        oracle         2     4     7    10          1.00
        trained P10    2     4     6     9          0.86
        trained P5     1     2     4     4          0.57

    against planner `mse_ratio` 0.086, 3.16 and 3.20 — which is the n=3
    comparison that did not survive n=8. Across the 88 clips of H14's own
    training set the median fidelity is **0.35**, so clip `00150010`, on which
    the H14 headline was measured, is the model's **best case** and not a
    typical one.

    Needs no planner, no ground truth and no model — only the export's own
    latents. That makes it cheap, which is not the same as making it
    predictive.

    `steps_per_frame` is the **mean of the per-k ratios**, not the ratio at any
    single k. On H14's P10 arm that gives 0.94 where the ratio at k=7 alone is
    0.86; both describe the same measurement and the earlier report log quotes
    the latter. `per_k` is the unambiguous form and is what to quote.

    Returns `steps_per_frame` (None for a dead latent, which has no distance to
    measure rather than zero distance) and `per_k`, the median step count per
    separation.
    """
    import numpy as np

    from tools.planner.onmanifold import shortest_observed_path

    z = np.asarray(latents, dtype=np.int8)
    if len(z) > max_states:
        # Bound the BFS. Sub-sampling frames would change the transitions, so
        # the prefix is taken whole instead.
        z = z[:max_states]

    # A code with one distinct state has no distance to measure. That is a
    # DEAD latent, not a maximally compressed one, and the two must not read
    # alike: compression is a property of a working code (SPEC V29).
    n_distinct = len({row.tobytes() for row in z})
    if n_distinct < 2:
        return {"steps_per_frame": None, "per_k": {},
                "n_states": int(len(z)), "n_distinct": n_distinct,
                "note": "dead latent: %d distinct state(s)" % n_distinct}

    per_k, ratios = {}, []
    for k in ks:
        if len(z) <= k:
            continue
        lens = []
        stride = max(1, (len(z) - k) // samples)
        for i in range(0, len(z) - k, stride):
            path = shortest_observed_path(z, i, i + k)
            if path is not None:
                lens.append(len(path) - 1)
        if not lens:
            continue
        per_k[k] = float(np.median(lens))
        if k:
            ratios.append(per_k[k] / float(k))

    return {
        # None, not 0.0: a dead latent has no path at all, which is undefined
        # rather than "zero steps" (SPEC V29).
        "steps_per_frame": float(np.mean(ratios)) if ratios else None,
        "per_k": per_k,
        "n_states": int(len(z)),
        "n_distinct": n_distinct,
    }

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

    print(f"{'export':46} {'bits':>6} {'distinct':>9} "
          f"{'spearman':>9} {'nn_err_px':>10} {'time_fid':>9}")
    for r in rows:
        rho = "    n/a" if r["spearman"] is None else f"{r['spearman']:+.3f}"
        tf = r.get("temporal_steps_per_frame")
        tfs = "n/a" if tf is None else f"{tf:.2f}"
        print(f"{r['export'][:46]:46} {r['n_bits']:>6} {r['distinct']:>9} "
              f"{rho:>9} {r['nearest_box_error']:>10.2f} {tfs:>9}")

    print("\ntime_fid is latent steps per frame of real time (SPEC V35). It "
          "describes a code;\nit does NOT predict planner error. Tested over "
          "8 clips it correlates +0.238 with\nmse_ratio, the wrong sign, "
          "where spearman manages -0.452. Do not rank by it.\n"
          "n/a means a dead latent.\n")
    print("A screen, not a ranking. Below roughly +0.5 the code does not "
          "order frames the\nway the world does, so the fallback decode is "
          "arbitrary — which is what the\ntrained models do at +0.34, and it "
          "is invisible to reconstruction loss.\n\nAbove that, do NOT rank by "
          "this. Across oracle variants a higher score just\nmeans coarser "
          "bins and tracks planner error the wrong way; use the\n"
          "quantisation floor printed by tools/planner/oracle.py instead.")

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
