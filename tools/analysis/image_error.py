#!/usr/bin/env python3
"""The reconstruction error of the image part and of the box part, apart.

WHY. A FOSAE video state holds, for each object, its image patch and then a
one-hot code of its box. The box code is 51 per cent of the loss at patch 8
and 3 per cent at patch 48 (`git show 56dac44:experiments/G5_patch_size/
README.md`). So a whole-vector MSE or BCE is not comparable across patch
sizes, and an adversary showed on 2026-10-05 that the overfit table invited
that comparison. The image part alone is comparable. Plan
`notes/plans/2026-10-04-overfit-viability.md`, Task 5.

    python3 tools/analysis/image_error.py RUN_DIR [RUN_DIR ...]

Each run is rebuilt from the npz its manifest names, so the states are the
ones the model saw. Needs the training venv, because it loads the model.
"""

import argparse
import json
import os
import sys

import numpy as np


def split_errors(x, y, image_dim):
    """`{image_MSE, box_MSE, all_MSE}` of reconstruction `y` of states `x`,
    shape (states, objects, features), image features first."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sq = np.square(x - y)
    return {"image_MSE": float(sq[..., :image_dim].mean()),
            "box_MSE": float(sq[..., image_dim:].mean())
            if sq.shape[-1] > image_dim else 0.0,
            "all_MSE": float(sq.mean())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import replot
    from model.loaders.labeled_objects import PICSIZE
    print("| run | states | image dims | image MSE | box MSE | all MSE |")
    print("|---|---|---|---|---|---|")
    for run in args.runs:
        states, _mode, _n = replot._load_video(run, "vidvrd", 10 ** 9)
        grid = (np.array(PICSIZE) // 5).astype(int)
        box_dim = replot.bboxes_to_onehot(np.zeros((1, 1, 4), dtype=int), grid[1],
                                          grid[0]).reshape(-1).size
        image_dim = states.shape[-1] - box_dim
        ae = replot.model.model.load(run)
        ae.load()
        got = split_errors(states, ae.autoencode(states, batch_size=1000),
                           image_dim)
        label = os.path.basename(os.path.normpath(run))
        print("| %s | %d | %d | %.5f | %.5f | %.5f |"
              % (label, len(states), image_dim, got["image_MSE"],
                 got["box_MSE"], got["all_MSE"]))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
