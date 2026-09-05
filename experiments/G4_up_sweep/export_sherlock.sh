#!/usr/bin/env bash
# G4 -- export every cell of the grid that finished, and record what each one
# reconstructed. Chained off run_sherlock.sh by --dependency, so it is not
# normally submitted by hand.
#
# The glob has to be evaluated HERE rather than in run_sherlock.sh, because the
# run directories do not exist until the training jobs have started.
#
# Export cannot run on a login node: TensorFlow fails to build its thread pool
# against the login node's process limit and aborts. See sh/export_model.sh.
#
# Two outputs:
#
#   eval/exports/<U..P..cat..>.npz   one planner export per cell
#   eval/exports/G4_train.csv        the reconstruction axis of the figure
#
# The CSV exists because the planner export carries no training loss, and the
# run directories stay on the cluster. Without it the figure has only one axis.
#
#SBATCH --job-name=fosae-G4-export
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=logs/G4-export.%j.out
#SBATCH --error=logs/G4-export.%j.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs eval/exports

shopt -s nullglob
# The grid sits on H14's npz, so H14's own four cells match this glob too and
# join the figure at no extra cost. That is deliberate.
DIRS=(out/video/vidvrd/*catH14-winnable*)
shopt -u nullglob

if (( ${#DIRS[@]} == 0 )); then
    echo "no G4 run directories under out/video/vidvrd/ -- every cell failed" >&2
    exit 1
fi

echo "exporting ${#DIRS[@]} run(s)"
rc=0
bash sh/export_model.sh "${DIRS[@]}" || rc=$?

echo
echo "=========================================="
echo "recording the reconstruction axis"
echo "=========================================="
python3 - "${DIRS[@]}" <<'PY'
"""Write eval/exports/G4_train.csv: one row per run directory.

Reads training_history.csv only, so it needs neither keras nor the weights.
The validation-loss column is found by pattern because its name differs by
domain, which is the same rule tools/rank_models.py uses.

Python 3.6 clean, standard library only.
"""
import csv
import os
import re
import sys

CELL = re.compile(r"_U(\d+)_A(\d+)_P(\d+)_")


def history(run_dir):
    path = os.path.join(run_dir, "training_history.csv")
    try:
        with open(path) as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return None
    if not rows:
        return None
    keys = [k for k in rows[0] if k]
    val_key = None
    for k in keys:
        if k.startswith("val_") and "loss" in k.lower():
            val_key = k
            break
    if val_key is None:
        for k in keys:
            if k.startswith("val_"):
                val_key = k
                break
    if val_key is None:
        return None
    vals = []
    for r in rows:
        try:
            vals.append(float(r[val_key]))
        except (TypeError, ValueError, KeyError):
            pass
    if not vals:
        return None
    return {"metric": val_key, "best_val": min(vals), "last_val": vals[-1],
            "epochs": len(rows), "moved": (max(vals) - min(vals)) > 1e-9}


out_path = os.path.join("eval", "exports", "G4_train.csv")
written = 0
with open(out_path, "w") as handle:
    writer = csv.writer(handle)
    writer.writerow(["run_dir", "U", "A", "P", "bits",
                     "metric", "best_val", "last_val", "epochs", "moved"])
    for run_dir in sys.argv[1:]:
        base = os.path.basename(run_dir.rstrip("/"))
        m = CELL.search(base)
        if not m:
            print("no U/A/P in %s, skipped" % base)
            continue
        u, a, p = (int(x) for x in m.groups())
        info = history(run_dir)
        if info is None:
            print("no usable training_history.csv in %s" % base)
            writer.writerow([base, u, a, p, u * p, "", "", "", 0, ""])
            written += 1
            continue
        writer.writerow([base, u, a, p, u * p, info["metric"],
                         "%.6g" % info["best_val"], "%.6g" % info["last_val"],
                         info["epochs"], info["moved"]])
        written += 1
        print("U%-3d P%-3d  %s best %.6g  over %d epoch(s)%s"
              % (u, p, info["metric"], info["best_val"], info["epochs"],
                 "" if info["moved"] else "   LOSS NEVER MOVED"))

print("wrote %s, %d row(s)" % (out_path, written))
PY

cat <<'NOTE'

Push the exports and the training summary together. The CSV is small and the
figure needs it:

    git add -f eval/exports/*catH14-winnable*.npz eval/exports/G4_train.csv
    git commit -m "G4 exports" && git push

Then, on the workstation:

    git pull && bash experiments/G4_up_sweep/score_local.sh
NOTE

exit "${rc}"
