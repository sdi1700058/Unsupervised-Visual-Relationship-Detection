#!/bin/bash
#
# G5 export -- one planner export per patch arm, plus the liveness column.
#
# Runs itself, chained off the training jobs by run_sherlock.sh. Nothing here
# needs a visit.
#
#SBATCH --job-name=fosae-G5-export
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/G5-export.%j.out
#SBATCH --error=logs/G5-export.%j.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs eval/exports

source venv/bin/activate 2>/dev/null || source activate.sh

OUT_ROOT="out/video/vidvrd"
CSV="eval/exports/G5_train.csv"

echo "u,p,patch,best_val,epochs,distinct_codes,bits_set" > "${CSV}"

# A run directory that is absent is the whole arm missing. Exiting 0 here made
# SLURM record COMPLETED for an export job that exported nothing, which is the
# defect run_training.sh was changed for.
shopt -s nullglob
RUNS=("${OUT_ROOT}"/FirstOrderSAE_*catG5-winnable*)
shopt -u nullglob
if (( ${#RUNS[@]} == 0 )); then
    echo "no G5 run directories under ${OUT_ROOT} -- every arm failed" >&2
    exit 1
fi

rc=0
NEXPORT=0
NDEAD=0
for RUN in "${RUNS[@]}"; do
    [[ -d "${RUN}" ]] || continue
    NAME="$(basename "${RUN}")"
    # U and P read from the run directory, not assumed. The row used to carry
    # a hard-coded "40,10", so a sweep launched with U= or P= overridden wrote
    # a CSV that described a configuration it had not run.
    U="$(echo "${NAME}" | sed -n 's/.*_U\([0-9][0-9]*\)_A.*/\1/p')"
    P="$(echo "${NAME}" | sed -n 's/.*_P\([0-9][0-9]*\)_cat.*/\1/p')"
    PATCH="$(echo "${NAME}" | sed -n 's/.*[-_]p\([0-9][0-9]*\)_fps.*/\1/p')"
    if [[ -z "${U}" || -z "${P}" || -z "${PATCH}" ]]; then
        echo "SKIP ${NAME}: cannot read U, P and the patch size from the name" >&2
        rc=1
        continue
    fi
    OUT="eval/exports/${NAME#FirstOrderSAE_}.npz"

    echo "=== ${NAME}"
    if python3 tools/planner/export_latents.py "${RUN}" --out "${OUT}" \
            2>&1 | tail -4; then
        NEXPORT=$((NEXPORT + 1))
    else
        echo "FAILED ${NAME}"
        rc=1
        continue
    fi

    # The reconstruction axis, read from the run's own history rather than
    # recomputed. A missing history is reported, never treated as a zero.
    HIST="${RUN}/training_history.csv"
    BEST=""
    EPOCHS=""
    if [[ -f "${HIST}" ]]; then
        BEST="$(python3 - "${HIST}" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
key = next((k for k in (rows[0] if rows else {})
            if "val" in k.lower() and "loss" in k.lower()), None)
vals = []
for r in rows:
    try:
        vals.append(float(r[key]))
    except (TypeError, ValueError, KeyError):
        pass
print("%s,%d" % (("%.6f" % min(vals)) if vals else "", len(rows)))
PY
)"
        EPOCHS="${BEST#*,}"
        BEST="${BEST%%,*}"
    else
        echo "no training_history.csv in ${NAME}"
    fi

    # Liveness is the number of DISTINCT codes, not the number of set bits.
    # An encoder that emits one code for every frame is dead whatever that
    # code is, and the old test -- bits_set == 0 -- saw only the all-zero
    # case, so a constant all-ones latent counted as alive and was scored.
    # The failure path used to print an empty string, which the counter then
    # read as "not dead"; now an unreadable export is a failure of this job.
    if ! CODES="$(python3 - "${OUT}" <<'PY'
import sys
import numpy as np
z = np.load(sys.argv[1])
a = np.asarray(z["latents"])
flat = a.reshape(a.shape[0], -1)
print("%d %d" % (len(np.unique(flat, axis=0)), int(flat.sum())))
PY
)"; then
        echo "cannot read the latents of ${OUT}" >&2
        rc=1
        DISTINCT=""
        BITS=""
    else
        DISTINCT="${CODES%% *}"
        BITS="${CODES##* }"
        if (( DISTINCT <= 1 )); then NDEAD=$((NDEAD + 1)); fi
    fi
    echo "${U},${P},${PATCH},${BEST},${EPOCHS},${DISTINCT},${BITS}" >> "${CSV}"
done

echo
echo "=========================================="
echo "exported ${NEXPORT} arm(s); ${NDEAD} produced a dead latent"
echo "=========================================="
cat "${CSV}"

# A dead arm is an arm whose encoder emitted one code for every frame. It has
# no state for a planner to search, so it is not a slow arm or a poor arm, it
# is an absent measurement. Saying so here stops it being scored downstream.
if (( NDEAD > 0 )); then
    echo
    echo "WARNING: ${NDEAD} arm(s) are dead and must not be read as planning"
    echo "         results. tools/planner/liveness.py explains the test."
fi

cat <<'EOF'

Push the exports and the training summary together:

    git add -f eval/exports/*catG5-winnable*.npz eval/exports/G5_train.csv
    git commit -m "G5 exports" && git push

THERE IS NO experiments/G5_patch_size/score_local.sh. This line used to name
one and G5 has never had it, nor a README. Until one is written, score an arm
by hand the way G4 does, one export at a time:

    bash tools/planner/eval_plannability.sh eval/exports/<arm>.npz \
        --methods bfs,pddl --window 16 --budget 30 --name G5-p<patch>

and read best_val ONLY down a column, never across arms: the patch size
changes the size of the image half of the feature vector, so each arm's loss
is a differently weighted quantity. See the header of run_sherlock.sh.
EOF

exit "${rc}"
