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

echo "u,p,patch,best_val,epochs,bits_set" > "${CSV}"

NEXPORT=0
NDEAD=0
for RUN in "${OUT_ROOT}"/FirstOrderSAE_*catG5-winnable*; do
    [[ -d "${RUN}" ]] || continue
    NAME="$(basename "${RUN}")"
    PATCH="$(echo "${NAME}" | grep -oE 'p[0-9]+_fps' | grep -oE '[0-9]+' || echo "")"
    OUT="eval/exports/${NAME#FirstOrderSAE_}.npz"

    echo "=== ${NAME}"
    if python3 tools/planner/export_latents.py "${RUN}" --out "${OUT}" \
            2>&1 | tail -4; then
        NEXPORT=$((NEXPORT + 1))
    else
        echo "FAILED ${NAME}"
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

    BITS="$(python3 - "${OUT}" <<'PY'
import sys
import numpy as np
try:
    z = np.load(sys.argv[1])
    a = np.asarray(z["latents"])
    print(int(a.reshape(a.shape[0], -1).sum()))
except Exception:
    print("")
PY
)"
    [[ "${BITS}" == "0" ]] && NDEAD=$((NDEAD + 1))
    echo "40,10,${PATCH},${BEST},${EPOCHS},${BITS}" >> "${CSV}"
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

Then, on the workstation:

    git pull && bash experiments/G5_patch_size/score_local.sh
EOF
