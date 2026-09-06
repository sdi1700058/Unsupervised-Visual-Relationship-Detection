#!/bin/bash
#
# G5 -- patch size, held against everything else.
#
#     mkdir -p logs && sbatch experiments/G5_patch_size/run_sherlock.sh
#
# THE HYPOTHESIS, in one sentence.
# The 8x8 patch starves the image half of the loss, so a larger patch
# reconstructs the frame better, and the question is whether it also plans
# better or whether the two come apart.
#
# WHY THIS ARM EXISTS.
# Every bake in this project uses --patch-size 8. The feature vector is
# [patch_size^2 * 3 | x1,y1,x2,y2 one-hot], and the box block is fixed at 200
# dimensions, so the patch decides the balance:
#
#     patch 8    192 dims    box is 51% of the loss
#     patch 16   768 dims    box is 21%
#     patch 32   3072 dims   box is 6%
#     patch 48   6912 dims   box is 3%
#
# At patch 8 the model spends half its capacity on four numbers per object.
# The author reports that the reconstructions are unrecognisable, which is
# consistent with that split, and the overfit runs that reconstructed
# acceptably used patch 48 and 64. Nothing has ever measured the two against
# each other on the same data.
#
# WHAT EACH OUTCOME MEANS, decided before the run.
#   Reconstruction improves AND planning improves -> patch 8 was a mistake and
#       every earlier planner number was taken on a starved model. Rebake.
#   Reconstruction improves AND planning does not -> reconstruction loss does
#       not select for plannability, which is EVAL.md section 5.7 measured
#       directly rather than argued. A publishable negative.
#   Neither improves -> the patch is not the limit, and the limit is upstream
#       in the data. Look at the bake, not the model.
#   A larger patch fails to train at all -> record the memory it needed and
#       raise the request. One failure is a cost, not a verdict.
#
# EXPECTED RUNTIME. Four bakes, then four training jobs of about 40 minutes
# each at 3000 epochs, run in parallel. Under two hours wall clock.
#
#SBATCH --job-name=fosae-G5
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/G5.%j.out
#SBATCH --error=logs/G5.%j.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs

source venv/bin/activate 2>/dev/null || source activate.sh
source sh/sweep_lib.sh

FPS="${FPS:-30}"
NPZ="data/npz/video/vidvrd/overfit"

# The same 88 screened clips G4 and H14 used, inlined for the same reason:
# eval/ is gitignored, so the list never reaches the cluster as a file.
IDS="$(echo "\
    ILSVRC2015_train_00150010,ILSVRC2015_train_00040018,ILSVRC2015_train_00773000,ILSVRC2015_train_00069006,
    ILSVRC2015_train_00194008,ILSVRC2015_train_00040001,ILSVRC2015_train_00040005,ILSVRC2015_train_00150024,
    ILSVRC2015_train_00040029,ILSVRC2015_train_00265048,ILSVRC2015_train_00058003,ILSVRC2015_train_00415006,
    ILSVRC2015_train_00058001,ILSVRC2015_train_01081000,ILSVRC2015_train_00040009,ILSVRC2015_train_00308005,
    ILSVRC2015_val_00036008,ILSVRC2015_train_00548000,ILSVRC2015_train_00119025,ILSVRC2015_train_00466000,
    ILSVRC2015_train_00040020,ILSVRC2015_train_00218002,ILSVRC2015_train_00071019,ILSVRC2015_train_00127000,
    ILSVRC2015_val_00028003,ILSVRC2015_val_00159002,ILSVRC2015_train_00772005,ILSVRC2015_train_00265004,
    ILSVRC2015_train_00797000,ILSVRC2015_train_00272001,ILSVRC2015_train_00040022,ILSVRC2015_train_00415008,
    ILSVRC2015_train_00118005,ILSVRC2015_train_00804001,ILSVRC2015_val_00015001,ILSVRC2015_train_00016000,
    ILSVRC2015_train_00040025,ILSVRC2015_val_00037004,ILSVRC2015_train_00290020,ILSVRC2015_train_00068002,
    ILSVRC2015_train_00077001,ILSVRC2015_train_00265008,ILSVRC2015_train_00119014,ILSVRC2015_train_00411000,
    ILSVRC2015_train_00312007,ILSVRC2015_train_00987000,ILSVRC2015_train_00211004,ILSVRC2015_train_00185001,
    ILSVRC2015_train_00119045,ILSVRC2015_train_00010009,ILSVRC2015_train_00100002,ILSVRC2015_train_00010006,
    ILSVRC2015_train_00033006,ILSVRC2015_train_00071012,ILSVRC2015_train_00025022,ILSVRC2015_train_00057003,
    ILSVRC2015_train_00300009,ILSVRC2015_train_00149006,ILSVRC2015_train_00010012,ILSVRC2015_train_00897007,
    ILSVRC2015_train_00119040,ILSVRC2015_train_01020000,ILSVRC2015_train_00234013,ILSVRC2015_train_00010024,
    ILSVRC2015_train_00165000,ILSVRC2015_train_00181011,ILSVRC2015_train_00308009,ILSVRC2015_train_00375001,
    ILSVRC2015_train_01081001,ILSVRC2015_train_00119037,ILSVRC2015_val_00026002,ILSVRC2015_train_00065002,
    ILSVRC2015_train_00535000,ILSVRC2015_train_00165011,ILSVRC2015_train_00324000,ILSVRC2015_train_00234021,
    ILSVRC2015_train_00211000,ILSVRC2015_val_00035008,ILSVRC2015_train_00040031,ILSVRC2015_train_00253030,
    ILSVRC2015_train_00415004,ILSVRC2015_train_00010010,ILSVRC2015_train_00897009,ILSVRC2015_train_00729000,
    ILSVRC2015_val_00081000,ILSVRC2015_train_01052000,ILSVRC2015_train_00574002,ILSVRC2015_train_00962007" | tr -d ' \n')"

NCLIPS="$(echo "${IDS}" | tr ',' '\n' | grep -c .)"

# Held fixed at the configuration measured to be alive. G4 established that
# U40 P10 encodes 513,101 bits over these clips, while five other shapes
# collapsed to a constant latent. Sweeping the patch on a dead shape would
# measure the collapse.
SWEEP_DEFAULTS=(EPOCH=3000 LR=0.001 BATCH=1000
                PREENC_LAYERS=2 PREENC_DIM=1000
                MAX_TEMPERATURE=1.0 TRANSITION_MODE=sequential
                U=40 A=2 P=10)

PATCH_LIST="${PATCH_LIST:-8 16 32 48}"

# Memory scales with the square of the patch, so a single figure for every arm
# would either starve patch 48 or waste the budget on patch 8. These are
# requests, not measurements; seff after the run gives the real numbers and
# the next revision should use them.
mem_for () {
    case "$1" in
        8)  echo "16G" ;;
        16) echo "24G" ;;
        32) echo "48G" ;;
        *)  echo "80G" ;;
    esac
}

section "G5  bake -- one npz per patch size"

NBAKED=0
for PATCH in ${PATCH_LIST}; do
    STEM="G5-winnable${NCLIPS}-${FPS}fps-mo3-nofill-p${PATCH}"
    if [[ -f "${NPZ}/${STEM}.npz" ]]; then
        echo "have  ${STEM}"
        continue
    fi
    echo "bake  ${STEM}  (${NCLIPS} clips, patch ${PATCH}, no fill)"
    BAKE_LOG="logs/G5-bake-p${PATCH}.$$.log"
    python3 setup-dataset.py video_vidvrd all \
        --video-id "${IDS}" --fps "${FPS}" \
        --max-objects 3 --patch-size "${PATCH}" --out-name "${STEM}" \
        2>&1 | tee "${BAKE_LOG}"

    # A clip whose frames are missing is skipped silently by the loader, so a
    # partial extraction would shrink one arm while its name still claims 88.
    # An arm baked on fewer clips than its neighbours measures the data volume.
    LOADED="$(grep -oE '[0-9]+ videos? loaded' "${BAKE_LOG}" | tail -1 \
              | grep -oE '^[0-9]+' || true)"
    if [[ -n "${LOADED}" && "${LOADED}" != "${NCLIPS}" ]]; then
        echo "FATAL: patch ${PATCH} baked ${LOADED} clips, expected ${NCLIPS}." >&2
        echo "       Frames are missing. See ${BAKE_LOG}." >&2
        exit 3
    fi
    NBAKED=$((NBAKED + 1))
done

section "G5  train -- one arm per patch size, everything else held"

TIME_ALL="${TIME_ALL:-6:00:00}"
for PATCH in ${PATCH_LIST}; do
    STEM="G5-winnable${NCLIPS}-${FPS}fps-mo3-nofill-p${PATCH}"
    submit "G5 patch ${PATCH}" "${STEM}" "$(mem_for "${PATCH}")" "${TIME_ALL}"
done

sweep_totals

if (( ${#SUBMITTED_IDS[@]} == 0 )); then
    echo "nothing submitted, so nothing to export" >&2
    exit 1
fi

DEP="$(IFS=:; echo "${SUBMITTED_IDS[*]}")"
echo
echo "chaining export after ${#SUBMITTED_IDS[@]} training job(s): ${DEP}"
# afterany, not afterok: an arm that dies must not withhold the arms that
# lived, and a dead arm is itself a result worth exporting.
sbatch --dependency="afterany:${DEP}" \
       experiments/G5_patch_size/export_sherlock.sh

echo
echo "Submitted. The export runs itself when the last arm ends. Watch with:"
echo
echo "    squeue -u \$USER"
echo
