#!/usr/bin/env bash
# G4 -- sweep U and P together. See README.md.
#
# U is the number of predicate units and P the number of predicates. The latent
# is U*P bits. U has never been moved: every trained-model number in this
# project comes from U=40 or U=20. P has moved a little (H14 ran P5, P10, P20),
# always at U=40 or U=20.
#
# This bakes nothing new. It reuses the exact npz H14 trained on, so the four
# H14 cells and the ten new cells form one grid over identical data, and the
# only thing that differs between cells is the latent shape.
#
# One command, then walk away:
#
#     cd $SCRATCH/panos/sgg-thesis && git pull
#     mkdir -p logs && sbatch experiments/G4_up_sweep/run_sherlock.sh
#
# The export is chained on with --dependency=afterany, so there is no second
# visit. When the last training job ends, the export job runs itself.
#
# The wall clock below covers the BAKE, not the training. Each arm is a
# separate job with its own budget. The bake is skipped when the npz is already
# on disk, which is the expected case because H14 built it.
#SBATCH --job-name=fosae-G4
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/G4.%j.out
#SBATCH --error=logs/G4.%j.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs

source venv/bin/activate 2>/dev/null || source activate.sh
source sh/sweep_lib.sh

FPS="${FPS:-30}"
NPZ="data/npz/video/vidvrd/overfit"

# ── the data, pinned ────────────────────────────────────────────────────────
# The same 88 screened clips H14 trained on, inlined here rather than read from
# a file. eval/ is gitignored, so eval/vidvrd_winnable_clips.txt never reaches
# Sherlock and reading it would abort the run.
#
# Produced by, and re-derivable with, exactly this command:
#
#   python3 tools/video/screen_vidvrd.py --winnable-only --no-fill-only \
#       --min-frames 45 --list eval/vidvrd_winnable_clips.txt
#
# Criteria (EVAL.md 4.2): fully annotated, and the quantisation floor sits
# below the linear-interpolation baseline, so mse_ratio < 1 is arithmetically
# reachable. 8,522 real transitions in total.
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

# An override, when a newer screen exists locally and has been copied over.
# The pinned list stays the default, because the grid must sit on the same data
# H14 sat on.
CLIPS="${CLIPS:-}"
if [[ -n "${CLIPS}" && -f "${CLIPS}" ]]; then
    echo "using clip list from ${CLIPS}"
    OVERRIDE="$(tr -d '\r' < "${CLIPS}" | grep -v '^[[:space:]]*$' | paste -sd, - || true)"
    if [[ -z "${OVERRIDE}" ]]; then
        echo "FATAL: ${CLIPS} exists but contains no clip ids." >&2
        exit 2
    fi
    IDS="${OVERRIDE}"
    echo "WARNING: the grid no longer shares its data with H14, so the four"
    echo "         H14 cells will not appear in the figure." >&2
fi

NCLIPS="$(echo "${IDS}" | tr ',' '\n' | grep -c .)"

# The stem is H14's, on purpose. Same data, same name, one grid.
STEM="H14-winnable${NCLIPS}-${FPS}fps-mo3-nofill-p8"

section "G4  data -- reuse H14's bake of ${NCLIPS} screened clips, fill OFF"

if [[ -f "${NPZ}/${STEM}.npz" ]]; then
    echo "have  ${STEM}  (built by H14; nothing to bake)"
else
    echo "bake  ${STEM}  (${NCLIPS} clips, no fill)"
    BAKE_LOG="logs/G4-bake.$$.log"
    python3 setup-dataset.py video_vidvrd all \
        --video-id "${IDS}" --fps "${FPS}" \
        --max-objects 3 --patch-size 8 --out-name "${STEM}" \
        2>&1 | tee "${BAKE_LOG}"

    # puzzle_vidvrd skips a clip whose frames are missing and carries on, so a
    # partial frame extraction would silently shrink the grid while the stem
    # still says "winnable88". Fail loudly instead.
    LOADED="$(grep -oE '[0-9]+ videos? loaded' "${BAKE_LOG}" | tail -1 \
              | grep -oE '^[0-9]+' || true)"
    if [[ -n "${LOADED}" && "${LOADED}" != "${NCLIPS}" ]]; then
        echo "FATAL: baked ${LOADED} clips, expected ${NCLIPS}." >&2
        echo "       Frames are missing on this machine. See ${BAKE_LOG}." >&2
        exit 3
    fi
    NBAKED=$((NBAKED + 1))
fi

section "G4  train -- the grid over U and P"

# Everything except U and P is H14's configuration, which is the one that
# learned (val 0.1216). A cell must differ from its neighbours in the latent
# shape and in nothing else.
SWEEP_DEFAULTS=(EPOCH=3000 LR=0.001 BATCH=1000
                PREENC_LAYERS=2 PREENC_DIM=1000
                MAX_TEMPERATURE=1.0 TRANSITION_MODE=sequential
                A=2)

# IDENTICAL resources for every cell. A cell that ran out of wall clock would
# train for fewer epochs than its neighbours, and the grid would then measure
# the budget instead of the latent shape. 10 hours covers the widest cell here,
# and a cell that finishes early costs only what it used.
MEM_ALL="${MEM_ALL:-48G}"
TIME_ALL="${TIME_ALL:-10:00:00}"

U_LIST="${U_LIST:-5 10 20 40 80}"
P_LIST="${P_LIST:-5 10 20}"

# Cells H14 already trained on this exact npz. Skipped, not repeated. Their
# exports drop into the same grid because they carry the same stem.
H14_CELLS="U40P5 U40P10 U40P20 U20P10"

# One corner left out on purpose. H14 measured that 800 bits already collapses
# planning to 3 of 19 windows; 1600 bits doubles the exponent of the search
# space for the cell that costs the most and tells the least.
DROP_CELLS="U80P20"

in_list () {
    local needle="$1"; shift
    local item
    for item in "$@"; do
        if [[ "${item}" == "${needle}" ]]; then
            return 0
        fi
    done
    return 1
}

NSKIP=0
for U in ${U_LIST}; do
    for P in ${P_LIST}; do
        CELL="U${U}P${P}"
        BITS=$(( U * P ))
        if in_list "${CELL}" ${H14_CELLS}; then
            echo "skip  ${CELL}  (${BITS} bits) -- H14 ran this cell on this npz"
            NSKIP=$((NSKIP + 1))
            continue
        fi
        if in_list "${CELL}" ${DROP_CELLS}; then
            echo "skip  ${CELL}  (${BITS} bits) -- left out of the grid on purpose"
            NSKIP=$((NSKIP + 1))
            continue
        fi
        submit "G4 ${CELL}  (${BITS} bits)" "${STEM}" \
               "${MEM_ALL}" "${TIME_ALL}" U="${U}" P="${P}"
    done
done

echo
echo "skipped ${NSKIP} cell(s)"
sweep_totals

# ── chain the export, so this is a one-visit experiment ─────────────────────
if (( ${#SUBMITTED_IDS[@]} == 0 )); then
    echo "nothing submitted, so nothing to export" >&2
    exit 1
fi

DEP="$(IFS=:; echo "${SUBMITTED_IDS[*]}")"
echo
echo "chaining export after ${#SUBMITTED_IDS[@]} training job(s): ${DEP}"
# afterany, not afterok: a cell that dies must not withhold the cells that
# lived. The export script exports whatever landed and says what did not.
sbatch --dependency="afterany:${DEP}" experiments/G4_up_sweep/export_sherlock.sh

cat <<'NOTE'

Submitted. Nothing else to do here -- the export runs itself when the last
training job ends. Watch it with:

    squeue -u $USER

When the queue is empty, push the exports and the training summary:

    git add -f eval/exports/*catH14-winnable*.npz eval/exports/G4_train.csv
    git commit -m "G4 exports" && git push

Then, on the workstation:

    git pull && bash experiments/G4_up_sweep/score_local.sh
NOTE
