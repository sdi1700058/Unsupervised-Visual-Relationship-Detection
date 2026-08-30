#!/usr/bin/env bash
# H14 — can a TRAINED FOSAE reach the number the oracle reached?
#
# This is the experiment the thesis turns on. Everything measured so far says
# the task is winnable and the measuring apparatus is right:
#
#     oracle latents, PDDL, window 8, ILSVRC2015_train_00150010
#     14/14 windows solved, mse_ratio 0.046, planner error 12.0
#     against a quantisation floor of 9.91 -- the planner is AT the floor
#
# but those latents were built from ground-truth boxes, not learned. No
# trained model has ever had a fair trial, because every trained-model number
# so far came from `ILSVRC2015_train_00005005`, a clip whose annotation this
# project's own --fill-annotations invented 56% of (SPEC V24).
#
# So H14 removes both defects at once:
#
#   1. Train on the 88 screened clips in eval/vidvrd_winnable_clips.txt --
#      every one fully annotated AND arithmetically able to beat the baseline
#      (the crossover criterion, EVAL.md 4.2). 8,522 real transitions, against
#      the 59 real transitions in the clip everything used to be measured on.
#   2. --fill-annotations is ABSENT. Not forgotten: filling these clips would
#      fabricate transitions, and fabricated transitions are what produced
#      every earlier negative result.
#
# The configuration is the one that learned (val 0.1216): pre-encoder on, 2
# layers, width 1000. The one thing swept is the latent size, because that is
# the knob measured to decide plannability rather than reconstruction:
#
#     U40 A2 P10, 400 bits -> 17 of 19 windows solved
#     U40 A2 P20, 800 bits ->  3 of 19 windows solved
#
# Same clip, same planner, same budget (sweep3.sh section F5). Fast Downward
# searches 2^(U*P) states, so the latent size is a planning decision that
# reconstruction loss alone would never surface.
#
# Usage -- one command, then walk away:
#
#     cd $SCRATCH/panos/sgg-thesis && git pull
#     mkdir -p logs && sbatch sh/h14.sh
#
# The exports are chained on with --dependency, so there is no second visit:
# when the training finishes the export job runs itself. Then, locally:
#
#     bash sh/h14_score.sh
#
#SBATCH --job-name=fosae-H14
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/H14.%j.out
#SBATCH --error=logs/H14.%j.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs

source venv/bin/activate 2>/dev/null || source activate.sh
source sh/sweep_lib.sh

FPS="${FPS:-30}"
NPZ="data/npz/video/vidvrd/overfit"
CLIPS="eval/vidvrd_winnable_clips.txt"

MID=(32G 6:00:00); BIG=(48G 10:00:00)

if [[ ! -f "${CLIPS}" ]]; then
    echo "FATAL: ${CLIPS} missing. It is produced by" >&2
    echo "  python3 tools/video/screen_vidvrd.py --winnable > ${CLIPS}" >&2
    exit 2
fi

# --video-id takes one comma-separated list, so the screened selection is
# folded into a single argument here rather than being pasted by hand.
IDS="$(tr -d '\r' < "${CLIPS}" | grep -v '^[[:space:]]*$' | paste -sd, -)"
NCLIPS="$(tr -d '\r' < "${CLIPS}" | grep -cv '^[[:space:]]*$')"

section "H14  bake ${NCLIPS} screened clips, fill OFF"

STEM="H14-winnable${NCLIPS}-${FPS}fps-mo3-nofill-p8"

if [[ -f "${NPZ}/${STEM}.npz" ]]; then
    echo "have  ${STEM}"
else
    echo "bake  ${STEM}  (${NCLIPS} clips, no fill)"
    python3 setup-dataset.py video_vidvrd all \
        --video-id "${IDS}" --fps "${FPS}" \
        --max-objects 3 --patch-size 8 --out-name "${STEM}"
    NBAKED=$((NBAKED + 1))
fi

section "H14  train -- the configuration that learned, latent size swept"

# Not listed = the 0.1216 run. Only the latent shape moves.
SWEEP_DEFAULTS=(EPOCH=3000 LR=0.001 BATCH=1000
                PREENC_LAYERS=2 PREENC_DIM=1000
                MAX_TEMPERATURE=1.0 TRANSITION_MODE=sequential
                A=2)

# P10 is the primary arm: it is the one measured to plan. The others bracket
# it, so a single bad seed does not decide the thesis.
submit "H14 U40 P10  (400 bits, primary)" "${STEM}" "${MID[@]}"  U=40 P=10
submit "H14 U20 P10  (200 bits)"          "${STEM}" "${MID[@]}"  U=20 P=10
submit "H14 U40 P5   (200 bits, wide)"    "${STEM}" "${MID[@]}"  U=40 P=5
submit "H14 U40 P20  (800 bits, control)" "${STEM}" "${BIG[@]}"  U=40 P=20

sweep_totals

# ── chain the exports, so this is a one-visit experiment ────────────────────
if (( ${#SUBMITTED_IDS[@]} == 0 )); then
    echo "nothing submitted, so nothing to export" >&2
    exit 1
fi

DEP="$(IFS=:; echo "${SUBMITTED_IDS[*]}")"
echo
echo "chaining export after ${#SUBMITTED_IDS[@]} training job(s): ${DEP}"
# afterany, not afterok: an arm that dies should not withhold the arms that
# lived. h14_export.sh exports whatever landed and says what did not.
sbatch --dependency="afterany:${DEP}" sh/h14_export.sh

cat <<'NOTE'

Submitted. Nothing else to do here -- the export runs itself when training
ends. Watch it with:

    squeue -u $USER

When the queue is empty, push the exports and score them locally:

    git add -f eval/exports/*H14*.npz && git commit -m "H14 exports" && git push
    # then on the workstation
    git pull && bash sh/h14_score.sh
NOTE
