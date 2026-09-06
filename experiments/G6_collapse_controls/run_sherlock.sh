#!/usr/bin/env bash
# G6 -- five arms that decide why every video run lands on the same number.
# See README.md.
#
# val_loss 0.5245 is H(0.2182), the entropy of the mean of the data: what a
# decoder emitting one constant costs. The loss is BCE averaged over the whole
# flattened feature vector, so its floor depends on the data density and not on
# U, A or P -- which is why sixteen runs across different architectures all
# landed between 0.488 and 0.597.
#
# Three changes this fork made against upstream push toward that solution.
# Each arm reverts exactly one. Arm D is a positive control, not a candidate.
#
# RUN STEP 0 FIRST. It costs seconds and it needs no GPU:
#
#     cd $SCRATCH/panos/sgg-thesis && git pull
#     python3 tools/planner/collapse_floor.py data/npz/video/vidvrd/overfit/*.npz
#
# If the existing runs are NOT at the floor, the hypothesis is wrong and these
# arms are the wrong arms. Read the result before submitting.
#
# Then, one command:
#
#     mkdir -p logs && sbatch experiments/G6_collapse_controls/run_sherlock.sh
#
# The wall clock below covers the BAKE, not the training. Each arm is its own
# job with its own budget.
#SBATCH --job-name=fosae-G6
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/G6.%j.out
#SBATCH --error=logs/G6.%j.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs

source venv/bin/activate 2>/dev/null || source activate.sh
source sh/sweep_lib.sh

FPS="${FPS:-30}"
NPZ="data/npz/video/vidvrd/overfit"

# ── the data ────────────────────────────────────────────────────────────────
# One clip, not a set of them. This experiment asks why the loss stops
# falling, and that question is answered on the smallest data that shows it.
# ILSVRC2015_train_00150010 is the clip H14, the oracle exports and the
# geometry sweep all use, so its behaviour is already characterised.
CLIP="${CLIP:-ILSVRC2015_train_00150010}"
STEM="G6-${CLIP}-${FPS}fps-mo3-p32"
NCLIPS=1

section "G6  data -- one clip, patch 32, fill OFF"

# patch 32 on purpose: it is the sweep's configuration, the one that collapsed.
# Patch is held FIXED across every arm below. It sets the share of the gradient
# the position signal gets (51.0 percent at patch 8, 6.1 at 32), so moving it
# would change the loss function itself and the arms would stop being
# comparable. See tools/planner/collapse_floor.py::box_share.
if [[ -f "${NPZ}/${STEM}.npz" ]]; then
    echo "have  ${STEM}"
else
    echo "bake  ${STEM}"
    BAKE_LOG="logs/G6-bake.$$.log"
    python3 setup-dataset.py video_vidvrd all \
        --video-id "${CLIP}" --fps "${FPS}" \
        --max-objects 3 --patch-size 32 --out-name "${STEM}" \
        2>&1 | tee "${BAKE_LOG}"

    # The loader skips a frame whose jpg is missing and carries on. It now says
    # so, but a bake that lost most of its frames must stop the run rather than
    # train on what survived. The pattern is the loader's own line:
    #     [vidvrd-loader] category_filter=None strict=False loaded 1/1 videos, N states
    LOADED="$(sed -n 's/.*loaded \([0-9][0-9]*\)\/[0-9][0-9]* videos.*/\1/p' \
              "${BAKE_LOG}" | tail -1)"
    if [[ -z "${LOADED}" ]]; then
        echo "FATAL: no '[vidvrd-loader] ... loaded N/M videos' line in" >&2
        echo "       ${BAKE_LOG}, so the clip count could not be checked." >&2
        exit 3
    fi
    if [[ "${LOADED}" != "${NCLIPS}" ]]; then
        echo "FATAL: baked ${LOADED} clips, expected ${NCLIPS}." >&2
        exit 3
    fi

    # The loader also reports how many annotated frames it kept. A bake that
    # kept a tenth of them is the frame-rate mismatch, and training on it
    # measures nothing.
    grep -F '[vidvrd-loader] kept' "${BAKE_LOG}" || true
fi

# The floor this whole experiment is about, printed before anything trains, so
# it is in the log beside the results rather than recomputed afterwards.
section "G6  the floor of this dataset"
python3 tools/planner/collapse_floor.py "${NPZ}/${STEM}.npz" || true

section "G6  train -- one knob per arm"

# Everything not named by an arm is the configuration that collapsed. The
# baseline repeats it, so the comparison is against a number this code produced
# today and not against a remembered one.
#
# Note what is NOT here: PREENC_LAYERS and PREENC_DIM. H14 set them to 2 and
# 1000 and reached val 0.1216. Leaving them out is what makes the base arm the
# collapsed configuration, and arm D puts them back as a positive control.
SWEEP_DEFAULTS=(EPOCH=3000 LR=0.001 BATCH=1000
                TRANSITION_MODE=sequential
                U=40 A=2 P=10)

# Identical resources for every arm. An arm that ran out of wall clock would
# train for fewer epochs than its neighbours, and the experiment would then
# measure the budget instead of the knob.
MEM_ALL="${MEM_ALL:-32G}"
TIME_ALL="${TIME_ALL:-10:00:00}"

# arm            tag                                          overrides
submit "G6 base   (as it collapsed)"          "${STEM}" "${MEM_ALL}" "${TIME_ALL}" \
       ZEROSUPPRESS=0.05 ZEROSUPPRESS_DELAY=0.05 MAX_TEMPERATURE=1.0

submit "G6 A      (zerosuppress -> upstream)" "${STEM}" "${MEM_ALL}" "${TIME_ALL}" \
       ZEROSUPPRESS=0.0  ZEROSUPPRESS_DELAY=0.05 MAX_TEMPERATURE=1.0

submit "G6 B      (temperature  -> upstream)" "${STEM}" "${MEM_ALL}" "${TIME_ALL}" \
       ZEROSUPPRESS=0.05 ZEROSUPPRESS_DELAY=0.05 MAX_TEMPERATURE=5.0

submit "G6 C      (delay        -> upstream)" "${STEM}" "${MEM_ALL}" "${TIME_ALL}" \
       ZEROSUPPRESS=0.05 ZEROSUPPRESS_DELAY=0.1  MAX_TEMPERATURE=1.0

submit "G6 D      (pre-encoder, POSITIVE CONTROL)" "${STEM}" "${MEM_ALL}" "${TIME_ALL}" \
       ZEROSUPPRESS=0.05 ZEROSUPPRESS_DELAY=0.05 MAX_TEMPERATURE=1.0 \
       PREENC_LAYERS=2 PREENC_DIM=1000

echo
echo "Arm D is the configuration already known to learn (H14 reached val"
echo "0.1216 with it). If D does not come out below the floor printed above,"
echo "the measurement is broken rather than the hypothesis, and nothing else"
echo "in this run should be believed."
echo
echo "When the jobs finish:"
echo "  python3 tools/list_runs.py --start today"
echo "  python3 tools/planner/collapse_floor.py ${NPZ}/${STEM}.npz --run <out_dir>"
sweep_totals
