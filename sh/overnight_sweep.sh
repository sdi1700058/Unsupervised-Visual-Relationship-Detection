#!/usr/bin/env bash
# Unattended run: fetch data, inspect, bake, then fan out training jobs.
#
# Submit this one job and go. It prepares the data on a CPU node, records
# which categories are worth using, bakes the npz files, and submits the GPU
# training jobs itself. Nothing else needs a human.
#
#   sbatch sh/overnight_sweep.sh
#   squeue -u $USER
#
# Skips the download when the frames are already there, so it is safe to
# resubmit after a failure.
#
#SBATCH --job-name=fosae-sweep-prep
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=logs/sweep-prep.%j.out
#SBATCH --error=logs/sweep-prep.%j.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs eval/exports

source venv/bin/activate 2>/dev/null || source activate.sh

FPS="${FPS:-30}"
EPOCH="${EPOCH:-2000}"
FRAMES="data/video/vidvrd/frames_${FPS}fps/train"
OVERFIT="data/npz/video/vidvrd/overfit"

echo "=============================================================="
echo "STEP 1  data"
echo "=============================================================="
if [[ -d "${FRAMES}" ]] && [[ $(ls "${FRAMES}" 2>/dev/null | wc -l) -gt 700 ]]; then
    echo "frames already present: $(ls "${FRAMES}" | wc -l) videos"
else
    echo "downloading and extracting at ${FPS} fps; this is the slow part"
    FPS="${FPS}" bash sh/download_vidvrd.sh
fi

echo "annotations: $(ls data/video/vidvrd/annotations/train 2>/dev/null | wc -l) train"
echo "frames:      $(ls "${FRAMES}" 2>/dev/null | wc -l) videos"

echo
echo "=============================================================="
echo "STEP 2  which categories carry enough data"
echo "=============================================================="
# Recorded so the choices below can be judged after the fact, and so the
# next sweep can be aimed better.
python3 tools/video/inspect_vidvrd.py --fps "${FPS}" --threshold 5000 \
    2>&1 | tee logs/inspect_vidvrd_${FPS}fps.txt

echo
echo "=============================================================="
echo "STEP 3  bake"
echo "=============================================================="

# Single clips. These four came out of earlier rounds: each has one clear
# subject and few enough objects that padding does not swamp the loss.
# Format: category:video_id:max_objects
CLIPS=(
    "dog:ILSVRC2015_train_00005005:3"
    "bird:ILSVRC2015_train_00194010:2"
    "panda:ILSVRC2015_train_00134003:2"
    "sheep:ILSVRC2015_train_00256010:3"
)

for SPEC in "${CLIPS[@]}"; do
    IFS=':' read -r CAT VID MO <<< "${SPEC}"
    NAME="${CAT}-${VID}-${FPS}fps-mo${MO}-fill"
    if [[ -f "${OVERFIT}/${NAME}.npz" ]]; then
        echo "have ${NAME}"
        continue
    fi
    echo "baking ${NAME}"
    python3 setup-dataset.py video_vidvrd "${CAT}" \
        --video-id "${VID}" --fps "${FPS}" --max-objects "${MO}" \
        --fill-annotations --out-name "${NAME}" \
    || echo "  FAILED ${NAME}, continuing"
done

# Whole categories, for the cross-video question. Bigger and slower, and
# the per-category counts sit under the threshold, so treat these as a
# stretch rather than the main event.
for CAT in dog person; do
    NAME="${CAT}-${FPS}fps-allvids"
    if [[ -f "${OVERFIT}/${NAME}.npz" ]]; then
        echo "have ${NAME}"
        continue
    fi
    echo "baking ${NAME}"
    python3 setup-dataset.py video_vidvrd "${CAT}" \
        --fps "${FPS}" --max-objects 5 --fill-annotations \
        --max-videos 25 --out-name "${NAME}" \
    || echo "  FAILED ${NAME}, continuing"
done

echo
ls -lh "${OVERFIT}"/*.npz 2>/dev/null || echo "NO NPZ BAKED - stopping"
ls "${OVERFIT}"/*.npz >/dev/null 2>&1 || exit 1

echo
echo "=============================================================="
echo "STEP 4  submit training"
echo "=============================================================="

PRIMARY="${OVERFIT}/dog-ILSVRC2015_train_00005005-${FPS}fps-mo3-fill.npz"

submit () {
    local tag="$1"; shift
    echo "--- ${tag}"
    env "$@" \
        DOMAIN=vidvrd EPOCH="${EPOCH}" NO_EARLYSTOP=1 \
        MEM=16G TIME=1:00:00 AUTO_RESOURCES=0 \
        bash sh/submit.sh 2>&1 | grep -E "Submitted|OUT_DIR" || true
}

# A. The MNIST-proven configuration, on each clip. If none of these learn,
#    the problem is the data, not the hyperparameters.
for NPZ in "${OVERFIT}"/*-mo[0-9]-fill.npz; do
    [[ -f "${NPZ}" ]] || continue
    submit "baseline $(basename "${NPZ}" .npz)" \
        NPZ_PATH="${NPZ}" TRANSITION_MODE=sequential \
        LR=0.001 PREENC_LAYERS=0 MAX_TEMPERATURE=1.0
done

# B. Sweep on the primary clip. One knob at a time, so a result points at
#    a cause.
if [[ -f "${PRIMARY}" ]]; then
    # Sparsity is the first suspect for a collapsed latent.
    submit "zerosuppress off" NPZ_PATH="${PRIMARY}" TRANSITION_MODE=sequential \
        LR=0.001 PREENC_LAYERS=0 MAX_TEMPERATURE=1.0 ZEROSUPPRESS=0

    submit "zerosuppress 0.2" NPZ_PATH="${PRIMARY}" TRANSITION_MODE=sequential \
        LR=0.001 PREENC_LAYERS=0 MAX_TEMPERATURE=1.0 ZEROSUPPRESS=0.2

    # Colder start: discretise sooner.
    submit "maxtemp 0.5" NPZ_PATH="${PRIMARY}" TRANSITION_MODE=sequential \
        LR=0.001 PREENC_LAYERS=0 MAX_TEMPERATURE=0.5

    # The paper's own temperature, as a control.
    submit "maxtemp 5.0" NPZ_PATH="${PRIMARY}" TRANSITION_MODE=sequential \
        LR=0.001 PREENC_LAYERS=0 MAX_TEMPERATURE=5.0

    # U=40,P=20 gives 800 bits for a two-object scene. Almost certainly far
    # more capacity than the scene needs.
    submit "small U10 P8" NPZ_PATH="${PRIMARY}" TRANSITION_MODE=sequential \
        LR=0.001 PREENC_LAYERS=0 MAX_TEMPERATURE=1.0 U=10 A=2 P=8

    submit "mid U20 P16" NPZ_PATH="${PRIMARY}" TRANSITION_MODE=sequential \
        LR=0.001 PREENC_LAYERS=0 MAX_TEMPERATURE=1.0 U=20 A=2 P=16

    # More gradient steps per epoch on a small set.
    submit "batch 32" NPZ_PATH="${PRIMARY}" TRANSITION_MODE=sequential \
        LR=0.001 PREENC_LAYERS=0 MAX_TEMPERATURE=1.0 BATCH=32

    # Every frame pair, not only neighbours. Fewer epochs, far more batches.
    submit "all_pairs" NPZ_PATH="${PRIMARY}" TRANSITION_MODE=all_pairs \
        LR=0.001 PREENC_LAYERS=0 MAX_TEMPERATURE=1.0 EPOCH=500

    # The old video default, to confirm it is the thing that was wrong.
    submit "old default" NPZ_PATH="${PRIMARY}" TRANSITION_MODE=sequential \
        LR=0.0001 PREENC_LAYERS=2
fi

# C. Cross-video.
for NPZ in "${OVERFIT}"/*-allvids.npz; do
    [[ -f "${NPZ}" ]] || continue
    submit "crossvid $(basename "${NPZ}" .npz)" \
        NPZ_PATH="${NPZ}" TRANSITION_MODE=sequential \
        LR=0.001 PREENC_LAYERS=0 MAX_TEMPERATURE=1.0 TIME=2:00:00
done

echo
echo "=============================================================="
echo "submitted. in the morning:"
echo "    squeue -u \$USER"
echo "    python3 tools/rank_models.py --limit 40"
echo "    python3 tools/rank_models.py --plannable --weights-only"
echo "=============================================================="
