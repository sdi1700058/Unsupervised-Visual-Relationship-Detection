#!/usr/bin/env bash
# Sweep 2: vary everything the pipeline exposes.
#
# Round one moved the model hyperparameters and every run landed at
# val_loss ~0.5. What round one never touched was the data itself: patch
# size, object count, how many videos, whether annotation gaps are filled,
# and the frame rate. This sweep moves those, then repeats the model knobs
# on top of the data variant where position dominates the loss.
#
#   sbatch sh/sweep2.sh
#   squeue -u $USER
#
# Each section bakes what it needs and submits it right away, cheapest
# first, so GPU jobs start running while the big categories are still
# baking. Baking is skipped for any npz already on disk, so resubmitting
# after a failure costs nothing.
#
#SBATCH --job-name=fosae-sweep2
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sweep2.%j.out
#SBATCH --error=logs/sweep2.%j.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs

source venv/bin/activate 2>/dev/null || source activate.sh

FPS="${FPS:-30}"
NPZ="data/npz/video/vidvrd/overfit"

# The base point. Every single-variable run below moves one thing away
# from this and holds the rest fixed, so a result points at a cause.
BASE_CAT="dog"
BASE_VID="ILSVRC2015_train_00005005"
BASE_MO=3
BASE_PATCH=32
BASE="${BASE_CAT}-${BASE_VID}-${FPS}fps-mo${BASE_MO}-fill"

NBAKED=0
NJOBS=0

# bake <out_name> <category> [setup-dataset args...]
bake () {
    local name="$1" cat="$2"; shift 2
    if [[ -f "${NPZ}/${name}.npz" ]]; then
        echo "have  ${name}"
        return 0
    fi
    echo "bake  ${name}"
    if python3 setup-dataset.py video_vidvrd "${cat}" \
           --fps "${FPS}" --out-name "${name}" "$@"; then
        NBAKED=$((NBAKED + 1))
    else
        echo "  FAILED ${name}, continuing"
    fi
}

# submit <tag> <npz_stem> [VAR=val ...]
submit () {
    local tag="$1" stem="$2"; shift 2
    local f="${NPZ}/${stem}.npz"
    if [[ ! -f "${f}" ]]; then
        echo "skip  ${tag}  (no ${stem}.npz)"
        return 0
    fi
    # A big patch means a big input layer, so the request follows the file.
    local sz mem time
    sz=$(stat -c %s "${f}" 2>/dev/null || echo 0)
    if   [[ ${sz} -gt 2000000000 ]]; then mem=96G; time=8:00:00
    elif [[ ${sz} -gt  500000000 ]]; then mem=64G; time=6:00:00
    elif [[ ${sz} -gt  100000000 ]]; then mem=32G; time=4:00:00
    else                                  mem=16G; time=2:00:00
    fi
    echo "--- ${tag}"
    NJOBS=$((NJOBS + 1))
    env NPZ_PATH="${PWD}/${f}" \
        DOMAIN=vidvrd NO_EARLYSTOP=1 AUTO_RESOURCES=0 \
        MEM="${mem}" TIME="${time}" \
        EPOCH=2000 LR=0.001 PREENC_LAYERS=0 MAX_TEMPERATURE=1.0 \
        TRANSITION_MODE=sequential \
        "$@" \
        bash sh/submit.sh 2>&1 | grep -E "Submitted" || true
}

section () {
    echo
    echo "=============================================================="
    echo "$*"
    echo "=============================================================="
}

BV=(--video-id "${BASE_VID}")

# ====================================================================
section "A  patch size, on one clip"
# The feature vector is [patch**2 * 3 | bbox onehot] and the bbox block is
# a fixed 200 dims, so the patch alone decides how much of the loss is
# position rather than texture: 81% position at patch 4, 6% at patch 32,
# 2% at patch 64. The lost prior work used 48 and 64. Measure both ways.
for P in 4 8 16 32 48 64; do
    bake "${BASE}-p${P}" "${BASE_CAT}" "${BV[@]}" \
        --max-objects "${BASE_MO}" --patch-size "${P}" --fill-annotations
    submit "patch ${P}" "${BASE}-p${P}"
done

# ====================================================================
section "B  gap filling on and off"
# Filling interpolates the frames VidVRD left unannotated. It multiplies
# the sample count, and it may also be feeding the model invented
# positions. Both npz below are patch 32, so only the filling differs.
bake "${BASE_CAT}-${BASE_VID}-${FPS}fps-mo${BASE_MO}-nofill" "${BASE_CAT}" \
    "${BV[@]}" --max-objects "${BASE_MO}" --patch-size "${BASE_PATCH}"
submit "nofill" "${BASE_CAT}-${BASE_VID}-${FPS}fps-mo${BASE_MO}-nofill"

# ====================================================================
section "C  object slots"
# Empty padded slots are free reconstruction credit at high counts, and a
# dropped object at low ones.
for MO in 2 5 8; do
    bake "${BASE_CAT}-${BASE_VID}-${FPS}fps-mo${MO}-fill" "${BASE_CAT}" \
        "${BV[@]}" --max-objects "${MO}" --patch-size "${BASE_PATCH}" --fill-annotations
    submit "max-objects ${MO}" "${BASE_CAT}-${BASE_VID}-${FPS}fps-mo${MO}-fill"
done

# ====================================================================
section "D  frame rate"
# 3 fps is near the native annotation rate, so almost every frame carries
# a real annotation and consecutive frames differ a lot. 30 fps gives
# smooth motion and mostly interpolated positions.
if [[ -d "data/video/vidvrd/frames_3fps/train" ]]; then
    FPS=3 bake "${BASE_CAT}-${BASE_VID}-3fps-mo${BASE_MO}-fill" "${BASE_CAT}" \
        "${BV[@]}" --max-objects "${BASE_MO}" --patch-size "${BASE_PATCH}" --fill-annotations
    submit "dog 3fps" "${BASE_CAT}-${BASE_VID}-3fps-mo${BASE_MO}-fill"
else
    echo "no frames_3fps/train, skipping the frame rate arm"
fi

# ====================================================================
section "E  model knobs at patch 8"
# Round one swept these at patch 32 and nothing moved. Repeat them where
# position dominates the loss, to see whether the patch size was masking
# them. Nothing here needs a new bake.
K="${BASE}-p8"
submit "p8 U10 P8"        "${K}" U=10 A=2 P=8
submit "p8 U40 P20"       "${K}" U=40 A=2 P=20
submit "p8 U80 P40"       "${K}" U=80 A=2 P=40
submit "p8 arity 3"       "${K}" U=20 A=3 P=16
submit "p8 zerosup 0"     "${K}" ZEROSUPPRESS=0
submit "p8 zerosup 0.5"   "${K}" ZEROSUPPRESS=0.5
submit "p8 maxtemp 0.5"   "${K}" MAX_TEMPERATURE=0.5
submit "p8 maxtemp 5"     "${K}" MAX_TEMPERATURE=5.0
submit "p8 lr 3e-4"       "${K}" LR=0.0003
submit "p8 lr 3e-3"       "${K}" LR=0.003
submit "p8 batch 32"      "${K}" BATCH=32
submit "p8 batch 256"     "${K}" BATCH=256
submit "p8 dropout 0"     "${K}" DROPOUT=0
submit "p8 noise 0"       "${K}" NOISE=0
submit "p8 preenc 2x1000" "${K}" PREENC_LAYERS=2 PREENC_DIM=1000
submit "p8 all_pairs"     "${K}" TRANSITION_MODE=all_pairs EPOCH=500
submit "p8 epoch 8000"    "${K}" EPOCH=8000 TIME=8:00:00

# ====================================================================
section "F  how many videos"
# One clip is an overfit check. A whole category asks whether the
# relations generalise. The prior work sat in between, at 3 and 5 clips.
for NV in 3 5 15; do
    bake "${BASE_CAT}-${NV}vids-${FPS}fps-mo${BASE_MO}-fill" "${BASE_CAT}" \
        --max-videos "${NV}" --max-objects "${BASE_MO}" \
        --patch-size "${BASE_PATCH}" --fill-annotations
    submit "${NV} videos" "${BASE_CAT}-${NV}vids-${FPS}fps-mo${BASE_MO}-fill"
done

# ====================================================================
section "G  whole categories"
# The six that carry the most transitions in the inspection. person is
# the largest at 22222. These bakes are the slow ones, which is why they
# come last.  Format: category:max_videos:max_objects
for SPEC in bird:53:3 bicycle:60:4 monkey:47:4 dog:110:5 car:115:5 person:120:5; do
    IFS=':' read -r CAT NV MO <<< "${SPEC}"
    N="${CAT}-${FPS}fps-all-mo${MO}-fill"
    bake "${N}" "${CAT}" --max-videos "${NV}" --max-objects "${MO}" \
        --patch-size "${BASE_PATCH}" --fill-annotations
    submit "category ${CAT}" "${N}" EPOCH=3000
done

# ====================================================================
section "H  patch size where there is real data volume"
# Confirms or kills the section A finding on the largest category.
for P in 8 48; do
    bake "person-${FPS}fps-all-mo5-fill-p${P}" person \
        --max-videos 120 --max-objects 5 --patch-size "${P}" --fill-annotations
    submit "person patch ${P}" "person-${FPS}fps-all-mo5-fill-p${P}" EPOCH=3000
done

if [[ -d "data/video/vidvrd/frames_3fps/train" ]]; then
    FPS=3 bake "person-3fps-all-mo5-fill" person \
        --max-videos 120 --max-objects 5 --patch-size "${BASE_PATCH}" --fill-annotations
    submit "person 3fps" "person-3fps-all-mo5-fill" EPOCH=3000
fi

echo
echo "=============================================================="
echo "${NBAKED} npz baked this run, ${NJOBS} jobs submitted"
echo
echo "  squeue -u \$USER"
echo "  python3 tools/rank_models.py --limit 60"
echo "  python3 tools/rank_models.py --plannable --weights-only"
echo "=============================================================="
