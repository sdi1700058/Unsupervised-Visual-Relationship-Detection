#!/usr/bin/env bash
# Sweep 2: vary the data, not the optimiser.
#
# Round one moved every model hyperparameter and all sixteen runs landed at
# val_loss ~0.5. What round one never touched was the data itself: patch
# size, object slots, how many videos, and whether annotation gaps are
# filled. This sweep moves those, then repeats the model knobs on top of
# the patch size where position dominates the loss.
#
#   mkdir -p logs && sbatch sh/sweep2.sh
#   squeue -u $USER
#
# The mkdir matters: Slurm opens logs/sweep2.%j.out before this script
# runs, and logs/ is gitignored, so on a fresh clone the job dies with no
# output file to explain why.
#
# Each section bakes what it needs and submits it immediately, cheapest
# first, so GPU jobs start while the big categories are still baking.
# Baking skips any npz already on disk, so resubmitting after a failure
# costs nothing.
#
#SBATCH --job-name=fosae-sweep2
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/sweep2.%j.out
#SBATCH --error=logs/sweep2.%j.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs

source venv/bin/activate 2>/dev/null || source activate.sh

FPS="${FPS:-30}"
NPZ="data/npz/video/vidvrd/overfit"
FRAMES="data/video/vidvrd/frames_${FPS}fps/train"

# Every arm below reads these frames. Without the check, a missing or
# partial extraction makes all 23 bakes raise, all 40 submits print "skip",
# and the job exits 0 having queued nothing.
if [[ ! -d "${FRAMES}" ]]; then
    echo "FATAL: no ${FRAMES}. Run:  FPS=${FPS} bash sh/download_vidvrd.sh"
    exit 1
fi
NVID=$(ls "${FRAMES}" 2>/dev/null | wc -l)
if [[ "${NVID}" -lt 700 ]]; then
    echo "FATAL: only ${NVID} videos in ${FRAMES}, expected ~800."
    echo "       Extraction is incomplete. Rerun sh/download_vidvrd.sh."
    exit 1
fi
echo "frames ok: ${NVID} videos at ${FPS} fps"

# The base point. Every single-variable arm moves one thing away from this
# and holds the rest fixed, so a result points at a cause.
BASE_CAT="dog"
BASE_VID="ILSVRC2015_train_00005005"
BASE_MO=3
BASE_PATCH=32

# The patch size goes in every npz name. Round one baked
# `dog-<vid>-30fps-mo3-fill` at the default patch 32; without the suffix
# this sweep would either collide with that file or, worse, silently reuse
# it when BASE_PATCH changes.
clip_stem () {   # <max_objects> <fill|nofill> <patch>
    echo "${BASE_CAT}-${BASE_VID}-${FPS}fps-mo${1}-${2}-p${3}"
}

NBAKED=0
NJOBS=0
NFAILED=0

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
        echo "  BAKE FAILED ${name}, continuing"
        NFAILED=$((NFAILED + 1))
    fi
}

# submit <tag> <npz_stem> <mem> <time> [VAR=val ...]
#
# Resources are stated per arm rather than derived. AUTO_RESOURCES is off
# because sh/estimate_resources.sh keys its estimate on CATEGORY+FPS, and
# an npz-driven run has no CATEGORY to give it — submit.sh would default to
# `bicycle` and size every job for the wrong dataset.
submit () {
    local tag="$1" stem="$2" mem="$3" time="$4"; shift 4
    local f="${NPZ}/${stem}.npz"
    if [[ ! -f "${f}" ]]; then
        echo "SKIP  ${tag}  (no ${stem}.npz)"
        return 0
    fi
    echo "--- ${tag}"
    local out
    if out="$(env NPZ_PATH="${PWD}/${f}" FPS="${FPS}" \
                  DOMAIN=vidvrd NO_EARLYSTOP=1 AUTO_RESOURCES=0 \
                  MEM="${mem}" TIME="${time}" \
                  EPOCH=2000 LR=0.001 PREENC_LAYERS=0 MAX_TEMPERATURE=1.0 \
                  TRANSITION_MODE=sequential \
                  "$@" \
                  bash sh/submit.sh 2>&1)"; then
        echo "${out}" | grep -E "Submitted|OUT_DIR" || echo "${out}" | tail -2
        NJOBS=$((NJOBS + 1))
    else
        # A silent failure here costs a whole arm, so show why.
        echo "  SUBMIT FAILED:"
        echo "${out}" | tail -5 | sed 's/^/    /'
        NFAILED=$((NFAILED + 1))
    fi
}

section () {
    echo
    echo "=============================================================="
    echo "$*"
    echo "=============================================================="
}

CLIP=(--video-id "${BASE_VID}")
SMALL=(16G 2:00:00)      # one clip, ~135 states
MULTI=(32G 4:00:00)      # a few clips
WHOLE=(48G 8:00:00)      # a whole category
HUGE=(64G 12:00:00)      # a whole category at a large patch

# ====================================================================
section "A  patch size, on one clip"
# The feature vector is [patch**2 * 3 | bbox onehot] and the bbox block is
# a fixed 200 dims, so the patch alone sets how much of the loss is
# position rather than texture: 81% position at patch 4, 21% at 16, 6% at
# 32, 2% at 64. Round one ran only at 32. The lost prior work used 48 and
# 64, so measure the whole range rather than argue about the direction.
for P in 4 8 16 32 48 64; do
    N="$(clip_stem "${BASE_MO}" fill "${P}")"
    bake "${N}" "${BASE_CAT}" "${CLIP[@]}" \
        --max-objects "${BASE_MO}" --patch-size "${P}" --fill-annotations
    submit "patch ${P}" "${N}" "${SMALL[@]}"
done

# Patch 8 is the reference point for everything below: position carries
# half the loss there, which is the half the planner reads.
REF="$(clip_stem "${BASE_MO}" fill 8)"

# ====================================================================
section "B  annotation gap filling"
# Filling interpolates the frames VidVRD left unannotated. It multiplies
# the sample count, and it may equally be feeding the model invented
# positions. The fill counterpart is section A's patch-8 npz, so only the
# nofill half needs baking.
N="$(clip_stem "${BASE_MO}" nofill 8)"
bake "${N}" "${BASE_CAT}" "${CLIP[@]}" --max-objects "${BASE_MO}" --patch-size 8
submit "nofill" "${N}" "${SMALL[@]}"

# ====================================================================
section "C  object slots"
# An empty padded slot is free reconstruction credit; too few slots drops
# a real object. mo3 is the base, so 2, 5 and 8 bracket it.
for MO in 2 5 8; do
    N="$(clip_stem "${MO}" fill 8)"
    bake "${N}" "${BASE_CAT}" "${CLIP[@]}" \
        --max-objects "${MO}" --patch-size 8 --fill-annotations
    submit "max-objects ${MO}" "${N}" "${SMALL[@]}"
done

# ====================================================================
section "D  model knobs, at patch 8"
# Round one swept these at patch 32, where the bbox is 6% of the BCE
# signal, and nothing moved. Repeat them at patch 8, where it is 51%, to
# separate "the knob does nothing" from "the patch size was masking it".
# No new bake needed.
submit "U10 P8"        "${REF}" "${SMALL[@]}" U=10 A=2 P=8
submit "U40 P20"       "${REF}" "${SMALL[@]}" U=40 A=2 P=20
submit "U80 P40"       "${REF}" "${SMALL[@]}" U=80 A=2 P=40
submit "arity 3"       "${REF}" "${SMALL[@]}" U=20 A=3 P=16
submit "zerosup 0"     "${REF}" "${SMALL[@]}" ZEROSUPPRESS=0
submit "zerosup 0.5"   "${REF}" "${SMALL[@]}" ZEROSUPPRESS=0.5
submit "maxtemp 0.5"   "${REF}" "${SMALL[@]}" MAX_TEMPERATURE=0.5
submit "maxtemp 5"     "${REF}" "${SMALL[@]}" MAX_TEMPERATURE=5.0
submit "lr 3e-4"       "${REF}" "${SMALL[@]}" LR=0.0003
submit "lr 3e-3"       "${REF}" "${SMALL[@]}" LR=0.003
submit "batch 32"      "${REF}" 16G 4:00:00   BATCH=32
submit "batch 256"     "${REF}" "${SMALL[@]}" BATCH=256
submit "dropout 0"     "${REF}" "${SMALL[@]}" DROPOUT=0
submit "noise 0"       "${REF}" "${SMALL[@]}" NOISE=0
submit "preenc 2x1000" "${REF}" 32G 4:00:00   PREENC_LAYERS=2 PREENC_DIM=1000
submit "all_pairs"     "${REF}" 32G 6:00:00   TRANSITION_MODE=all_pairs EPOCH=500
submit "epoch 8000"    "${REF}" 16G 8:00:00   EPOCH=8000

# ====================================================================
section "E  how many videos"
# One clip is an overfit check; a whole category asks whether the
# relations generalise. The prior work sat in between, at 3 and 5 clips.
# dog has 51 videos under the strict primary-subject filter, so `all`
# below is genuinely all of them.
for NV in 3 5 15; do
    N="${BASE_CAT}-${NV}vids-${FPS}fps-mo${BASE_MO}-fill-p8"
    bake "${N}" "${BASE_CAT}" --max-videos "${NV}" \
        --max-objects "${BASE_MO}" --patch-size 8 --fill-annotations
    submit "${NV} videos" "${N}" "${MULTI[@]}" CATEGORY="${BASE_CAT}"
done

# ====================================================================
section "F  whole categories"
# Counted from the 800 train annotations under the loader's default
# VIDVRD_STRICT_CATEGORY=1, which keeps only videos whose primary subject
# is the category: person 124, car 66, dog 51, antelope 44, horse 36,
# monkey 32, bird 29. No --max-videos, so `all` means all.
# Format: category:max_objects
for SPEC in bird:3 monkey:4 horse:3 antelope:3 dog:5 car:5 person:5; do
    IFS=':' read -r CAT MO <<< "${SPEC}"
    N="${CAT}-${FPS}fps-all-mo${MO}-fill-p8"
    bake "${N}" "${CAT}" --max-objects "${MO}" --patch-size 8 --fill-annotations
    submit "category ${CAT}" "${N}" "${WHOLE[@]}" EPOCH=3000 CATEGORY="${CAT}"
done

# ====================================================================
section "G  patch size where there is real data volume"
# Confirms or kills the section A finding on the largest category.
for P in 32 48; do
    N="person-${FPS}fps-all-mo5-fill-p${P}"
    bake "${N}" person --max-objects 5 --patch-size "${P}" --fill-annotations
    submit "person patch ${P}" "${N}" "${HUGE[@]}" EPOCH=3000 CATEGORY=person
done

echo
echo "=============================================================="
echo "${NBAKED} npz baked this run, ${NJOBS} jobs submitted, ${NFAILED} failures"
echo
echo "  squeue -u \$USER"
echo "  python3 tools/rank_models.py --limit 60"
echo "  python3 tools/rank_models.py --plannable --weights-only"
echo "=============================================================="
