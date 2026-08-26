#!/usr/bin/env bash
# Sweep 3: build out from the one configuration that learned.
#
# Across 123 runs, exactly two beat val_loss 0.14, and they are the only
# two with the pre-encoder switched on:
#
#   best 0.1216  preencoder_layers=2 preencoder_dimention=1000  40/50 codes
#   best 0.1308  preencoder_layers=2 preencoder_dimention=1000  16/50 codes
#
# Every other run had preencoder_layers=0 and landed between 0.24 and 0.44,
# most of them with a near-dead latent. So this sweep holds the pre-encoder
# on and moves everything else around it.
#
# The pre-encoder also decides the memory. FirstOrderSAE's attention step is
# einsum("buao,bof->buaf"), which builds batch x U x A x preencoder_dimention.
# With preencoder_layers=0 the model sets that dimension to the raw feature
# vector (model.py:1472), so patch 64 needs 16 GB for one tensor and OOMs.
# With the pre-encoder on it is fixed at preencoder_dimention, and patch 64
# costs the same 1.3 GB as patch 8. Bigger patches are cheap here, not
# expensive.
#
#   mkdir -p logs && sbatch sh/sweep3.sh
#
# Most arms reuse npz that sweep2 already baked, so this submits fast.
#
#SBATCH --job-name=fosae-sweep3
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/sweep3.%j.out
#SBATCH --error=logs/sweep3.%j.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs

source venv/bin/activate 2>/dev/null || source activate.sh

FPS="${FPS:-30}"
NPZ="data/npz/video/vidvrd/overfit"

CLIP="dog-ILSVRC2015_train_00005005-${FPS}fps-mo3-fill"
NBAKED=0; NJOBS=0; NFAILED=0

bake () {
    local name="$1" cat="$2"; shift 2
    [[ -f "${NPZ}/${name}.npz" ]] && { echo "have  ${name}"; return 0; }
    echo "bake  ${name}"
    if python3 setup-dataset.py video_vidvrd "${cat}" \
           --fps "${FPS}" --out-name "${name}" "$@"; then
        NBAKED=$((NBAKED + 1))
    else
        echo "  BAKE FAILED ${name}"; NFAILED=$((NFAILED + 1))
    fi
}

# submit <tag> <npz_stem> <mem> <time> [VAR=val ...]
#
# The defaults here ARE the winning configuration. An arm only names what it
# changes, so anything not listed is the 0.1216 run.
submit () {
    local tag="$1" stem="$2" mem="$3" time="$4"; shift 4
    local f="${NPZ}/${stem}.npz"
    if [[ ! -f "${f}" ]]; then
        echo "SKIP  ${tag}  (no ${stem}.npz)"; return 0
    fi
    echo "--- ${tag}"
    local out
    if out="$(env NPZ_PATH="${PWD}/${f}" FPS="${FPS}" \
                  DOMAIN=vidvrd NO_EARLYSTOP=1 AUTO_RESOURCES=0 \
                  MEM="${mem}" TIME="${time}" \
                  EPOCH=2000 LR=0.001 BATCH=1000 \
                  PREENC_LAYERS=2 PREENC_DIM=1000 \
                  MAX_TEMPERATURE=1.0 TRANSITION_MODE=sequential \
                  "$@" \
                  bash sh/submit.sh 2>&1)"; then
        echo "${out}" | grep -E "Submitted|OUT_DIR" || echo "${out}" | tail -2
        NJOBS=$((NJOBS + 1))
    else
        echo "  SUBMIT FAILED:"; echo "${out}" | tail -5 | sed 's/^/    /'
        NFAILED=$((NFAILED + 1))
    fi
}

section () { echo; echo "=========================================="; echo "$*"; echo "=========================================="; }

SMALL=(16G 2:00:00); MID=(32G 4:00:00); BIG=(48G 8:00:00); HUGE=(64G 16:00:00)

# ==========================================================
section "A  reproduce, then move the pre-encoder width"
# 1000 is the width that worked. Nothing has tried any other value.
submit "preenc 1000 (reproduce)" "${CLIP}-p8" "${SMALL[@]}"
for D in 256 512 2000 4000; do
    submit "preenc dim ${D}" "${CLIP}-p8" "${SMALL[@]}" PREENC_DIM="${D}"
done

# ==========================================================
section "B  pre-encoder depth"
for L in 1 3; do
    submit "preenc layers ${L}" "${CLIP}-p8" "${SMALL[@]}" PREENC_LAYERS="${L}"
done
# The control that says the pre-encoder is doing the work.
submit "preenc OFF (control)" "${CLIP}-p8" "${SMALL[@]}" PREENC_LAYERS=0

# ==========================================================
section "C  patch size, now that it is affordable"
# sweep2 already baked all of these. With the pre-encoder the attention
# tensor no longer grows with the patch, so this is the resolution sweep
# that previously OOMed.
for P in 4 16 32 48 64; do
    submit "patch ${P}" "${CLIP}-p${P}" "${MID[@]}"
done

# ==========================================================
section "D  zero-suppression"
# At preencoder_layers=0, zerosuppress=0.05 was killing the latent: those
# runs came back with 0-30 of 800 bits live, while the zerosuppress=0 runs
# kept 97-184. Worth re-testing now that the pre-encoder changed the regime.
for Z in 0.0 0.01 0.1; do
    submit "zerosuppress ${Z}" "${CLIP}-p8" "${SMALL[@]}" ZEROSUPPRESS="${Z}"
done

# ==========================================================
section "E  Gumbel temperature floor"
# min_temperature was hardcoded at 0.7 and never reachable until now. The
# latent anneals down to this value, and 0.7 never gets near-discrete, so
# the decoder learns to read fractional codes that rounding then destroys.
for T in 0.1 0.3 0.5; do
    submit "min temp ${T}" "${CLIP}-p8" "${SMALL[@]}" MIN_TEMPERATURE="${T}"
done
# A real anneal needs room between the two ends.
submit "temp 5.0 -> 0.1" "${CLIP}-p8" "${SMALL[@]}" MAX_TEMPERATURE=5.0 MIN_TEMPERATURE=0.1
submit "temp 2.0 -> 0.2" "${CLIP}-p8" "${SMALL[@]}" MAX_TEMPERATURE=2.0 MIN_TEMPERATURE=0.2

# ==========================================================
section "F  latent capacity, and the dense width"
submit "U20 P10"    "${CLIP}-p8" "${SMALL[@]}" U=20 A=2 P=10
submit "U80 P40"    "${CLIP}-p8" "${MID[@]}"   U=80 A=2 P=40
# layer was hardcoded at 200, chosen for MNIST-sized inputs.
for L in 400 1000; do
    submit "layer ${L}" "${CLIP}-p8" "${SMALL[@]}" LAYER="${L}"
done

# ==========================================================
section "G  whole categories, winning configuration"
for SPEC in bird:3 monkey:4 horse:3 antelope:3 dog:5 car:5 person:5; do
    IFS=':' read -r CAT MO <<< "${SPEC}"
    submit "category ${CAT}" "${CAT}-${FPS}fps-all-mo${MO}-fill-p8" \
        "${BIG[@]}" EPOCH=3000 CATEGORY="${CAT}"
done

# ==========================================================
section "H  resolution where there is data volume"
bake "dog-${FPS}fps-all-mo5-fill-p32" dog --max-objects 5 --patch-size 32 --fill-annotations
submit "dog all p32"    "dog-${FPS}fps-all-mo5-fill-p32"    "${BIG[@]}" EPOCH=3000 CATEGORY=dog
submit "person all p32" "person-${FPS}fps-all-mo5-fill-p32" "${HUGE[@]}" EPOCH=3000 CATEGORY=person

# ==========================================================
section "I  across categories"
# Does one predicate set cover several object types, or does each category
# need its own? Groups share a motion style; `all` shares nothing.
QUAD="dog,horse,antelope,zebra,sheep,cattle,elephant,lion,bear,tiger,fox"
VEHI="car,motorcycle,watercraft,airplane,bicycle,train,bus"

bake "quadrupeds-${FPS}fps-mo5-fill-p8" "${QUAD}" --max-objects 5 --patch-size 8 --fill-annotations
submit "quadrupeds" "quadrupeds-${FPS}fps-mo5-fill-p8" "${BIG[@]}" EPOCH=2000

bake "vehicles-${FPS}fps-mo5-fill-p8" "${VEHI}" --max-objects 5 --patch-size 8 --fill-annotations
submit "vehicles" "vehicles-${FPS}fps-mo5-fill-p8" "${BIG[@]}" EPOCH=2000

bake "allcat-${FPS}fps-mo5-fill-p8" all --max-objects 5 --patch-size 8 --fill-annotations
submit "all 800 videos" "allcat-${FPS}fps-mo5-fill-p8" "${HUGE[@]}" EPOCH=1000

echo
echo "=========================================="
echo "${NBAKED} baked, ${NJOBS} submitted, ${NFAILED} failed"
echo
echo "  python3 tools/diagnose_collapse.py --limit 60"
echo "=========================================="
