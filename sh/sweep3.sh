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
section "F  latent structure: U, A and P"
# These three define the representation, so they get a real study rather
# than two spot checks. U is the number of predicate units, P the
# predicates per unit, A the arity. The latent is U*P bits.
#
# U*P is also the number of PDDL propositions the planner will search over,
# so this section serves both halves of the thesis at once: reconstruction
# wants capacity, planning wants few bits, and the two pull opposite ways.
# The paper's 40/2/20 is only a starting point.
#
# Memory is linear in U*A and flat in P (P never enters the attention
# tensor), so the whole grid fits: worst case here is 5.1 GB at U=160.

# F1. U, holding P.
for U in 5 10 20 80 160; do
    M=("${SMALL[@]}"); [[ ${U} -ge 80 ]] && M=("${MID[@]}")
    submit "U ${U} (P20)" "${CLIP}-p8" "${M[@]}" U="${U}" A=2 P=20
done

# F2. P, holding U. Free in memory, so the range is wide.
for P in 5 10 40 80 160; do
    submit "P ${P} (U40)" "${CLIP}-p8" "${SMALL[@]}" U=40 A=2 P="${P}"
done

# F3. Arity — how many objects one predicate relates. The paper reports
#     the result is insensitive to A, so this is a confirmation arm rather
#     than a search: worth knowing it also holds on real video, where the
#     relations are not hand-designed. Uses the 5-object clip so the arity
#     is never larger than the scene.
for A in 3 4; do
    submit "arity ${A}" "${CLIP/-mo3-/-mo5-}-p8" "${MID[@]}" U=40 A="${A}" P=20
done
submit "arity 2 (mo5 control)" "${CLIP/-mo3-/-mo5-}-p8" "${SMALL[@]}" U=40 A=2 P=20

# F4. Same 800 bits, five different shapes. This is the one that separates
#     "the latent needs N bits" from "the latent needs this structure" —
#     every arm here has identical capacity and identical PDDL state size.
for SHAPE in 10:80 20:40 80:10 160:5; do
    IFS=':' read -r U P <<< "${SHAPE}"
    M=("${SMALL[@]}"); [[ ${U} -ge 80 ]] && M=("${MID[@]}")
    submit "shape U${U} P${P} (800 bits)" "${CLIP}-p8" "${M[@]}" U="${U}" A=2 P="${P}"
done

# F5. Planner-sized latents. Fast Downward searches 2^(U*P) states, so 800
#     propositions is a lot to ask. These are the arms most likely to plan
#     even if they reconstruct slightly worse.
for SHAPE in 10:5 10:10 20:10 20:20; do
    IFS=':' read -r U P <<< "${SHAPE}"
    submit "small U${U} P${P} (${U}x${P} bits)" "${CLIP}-p8" "${SMALL[@]}" \
        U="${U}" A=2 P="${P}"
done

# ==========================================================
section "G  dense width"
# layer was hardcoded at 200, sized for MNIST puzzles. Every object vector
# is flattened through it before any predicate forms.
for L in 50 400 1000; do
    submit "layer ${L}" "${CLIP}-p8" "${SMALL[@]}" LAYER="${L}"
done

# ==========================================================
section "K  batch size, with the two things that broke it"
# All three batch=32 runs in sweep2 collapsed, but every one of them also
# had preencoder_layers=0, so they say nothing about the batch size. Two
# real confounds were never controlled:
#
#   Steps. The clip has ~120 training transitions, so batch 1000 is
#   FULL-BATCH descent — one gradient step per epoch. batch 32 takes four,
#   so at equal epochs it does 4x the updates, and batch 8 does 15x. The
#   winning run is a full-batch run; that is likely why it is stable.
#
#   Learning rate. lr=0.001 was tuned at full batch. The linear scaling
#   rule puts batch 32 at 3.2e-5, thirty times lower than what it was given.
#
# So each batch gets two arms: equal gradient steps at the base lr, and
# equal steps at the scaled lr. If it still collapses at both, that is a
# result about batch size. Until then it is a result about step counts.

epochs_for () {   # batch -> epochs giving ~2000 steps on the ~120-sample clip
    case "$1" in
        8)   echo 134  ;;
        16)  echo 250  ;;
        32)  echo 500  ;;
        64)  echo 1000 ;;
        *)   echo 2000 ;;   # batch >= 128 is already full-batch here
    esac
}

scaled_lr () {    # batch -> lr under the linear scaling rule from 1000/0.001
    python3 -c "print('%.3g' % (0.001 * $1 / 1000))"
}

for B in 8 16 32 64 128 256 512 2000; do
    E="$(epochs_for "${B}")"
    L="$(scaled_lr "${B}")"
    submit "batch ${B} steps-equalised" "${CLIP}-p8" "${SMALL[@]}" \
        BATCH="${B}" EPOCH="${E}"
    submit "batch ${B} steps+lr ${L}" "${CLIP}-p8" "${SMALL[@]}" \
        BATCH="${B}" EPOCH="${E}" LR="${L}"
done

# And the original failing point, kept honestly: batch 32 at 2000 epochs
# and lr 0.001, the exact arm that collapsed, now with the pre-encoder on.
# If the pre-encoder alone rescues it, that is worth knowing.
submit "batch 32 as-it-failed" "${CLIP}-p8" "${SMALL[@]}" BATCH=32 EPOCH=2000

# ==========================================================
section "L  delayed zero-suppression"
# zerosuppress_delay is the fraction of training before the sparsity
# penalty switches on, and it defaults to 0.05 — the penalty starts almost
# immediately, before the latent has formed anything to keep. That is the
# textbook recipe for a dead code, and it matches what the runs show:
# zerosuppress=0.05 kept 0-30 of 800 bits, zerosuppress=0 kept 97-184.
# Letting the autoencoder learn first, then tightening, is the standard fix
# and it has never been tried here.
for Z in 0.05 0.1 0.2; do
    for DL in 0.3 0.5; do
        submit "zs ${Z} delay ${DL}" "${CLIP}-p8" "${SMALL[@]}" \
            ZEROSUPPRESS="${Z}" ZEROSUPPRESS_DELAY="${DL}"
    done
done

# ==========================================================
section "M  rescues for everything that collapsed in sweep2"
# Each of these failed once. None of them failed with a reason established,
# and all of them ran with the pre-encoder off. Each arm below pairs the
# failed setting with a stated fix.

# lr 0.003 collapsed at full batch. Too large a step for one update per
# epoch; give it more, smaller updates instead of a smaller lr.
submit "lr 0.003 + batch 32" "${CLIP}-p8" "${SMALL[@]}" LR=0.003 BATCH=32 EPOCH=500

# max_temperature 5.0 left the latent soft for the whole run, because
# min_temperature floors at 0.7. Give it somewhere to anneal to.
submit "tmax 5.0 -> tmin 0.1 slow" "${CLIP}-p8" "${SMALL[@]}" \
    MAX_TEMPERATURE=5.0 MIN_TEMPERATURE=0.1 EPOCH=6000

# max_temperature 0.5 was annealing UPWARD into the 0.7 floor. Invert it.
submit "tmax 0.5 -> tmin 0.05" "${CLIP}-p8" "${SMALL[@]}" \
    MAX_TEMPERATURE=0.5 MIN_TEMPERATURE=0.05

# mo5 and mo8 both died at U=40. More object slots means more relations to
# represent, so give the extra objects extra units rather than assuming 40
# covers every scene size.
submit "mo5 + U60"  "${CLIP/-mo3-/-mo5-}-p8" "${MID[@]}" U=60 A=2 P=20
submit "mo8 + U80"  "${CLIP/-mo3-/-mo8-}-p8" "${MID[@]}" U=80 A=2 P=20
submit "mo8 + U80 P40" "${CLIP/-mo3-/-mo8-}-p8" "${MID[@]}" U=80 A=2 P=40

# nofill has far fewer states, so 2000 full-batch epochs is 2000 updates on
# a much smaller set. Give it more.
submit "nofill 8000 epochs" "${CLIP/-fill/-nofill}-p8" "${SMALL[@]}" EPOCH=8000

# dog-all kept 3 of 800 bits and dog-15vids kept 6. Both are the sparsity
# penalty biting a harder dataset. Delay it.
submit "dog all + zs delay"    "dog-${FPS}fps-all-mo5-fill-p8" "${BIG[@]}" \
    EPOCH=3000 ZEROSUPPRESS_DELAY=0.5 CATEGORY=dog
submit "dog 15vids + zs delay" "dog-15vids-${FPS}fps-mo3-fill-p8" "${MID[@]}" \
    ZEROSUPPRESS_DELAY=0.5 CATEGORY=dog

# ==========================================================
section "H  whole categories, winning configuration"
for SPEC in bird:3 monkey:4 horse:3 antelope:3 dog:5 car:5 person:5; do
    IFS=':' read -r CAT MO <<< "${SPEC}"
    submit "category ${CAT}" "${CAT}-${FPS}fps-all-mo${MO}-fill-p8" \
        "${BIG[@]}" EPOCH=3000 CATEGORY="${CAT}"
done

# ==========================================================
section "I  resolution where there is data volume"
bake "dog-${FPS}fps-all-mo5-fill-p32" dog --max-objects 5 --patch-size 32 --fill-annotations
submit "dog all p32"    "dog-${FPS}fps-all-mo5-fill-p32"    "${BIG[@]}" EPOCH=3000 CATEGORY=dog
submit "person all p32" "person-${FPS}fps-all-mo5-fill-p32" "${HUGE[@]}" EPOCH=3000 CATEGORY=person

# ==========================================================
section "J  across categories"
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
