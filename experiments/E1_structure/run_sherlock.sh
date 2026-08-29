#!/usr/bin/env bash
# E1 — does innate structure predict plannability?  See README.md.
#
# Bakes two matched arms, trains an identical configuration on each, and
# exports both for the planner. Nothing here is interactive and nothing needs
# editing: paste the sbatch line and leave.
#
#   cd $SCRATCH/panos/sgg-thesis && git pull
#   mkdir -p logs && sbatch experiments/E1_structure/run_sherlock.sh
#
#SBATCH --job-name=fosae-E1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/E1.%j.out
#SBATCH --error=logs/E1.%j.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs

source venv/bin/activate 2>/dev/null || source activate.sh
source sh/sweep_lib.sh

FPS="${FPS:-30}"
NPZ="data/npz/video/vidvrd/overfit"

# The two arms, built by tools/video/screen_vidvrd.py and pinned here so the
# experiment is reproducible without re-deriving them. Each unstructured clip
# is matched to a structured one on motion, to within 0.5 px per step.
A_IDS="ILSVRC2015_train_00211000,ILSVRC2015_train_00897009,ILSVRC2015_val_00015001,ILSVRC2015_train_00548000,ILSVRC2015_val_00035008,ILSVRC2015_train_00574002,ILSVRC2015_train_00058003,ILSVRC2015_train_00181011,ILSVRC2015_val_00159002,ILSVRC2015_train_00025022,ILSVRC2015_train_00010006"
B_IDS="ILSVRC2015_train_00211004,ILSVRC2015_train_01081001,ILSVRC2015_train_01052000,ILSVRC2015_val_00037004,ILSVRC2015_train_00987000,ILSVRC2015_train_01081000,ILSVRC2015_train_00065002,ILSVRC2015_train_00057003,ILSVRC2015_val_00036008,ILSVRC2015_val_00028003,ILSVRC2015_train_00069006"

# The configuration that learned: pre-encoder on. Identical for both arms, so
# the arms differ only in their data. --fill-annotations is deliberately
# ABSENT: every clip here is fully annotated, and filling would fabricate
# transitions (SPEC B35, V24).
section "E1  bake both arms, no fill"

bake_arm () {
    local name="$1" ids="$2"
    if [[ -f "${NPZ}/${name}.npz" ]]; then echo "have  ${name}"; return 0; fi
    echo "bake  ${name}"
    if python3 setup-dataset.py video_vidvrd all \
           --video-id "${ids}" --fps "${FPS}" \
           --max-objects 3 --patch-size 8 --out-name "${name}"; then
        NBAKED=$((NBAKED + 1))
    else
        echo "  BAKE FAILED ${name}"; NFAILED=$((NFAILED + 1))
    fi
}

bake_arm "E1-structured"   "${A_IDS}"
bake_arm "E1-unstructured" "${B_IDS}"

section "E1  train, identical configuration on each arm"

SWEEP_DEFAULTS=(EPOCH=2000 LR=0.001 BATCH=1000
                PREENC_LAYERS=2 PREENC_DIM=1000
                MAX_TEMPERATURE=1.0 TRANSITION_MODE=sequential
                U=40 A=2 P=10)

submit "A structured"   "E1-structured"   32G 4:00:00
submit "B unstructured" "E1-unstructured" 32G 4:00:00

sweep_totals

cat <<'NOTE'

Both arms submitted. When they finish, export them and push:

    for d in out/video/vidvrd/*E1-structured* out/video/vidvrd/*E1-unstructured*; do
        [ -d "$d" ] && sbatch sh/export_model.sh "$d"
    done
    # then, once the exports exist
    git add -f eval/exports/*E1-*.npz && git commit -m "E1 exports" && git push

Then locally:  bash experiments/E1_structure/score_local.sh
NOTE
