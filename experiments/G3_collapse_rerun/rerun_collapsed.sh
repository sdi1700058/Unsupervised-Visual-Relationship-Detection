#!/usr/bin/env bash
# G3_collapse_rerun - retrain the five configurations whose export collapsed.
#
# Run the preflight first. It is seconds and it refuses for a named reason:
#
#     bash experiments/G3_collapse_rerun/preflight.sh
#
# Then, one command, and walk away:
#
#     cd $SCRATCH/panos/sgg-thesis && mkdir -p logs
#     sbatch experiments/G3_collapse_rerun/rerun_collapsed.sh
#
# The export is chained with --dependency, so there is no second visit. Score
# it afterwards with:
#
#     .venv-local/bin/python tools/planner/liveness.py --exports eval/exports --strict
#
# README.md beside this file holds the hypothesis and what each outcome means.
# Both readings were written before the run.
#
# NOTHING IS BAKED HERE. The dataset this needs already exists, because it
# produced the collapsed exports. A missing one is a preflight failure rather
# than a four-hour surprise, and workbench/sh/h14.sh is the script that bakes.
#SBATCH --job-name=fosae-G3-rerun
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/G3-rerun.%j.out
#SBATCH --error=logs/G3-rerun.%j.err

set -eo pipefail

# Resolve the public root, which holds strips.py, setup-dataset.py, data/ and
# venv/. Three things make this harder than it looks, and each one broke a run:
#
#   1. This file sits at workbench/sh/ since the split of 2026-09-10, so the
#      old `dirname/..` resolved to the PRIVATE root and every relative path
#      below missed by one level.
#   2. Under sbatch, ${BASH_SOURCE[0]} is NOT this path. Slurm copies the
#      batch script to /var/spool/slurmd/job<id>/slurm_script, so walking up
#      from it finds no repository at all. Measured 2026-09-18, job 44184992:
#      "FATAL: no public root above /var/spool/slurmd/job44184992".
#   3. SLURM_SUBMIT_DIR is wherever sbatch was typed, which is usually but not
#      always inside the tree.
#
# So try every candidate and take the first that has a public root above it.
_find_root () {
    local d r
    for d in "$@"; do
        [ -n "${d}" ] || continue
        [ -d "${d}" ] || continue
        r="$(cd "${d}" && until [ -f strips.py ] || [ "${PWD}" = / ]; do cd ..; done; pwd)"
        if [ -f "${r}/strips.py" ]; then echo "${r}"; return 0; fi
    done
    return 1
}
if ! PROJECT_DIR="$(_find_root "${SLURM_SUBMIT_DIR:-}" \
                                "$(dirname "${BASH_SOURCE[0]}")" \
                                "${PWD}" "${THESIS:-}")"; then
    echo "FATAL: no public root (no strips.py) above SLURM_SUBMIT_DIR," >&2
    echo "       this script, the working directory or \$THESIS." >&2
    exit 2
fi
# From the root, not from BASH_SOURCE: under sbatch that is the spool copy.
SH_DIR="${PROJECT_DIR}/workbench/sh"
cd "${PROJECT_DIR}"
mkdir -p logs

source venv/bin/activate 2>/dev/null || true
source "${SH_DIR}/sweep_lib.sh"

FPS="${FPS:-30}"
NPZ="data/npz/video/vidvrd/overfit"
STEM="H14-winnable88-30fps-mo3-nofill-p8"

if [[ ! -f "${NPZ}/${STEM}.npz" ]]; then
    echo "FATAL: ${NPZ}/${STEM}.npz is absent, and this script does not bake." >&2
    echo "       Run: sbatch workbench/sh/h14.sh" >&2
    exit 3
fi

# ── keep the collapsed runs, because they are the before half ──────────────
#
# A run directory is keyed by a hash of its parameters, so an identical
# configuration writes to the identical directory and overwrites it. That is
# what makes the comparison honest -- same name, fresh draw -- and it is also
# what would destroy the evidence. Move the five aside first.
ARCHIVE="out/video/vidvrd/collapsed-$(date +%Y-%m-%d)"
mkdir -p "${ARCHIVE}"
section "G3  archive the five collapsed run directories"
for arm in U20_A2_P10 U40_A2_P20 U10_A2_P10 U20_A2_P20 U5_A2_P5; do
    for d in out/video/vidvrd/FirstOrderSAE_${arm}_catH14-winnable88-*; do
        [[ -d "${d}" ]] || continue
        if [[ -e "${ARCHIVE}/$(basename "${d}")" ]]; then
            echo "  already archived: $(basename "${d}")"
        else
            echo "  ${d} -> ${ARCHIVE}/"
            mv "${d}" "${ARCHIVE}/"
        fi
    done
done

section "G3  retrain the five collapsed configurations"

# Identical to H14's, because the question is whether the configuration or the
# draw produced the collapse. Changing anything else answers a question nobody
# asked. Nothing in this project pins a random seed, so a re-run IS a fresh
# draw -- and it is for that same reason not reproducible. README.md records
# that as the limit of this experiment.
SWEEP_DEFAULTS=(EPOCH=3000 LR=0.001 BATCH=1000
                PREENC_LAYERS=2 PREENC_DIM=1000
                MAX_TEMPERATURE=1.0 TRANSITION_MODE=sequential
                A=2)

MID=(32G 6:00:00); BIG=(48G 10:00:00)

# Bits = U * P. The 800-bit arm took BIG in H14 and a wrong TIME costs a
# night, so both 400-bit-and-above P20 arms take it here.
submit "G3 U20 P10  (200 bits)" "${STEM}" "${MID[@]}"  U=20 P=10
submit "G3 U10 P10  (100 bits)" "${STEM}" "${MID[@]}"  U=10 P=10
submit "G3 U5  P5   (25 bits)"  "${STEM}" "${MID[@]}"  U=5  P=5
submit "G3 U20 P20  (400 bits)" "${STEM}" "${BIG[@]}"  U=20 P=20
submit "G3 U40 P20  (800 bits)" "${STEM}" "${BIG[@]}"  U=40 P=20

sweep_totals

if (( ${#SUBMITTED_IDS[@]} == 0 )); then
    echo "nothing submitted, so nothing to export" >&2
    exit 1
fi

DEP="$(IFS=:; echo "${SUBMITTED_IDS[*]}")"
echo
echo "chaining export after ${#SUBMITTED_IDS[@]} training job(s): ${DEP}"
# afterany, not afterok: an arm that dies must not withhold the arms that
# lived. h14_export.sh globs *H14-winnable* and exports whatever landed, so it
# re-exports the nine that already live too. Those are the same models in the
# same directories, so the export is identical and the overwrite is harmless.
sbatch --dependency="afterany:${DEP}" "${SH_DIR}/h14_export.sh"

cat <<'NOTE'

Submitted. Nothing else to do here. Watch it with:

    squeue --me

When the queue is empty, the one number this experiment turns on:

    .venv-local/bin/python tools/planner/liveness.py --exports eval/exports --strict

It read 7 of 31 dead on 2026-09-18. README.md says what each new reading
means, and both readings were written before the run.
NOTE
