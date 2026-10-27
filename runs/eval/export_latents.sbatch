#!/usr/bin/env bash
# Export the latents, bindings and decoded boxes of one trained run.
#
# WHY IT NEEDS A JOB. `planner/export_latents.py` holds the whole state
# array in memory: the KIT role clips are 29,078 states of 3 slots and 6,916
# features, about 2.4 GB in float32 before TensorFlow takes its share. It was
# killed at exit 137 twice inside an 8 GB allocation on 2026-10-10 and
# 2026-10-11. The encoder and the autoencoder now run in batches, which caps
# the model's own peak, but the state array itself is loaded whole.
#
#   sbatch --export=ALL,RUN=out/learnability/zs0/rep1/video/vidvrd/<dir> \
#          runs/eval/export_latents.sbatch
#
# Inputs:
#   RUN    the run directory, the one holding net0.h5 and loaded_videos.json
#   OUT    the npz to write, default <RUN>/planner_export.npz
#   BATCH  states per call to the model, default 512
#SBATCH --job-name=export-latents
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -z "${RUN:-}" ]]; then
    echo "FATAL: RUN is not set. See the header of this file." >&2
    exit 2
fi
if [[ ! -f "${RUN}/net0.h5" ]]; then
    echo "FATAL: no net0.h5 in ${RUN}" >&2
    exit 2
fi

OUT="${OUT:-${RUN}/planner_export.npz}"
BATCH="${BATCH:-512}"

echo "run   ${RUN}"
echo "out   ${OUT}"
echo "batch ${BATCH}"

venv/bin/python planner/export_latents.py "${RUN}" \
    -o "${OUT}" --batch "${BATCH}"

echo "wrote ${OUT}"
ls -la "${OUT}"
