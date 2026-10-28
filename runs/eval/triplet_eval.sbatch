#!/usr/bin/env bash
# Triplet mAP of the four rungs, in parallel, in one job.
#
# WHY ONE JOB. The code is the same for every rung and only the inputs
# change (report B.4, author decision of 2026-10-25). Something-Else runs
# first, alone. This job then scores SE, KIT, HA-ViD and VidVRD at the same
# time, one task for each rung, and writes into one directory.
#
# WHY THESE LIMITS. Measured on 2026-10-25 in a temporary run on a compute
# node: KIT takes 41 s and 0.57 GB, and each other rung takes less than 3 s.
# 30 minutes and 4 GB for each task leave a wide margin. CPU only.
#
# WHY --export=ALL ON srun. An interactive allocation sets
# SLURM_EXPORT_ENV=NONE, and a job submitted from it inherits that. A step
# then sees PATH=/usr/local/bin:/usr/bin and nothing else (measured
# 2026-10-25). The python path below is relative, so it does not need PATH.
#
#   sbatch --export=ALL,OUT=workbench/eval/triplet/2026-10-25 \
#          runs/eval/triplet_eval.sbatch
#
# Inside an allocation with at least 4 CPUs, the same file runs as four
# steps of that allocation, with no new job:
#
#   SLURM_SUBMIT_DIR=$PWD OUT=workbench/eval/triplet/2026-10-25 \
#       bash runs/eval/triplet_eval.sbatch
#
# Inputs:
#   OUT   the output directory, default workbench/eval/triplet/<today>
#SBATCH --job-name=triplet-eval
#SBATCH --partition=normal
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

OUT="${OUT:-workbench/eval/triplet/$(date +%F)}"
export TMPDIR="${PWD}/.claude/tmp"
mkdir -p "${OUT}" logs "${TMPDIR}"

echo "out   ${OUT}"
echo "job   ${SLURM_JOB_ID:-none}"

pids=()
for rung in se kit havid vidvrd; do
    srun --ntasks=1 --cpus-per-task=1 --exact --export=ALL \
        --output="logs/triplet-eval-%j-${rung}.out" \
        --error="logs/triplet-eval-%j-${rung}.err" \
        .venv-local/bin/python metrics/triplet_eval.py \
        --dataset "${rung}" --out "${OUT}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done

ls -la "${OUT}"
echo "exit ${status}"
exit "${status}"
