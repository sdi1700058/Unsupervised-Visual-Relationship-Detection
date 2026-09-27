#!/usr/bin/env bash
#
# lock_board.sbatch — the dataset lock board, on a compute node.
#
#   sbatch --job-name=lockboard sh/lock_board.sbatch
#
# WHY A JOB NOW. On 2026-09-27 the board exceeded 120 seconds on a login node
# after five candidates were added. Its axis-2 column sums the bytes of every
# payload directory, and `ext` joined PAYLOAD_DIRS the same day, which brings
# in KIT Bimanual's 663,865 extracted files among others. Summing that is the
# intensive I/O the cluster rules send to the scheduler.
#
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${ROOT}" || exit 2
module load python/3.12.1
python3 workbench/tools/lock_board.py
