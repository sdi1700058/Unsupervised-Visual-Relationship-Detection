#!/usr/bin/env bash
# Export every H14 model that finished. Chained off sh/h14.sh by --dependency,
# so it is not normally submitted by hand.
#
# The glob has to be evaluated HERE rather than in h14.sh, because the run
# directories do not exist until the training jobs have started.
#
# Export cannot run on a login node: TensorFlow fails to build its thread pool
# against the login node's process limit and aborts. See sh/export_model.sh.
#
#SBATCH --job-name=fosae-H14-export
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/H14-export.%j.out
#SBATCH --error=logs/H14-export.%j.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs eval/exports

shopt -s nullglob
DIRS=(out/video/vidvrd/*H14-winnable*)
shopt -u nullglob

if (( ${#DIRS[@]} == 0 )); then
    echo "no H14 run directories under out/video/vidvrd/ -- every arm failed" >&2
    exit 1
fi

echo "exporting ${#DIRS[@]} H14 run(s)"
exec bash sh/export_model.sh "${DIRS[@]}"
