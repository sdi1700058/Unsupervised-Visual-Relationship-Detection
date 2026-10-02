#!/usr/bin/env bash
#
# measure_kit_actions.sbatch — KIT's human action-and-object labels against its boxes
#
#   sbatch --job-name=kit-actions sh/measure_kit_actions.sbatch
#
# See workbench/tools/kit_actions.py. Reads the cereals-task labels of KIT's
# temporal-task-model release and the 3D object boxes KIT ships.
#
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
RAW="${ROOT}/data/video/kit_bimanual/raw"
cd "${ROOT}/workbench" || exit 2
PYTHONPATH="..:${PYTHONPATH:-}" "${ROOT}/.venv-local/bin/python" -m tools.kit_actions \
    --labels "${RAW}/bimacs_ttm_rgbd_data_ground_truth.zip" \
    --objects "${RAW}/bimacs_derived_data_3d_objects.zip"
