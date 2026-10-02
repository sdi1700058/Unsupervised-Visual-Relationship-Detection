#!/usr/bin/env bash
#
# measure_havid_solutions.sbatch — hand-object changes, and whether 3 slots hold each join
#
#   sbatch --job-name=havid-solutions sh/measure_havid_solutions.sbatch
#
# Reads the annotations that sh/unpack_havid_annotations.sbatch extracted.
# See workbench/tools/havid_relations.py for the code and the rule.
#
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
EXT="${ROOT}/data/video/ha_vid/ext"
cd "${ROOT}/workbench" || exit 2
PYTHONPATH="..:${PYTHONPATH:-}" "${ROOT}/.venv-local/bin/python" \
    -m tools.havid_relations --solutions \
    --temporal "${EXT}/HAViD_temporalAnnotation" \
    --detection "${EXT}/ObjectDetection"
