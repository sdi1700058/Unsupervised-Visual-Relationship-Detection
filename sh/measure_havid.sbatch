#!/usr/bin/env bash
#
# measure_havid.sbatch — HA-ViD's assembly relations and its boxes per frame
#
#   sbatch --job-name=havid-measure sh/measure_havid.sbatch
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
    -m tools.havid_relations \
    --temporal "${EXT}/HAViD_temporalAnnotation" \
    --detection "${EXT}/ObjectDetection"
