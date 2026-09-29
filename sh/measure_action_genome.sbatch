#!/usr/bin/env bash
#
# measure_action_genome.sbatch — pairs, change and spacing of Action Genome
#
#   sbatch --job-name=ag-measure sh/measure_action_genome.sbatch
#
# WHY. Row 15 of THE-TABLE carries no number. The annotation files were
# fetched to data/video/actiongenome/raw/ on 2026-09-28. This reads them
# through the restricted unpickler, never pickle.load, and prints the pairs,
# the succession change rate, the frame spacing and the entities per frame.
#
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
RAW="${ROOT}/data/video/actiongenome/raw"
cd "${ROOT}/workbench" || exit 2
PYTHONPATH="..:${PYTHONPATH:-}" "${ROOT}/.venv-local/bin/python" \
    -m tools.action_genome_relations \
    "${RAW}/object_bbox_and_relationship.pkl" \
    --person "${RAW}/person_bbox.pkl" \
    --frame-rates "${ROOT}/data/video/charades/frame_rates.tsv" \
    --timestamps "${ROOT}/data/video/charades/timestamps.tsv"
