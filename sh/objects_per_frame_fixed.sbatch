#!/usr/bin/env bash
#
# objects_per_frame_fixed.sbatch — DSAD and CholecT50, after the reader fixes.
#
#   sbatch --job-name=opf-fixed sh/objects_per_frame_fixed.sbatch
#
# WHY. An adversary found on 2026-09-27 that both readers were wrong. DSAD
# counted mask FILES, and 62.7 per cent of its 10,010 masks are entirely black,
# so "seven objects in every frame" was false. CholecT50 counted the string
# `null_target` as an entity in 10,079 frames. Both are fixed under TDD in
# workbench/tools/objects_per_frame.py, and this re-measures them.
#
# THE VENV AND NOT THE MODULE. The DSAD reader opens every mask with Pillow,
# which lives in the project venv. Python packages go in the venv and never
# global, so the job uses that interpreter.
#
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${ROOT}" || exit 2
PYTHONPATH="${ROOT}/workbench" "${ROOT}/.venv-local/bin/python" \
    "${ROOT}/workbench/tools/objects_per_frame.py" dsad cholect50 --root "${ROOT}"
