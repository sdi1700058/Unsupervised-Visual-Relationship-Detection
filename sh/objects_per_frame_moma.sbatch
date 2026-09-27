#!/usr/bin/env bash
#
# objects_per_frame_moma.sbatch — MOMA-LRG's per-snapshot entity histogram.
#
#   sbatch --job-name=opf-moma sh/objects_per_frame_moma.sbatch
#
# WHY A JOB FOR ONE DATASET. `workbench/tools/objects_per_frame.py` carries a
# reader for four datasets and three of them run on a login node in seconds.
# MOMA's `anns.json` is 551 MB, which is the same shape of work that had to be
# killed on a login node on 2026-09-23.
#
# WHAT IT FEEDS. `tools/slot_cap_survival.py` turns a histogram into the share
# of PAIRS lost under the pipeline's three-object cap, which is the column
# `notes/THE-TABLE.md` needs. The frame-overflow share alone misleads: a
# dataset overflowing 81 per cent of frames loses 71 per cent of its pairs.
#
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${ROOT}" || exit 2
module load python/3.12.1
PYTHONPATH="${ROOT}/workbench" python3 "${ROOT}/workbench/tools/objects_per_frame.py" moma --root "${ROOT}"
