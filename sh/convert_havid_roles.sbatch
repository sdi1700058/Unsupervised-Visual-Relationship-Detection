#!/usr/bin/env bash
# HA-ViD with role-based slot selection: one clip for each primitive task of
# the boxed M0 videos, holding only the acting hand, the moved object and the
# target object. Plan workbench/notes/plans/2026-10-05-dataset-fixes.md, F4.
# The clips share the decoded frames of sh/convert_havid.sbatch, and their
# annotations go to annotations_roles beside the window clips.
#
#   sbatch sh/convert_havid_roles.sbatch
#
# IDS_OUT defaults to a .rerun.txt path, not the tracked havid-roles.txt: a
# rerun diffs its output against the tracked list by hand before replacing it.
#SBATCH --job-name=convert-havid-roles
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${ROOT}"
# Two steps: `module load system ffmpeg/7.1` on one line fails on Sherlock.
module load system
module load ffmpeg/7.1
.venv-local/bin/python tools/video/convert_havid.py \
    --ext data/video/ha_vid/ext \
    --frame-rates data/video/ha_vid/frame_rates.tsv \
    --out data/video/ha_vid/vidvrd_format \
    --ids-out "${IDS_OUT:-workbench/sh/rung_ids/havid-roles.rerun.txt}" \
    --roles
