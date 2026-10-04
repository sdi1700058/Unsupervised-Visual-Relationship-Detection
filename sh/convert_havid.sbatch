#!/usr/bin/env bash
# HA-ViD into VidVRD format: two 150-frame windows per M0 video.
#
# Plan workbench/notes/plans/2026-10-03-train-the-rungs.md, steps 1.3b and
# 1.3c. Reads the 443 MB train COCO file, so it runs here and not on a login
# node. Each M0 video is decoded whole, once, with no seek, and its count is
# checked against the CSV and against data/video/ha_vid/frame_rates.tsv.
#
#   sbatch sh/convert_havid.sbatch
#SBATCH --job-name=convert-havid
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
    --ids-out workbench/sh/rung_ids/havid-m0-windows.txt
