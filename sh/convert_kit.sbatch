#!/usr/bin/env bash
# KIT Bimanual into VidVRD format: every whole 150-frame window of take 0 of
# the nine tasks, subjects 1, 2 and 6.
#
# Plan workbench/notes/plans/2026-10-03-train-the-rungs.md, step 2.3. Needs
# the RGB that sh/unpack_kit_rgb.sbatch extracted. Converts 11,267 PNG frames
# to JPG once per take; windows are symlinks into those.
#
#   sbatch sh/convert_kit.sbatch
#SBATCH --job-name=convert-kit
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${ROOT}"
.venv-local/bin/python tools/video/convert_kit.py \
    --derived data/video/kit_bimanual/ext/bimacs_derived_data \
    --rgb data/video/kit_bimanual/rgb_ext/bimacs_rgbd_data \
    --takes 1:task_1_k_cooking:0 1:task_2_k_cooking_with_bowls:0 1:task_3_k_pouring:0 1:task_4_k_wiping:0 1:task_5_k_cereals:0 1:task_6_w_hard_drive:0 1:task_7_w_free_hard_drive:0 1:task_8_w_hammering:0 1:task_9_w_sawing:0 2:task_1_k_cooking:0 2:task_2_k_cooking_with_bowls:0 2:task_3_k_pouring:0 2:task_4_k_wiping:0 2:task_5_k_cereals:0 2:task_6_w_hard_drive:0 2:task_7_w_free_hard_drive:0 2:task_8_w_hammering:0 2:task_9_w_sawing:0 6:task_1_k_cooking:0 6:task_2_k_cooking_with_bowls:0 6:task_3_k_pouring:0 6:task_4_k_wiping:0 6:task_5_k_cereals:0 6:task_6_w_hard_drive:0 6:task_7_w_free_hard_drive:0 6:task_8_w_hammering:0 6:task_9_w_sawing:0 \
    --out data/video/kit_bimanual/vidvrd_format \
    --ids-out workbench/sh/rung_ids/kit-windows.txt
