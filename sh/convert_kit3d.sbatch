#!/usr/bin/env bash
# KIT Bimanual from its 3D boxes: tracked instances and both hands, take 0 of
# the nine tasks of subjects 2 to 6, the 30 fps takes.
#
# Plan workbench/notes/plans/2026-10-05-dataset-fixes.md, F3. Subject 1 left
# because 83 of its 90 takes run at 15 fps. The RGB of subjects 3, 4 and 5
# is unpacked here first, as sh/unpack_kit_rgb.sbatch did for 2 and 6. Five
# subjects let SPLIT_GROUP hold out whole subjects: on 2026-10-05 subjects
# 2, 3 and 6 gave 66 windows, 27 of them from subject 6, so the default
# split would have trained on 21 windows of the test subject. Each take
# fits its own camera, and a take whose held-out error is above 10 px is
# dropped and named in the log.
#
#   sbatch sh/convert_kit3d.sbatch
#SBATCH --job-name=convert-kit3d
#SBATCH --partition=normal
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${ROOT}"
RAW=data/video/kit_bimanual/raw
RGB=data/video/kit_bimanual/rgb_ext
for s in 3 4 5; do
    unzip -q -n "${RAW}/bimacs_rgbd_data_subject_${s}.zip" \
        "bimacs_rgbd_data/subject_${s}/*/take_0/rgb/*" -d "${RGB}"
    echo "subject ${s}: $(find "${RGB}/bimacs_rgbd_data/subject_${s}" -name '*.png' | wc -l) png"
done
.venv-local/bin/python tools/video/convert_kit3d.py \
    --derived data/video/kit_bimanual/ext/bimacs_derived_data \
    --rgb "${RGB}/bimacs_rgbd_data" \
    --takes 2:task_1_k_cooking:0 2:task_2_k_cooking_with_bowls:0 2:task_3_k_pouring:0 2:task_4_k_wiping:0 2:task_5_k_cereals:0 2:task_6_w_hard_drive:0 2:task_7_w_free_hard_drive:0 2:task_8_w_hammering:0 2:task_9_w_sawing:0 3:task_1_k_cooking:0 3:task_2_k_cooking_with_bowls:0 3:task_3_k_pouring:0 3:task_4_k_wiping:0 3:task_5_k_cereals:0 3:task_6_w_hard_drive:0 3:task_7_w_free_hard_drive:0 3:task_8_w_hammering:0 3:task_9_w_sawing:0 4:task_1_k_cooking:0 4:task_2_k_cooking_with_bowls:0 4:task_3_k_pouring:0 4:task_4_k_wiping:0 4:task_5_k_cereals:0 4:task_6_w_hard_drive:0 4:task_7_w_free_hard_drive:0 4:task_8_w_hammering:0 4:task_9_w_sawing:0 5:task_1_k_cooking:0 5:task_2_k_cooking_with_bowls:0 5:task_3_k_pouring:0 5:task_4_k_wiping:0 5:task_5_k_cereals:0 5:task_6_w_hard_drive:0 5:task_7_w_free_hard_drive:0 5:task_8_w_hammering:0 5:task_9_w_sawing:0 6:task_1_k_cooking:0 6:task_2_k_cooking_with_bowls:0 6:task_3_k_pouring:0 6:task_4_k_wiping:0 6:task_5_k_cereals:0 6:task_6_w_hard_drive:0 6:task_7_w_free_hard_drive:0 6:task_8_w_hammering:0 6:task_9_w_sawing:0 \
    --out data/video/kit_bimanual/vidvrd_format_3d \
    --ids-out workbench/sh/rung_ids/kit3d-windows.txt
