#!/usr/bin/env bash
# KIT Bimanual: take the RGB frames of take 0 of every task, subjects 1, 2, 6.
#
# Plan workbench/notes/plans/2026-10-03-train-the-rungs.md, step 2.1. The six
# RGB-D archives are 13 to 20 GB each and hold depth too; only the RGB of
# the chosen takes is extracted, 27 takes and 11,267 frames by the count of
# the 2d_objects files. Subject 6 sorts last, so the clip split holds it out.
#
#   sbatch sh/unpack_kit_rgb.sbatch
#SBATCH --job-name=unpack-kit-rgb
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${ROOT}"
RAW=data/video/kit_bimanual/raw
OUT=data/video/kit_bimanual/rgb_ext
mkdir -p "${OUT}"
for s in 1 2 6; do
    unzip -q -n "${RAW}/bimacs_rgbd_data_subject_${s}.zip" \
        "bimacs_rgbd_data/subject_${s}/*/take_0/rgb/*" -d "${OUT}"
    echo "subject ${s}: $(find "${OUT}/bimacs_rgbd_data/subject_${s}" -name '*.png' | wc -l) png"
done
