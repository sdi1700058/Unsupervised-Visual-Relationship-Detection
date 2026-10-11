#!/usr/bin/env bash
# Something-Else into VidVRD format: choose the clips, decode their frames.
#
# Plan workbench/notes/plans/2026-10-03-train-the-rungs.md, steps 1.2b and
# 1.2c. Reads three 1.1 GB box files, so it runs here and not on a login node.
# The template list comes from
#   python3 workbench/tools/something_else_relations.py \
#       --flipping-templates workbench/sh/rung_ids/se-flip-templates.txt
#
#   sbatch sh/convert_something_else.sbatch
#SBATCH --job-name=convert-se
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${ROOT}"
# Two steps: `module load system ffmpeg/7.1` on one line fails on Sherlock.
module load system
module load ffmpeg/7.1
RAW=data/video/something_else/raw
.venv-local/bin/python tools/video/convert_something_else.py \
    --boxes "${RAW}/bounding_box_smthsmth_part2.json" \
            "${RAW}/bounding_box_smthsmth_part3.json" \
            "${RAW}/bounding_box_smthsmth_part4.json" \
    --videos data/video/ssv2/ext/20bn-something-something-v2 \
    --labels data/video/ssv2/raw/labels.json \
    --splits data/video/ssv2/raw/train.json data/video/ssv2/raw/validation.json \
    --templates workbench/sh/rung_ids/se-flip-templates.txt \
    --out data/video/something_else/vidvrd_format \
    --count 250 --seed 0 \
    --domain "${RAW}/twentybn_domain.pddl" \
    --ids-out workbench/sh/rung_ids/se-250clips.txt
