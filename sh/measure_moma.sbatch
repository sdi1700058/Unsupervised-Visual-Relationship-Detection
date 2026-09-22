#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --output=logs/moma-%j.out
#SBATCH --error=logs/moma-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:?}"
cd "${ROOT}" || exit 2
python3 workbench/tools/measure_moma.py data/video/moma_lrg/raw/anns
