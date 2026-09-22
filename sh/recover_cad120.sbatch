#!/usr/bin/env bash
#
# recover_cad120.sbatch — salvage what the Wayback Machine truncated.
#
# All four CAD-120 rgbd_images archives fail `gzip -t` with "unexpected end of
# file", measured 2026-09-22. They are real gzip streams with 2013 timestamps
# and they stop early, because the Wayback Machine truncates very large
# captures. A truncated gzip still decompresses up to the cut, so the frames
# before the cut are recoverable and the ones after are gone.
#
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --output=logs/cad120rec-%j.out
#SBATCH --error=logs/cad120rec-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:?}"
RAW="${ROOT}/data/video/cad120/raw"
OUT="${ROOT}/data/video/cad120/hf"
mkdir -p "${OUT}"
for f in "${RAW}"/Subject*_rgbd_images.tar.gz; do
    b=$(basename "$f")
    echo "== ${b} =="
    # tar exits non-zero on the truncated tail and that is expected. What
    # matters is how many members landed before the cut.
    gzip -dc "$f" 2>/dev/null | tar xf - -C "${OUT}" 2>/dev/null
    echo "   cumulative files: $(find "${OUT}" -type f | wc -l)"
done
echo "TOTAL files   : $(find "${OUT}" -type f | wc -l)"
echo "TOTAL png/jpg : $(find "${OUT}" -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l)"
echo "TOTAL bytes   : $(du -sb "${OUT}" | cut -f1)"
find "${OUT}" -type f | head -5
