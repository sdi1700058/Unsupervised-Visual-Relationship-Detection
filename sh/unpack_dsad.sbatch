#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/dsad-unpack-%j.out
#SBATCH --error=logs/dsad-unpack-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
EXT=data/video/dsad/ext
mkdir -p "${EXT}"
# ONLY the multilabel folder. The other eleven hold ONE structure per image
# and therefore no object pair, per the Nature paper's own sentence.
unzip -q -o data/video/dsad/raw/38494425 'multilabel/*' -d "${EXT}/" 2>/dev/null \
  || unzip -q -o data/video/dsad/raw/38494425 -d "${EXT}/"
echo "files: $(find "${EXT}" -type f | wc -l)"
find "${EXT}" -maxdepth 2 -type d | head -20
find "${EXT}" -type f -name '*.png' | head -8
