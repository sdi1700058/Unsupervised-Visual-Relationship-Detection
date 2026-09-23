#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/endo-unpack-%j.out
#SBATCH --error=logs/endo-unpack-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:?}"
cd "${ROOT}" || exit 2
RAW="data/video/endoscapes/raw"
EXT="data/video/endoscapes/ext"
mkdir -p "${EXT}"
for z in endoscapes cholecT50; do
  echo "== ${z} =="
  unzip -q -o "${RAW}/${z}.zip" -d "${EXT}/" && echo "   ok"
  echo "   files now: $(find "${EXT}" -type f | wc -l)"
done
echo "TOTAL: $(find "${EXT}" -type f | wc -l)"
find "${EXT}" -maxdepth 2 -name '*.json' | head -20
