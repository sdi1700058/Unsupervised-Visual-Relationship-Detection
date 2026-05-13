#!/usr/bin/env bash
# build_video_npz.sh — sbatch driver to materialize a per-category video npz cache.
#
# Usage:
#   DATASET=vidvrd CATEGORY=bicycle FPS=3 sbatch sh/build_video_npz.sh
#   DATASET=actiongenome CATEGORY=chair FPS=3 sbatch sh/build_video_npz.sh
#
# Env vars (defaults):
#   DATASET   — vidvrd | actiongenome             (required)
#   CATEGORY  — e.g. bicycle, chair               (required)
#   FPS       — 3                                 (must match an extracted frames_<FPS>fps/ dir)
#   SPLIT     — train | test                       (default: train)
#
#SBATCH --job-name=fosae-npz
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -eo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/sh/sherlock_env.sh"

: "${DATASET:?DATASET env var required (vidvrd | actiongenome)}"
: "${CATEGORY:?CATEGORY env var required}"
FPS="${FPS:-3}"
SPLIT="${SPLIT:-train}"

case "${DATASET}" in
    vidvrd)        MOD="latplan.puzzles.puzzle_vidvrd"       ;;
    actiongenome)  MOD="latplan.domains.video.actiongenome"  ;;
    *) echo "ERROR: unknown DATASET=${DATASET}" >&2; exit 2 ;;
esac

echo "[build_video_npz] DATASET=${DATASET} CATEGORY=${CATEGORY} FPS=${FPS} SPLIT=${SPLIT}"

python3 -c "
import time, importlib
p = importlib.import_module('${MOD}')
t = time.time()
p.build_dataset(category_filter='${CATEGORY}', fps=${FPS}, split='${SPLIT}')
print('elapsed', round(time.time()-t, 2))
"

OUT="data/npz/video/${DATASET}/${CATEGORY}-${FPS}fps.npz"
[[ -f "${OUT}" ]] && echo "[build_video_npz] OK -> ${OUT} ($(du -h "${OUT}" | cut -f1))" \
                  || { echo "ERROR: cache file ${OUT} not written" >&2; exit 3; }
