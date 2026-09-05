#!/usr/bin/env bash
# M4 — does latent distance track the number of actions between states?
#
#   bash experiments/M4_temporal_distance/run_local.sh
#   MAX_GAP=30 bash experiments/M4_temporal_distance/run_local.sh
#
# Local. numpy and the standard library, no model, no GPU and no video frames.
# Every step is memory-capped: an uncapped run crashed this workstation on
# 2026-08-28.
#
# Read README.md first. Both outcomes are written down there BEFORE the run,
# so a number cannot be reinterpreted after the fact.

set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

OUT="${OUT:-eval/M4}"
MAX_GAP="${MAX_GAP:-20}"
MEM_KB="${MEM_KB:-6000000}"
VIDVRD_DIR="${VIDVRD_DIR:-eval/probe/batch}"
VIDOR_DIR="${VIDOR_DIR:-eval/probe/vidor}"
MIN_CLIPS="${MIN_CLIPS:-10}"

mkdir -p "${OUT}"

count_npz() {
    ls -1 "$1"/*.npz 2>/dev/null | wc -l
}

# The two oracle corpora are the ceiling: latents built straight from
# ground-truth boxes, no model in the loop. They are built once and reused.
if [[ "$(count_npz "${VIDVRD_DIR}")" -lt "${MIN_CLIPS}" ]]; then
    echo "building the VidVRD oracle corpus (once, a few minutes)"
    OUT_DIR="${VIDVRD_DIR}" N_CLIPS=25 \
        bash experiments/M_evaluation_methods/build_oracle_corpus.sh
fi
if [[ "$(count_npz "${VIDOR_DIR}")" -lt "${MIN_CLIPS}" ]]; then
    echo "building the VidOR oracle corpus (once, a few minutes)"
    CLIPS_FILE=eval/vidor_winnable_w16.txt \
    ANN_DIR=data/video/vidor/annotations/training \
    OUT_DIR="${VIDOR_DIR}" N_CLIPS=25 \
        bash experiments/M_evaluation_methods/build_oracle_corpus.sh
fi

# Oracles first, so the figure reads ceiling-then-model.
ARGS=("vidvrd-oracle=${VIDVRD_DIR}" "vidor-oracle=${VIDOR_DIR}")

# The single clip invariant V35 was measured on, oracle beside trained. This
# is the only pair in the run that reproduces V35's own comparison exactly.
[[ -f eval/exports/oracle-150010-fixed.npz ]] && \
    ARGS+=("V35clip-oracle=eval/exports/oracle-150010-fixed.npz")
[[ -f eval/exports/H14-P10-150010.npz ]] && \
    ARGS+=("V35clip-trainedP10=eval/exports/H14-P10-150010.npz")

# Every H14 arm, over all 88 of its clips rather than its best one. V36
# records that the H14 headline was taken on the model's best clip.
shopt -s nullglob
for f in eval/exports/U*_catH14-winnable88-*.npz; do
    stem="$(basename "${f}" .npz)"
    arm="${stem%%_cat*}"
    ARGS+=("trained-${arm}=${f}")
done
shopt -u nullglob

echo
echo "=========================================="
echo "M4  latent distance against the frame gap"
echo "=========================================="
( ulimit -v "${MEM_KB}"
  python3 tools/planner/m4_temporal.py "${ARGS[@]}" \
      --max-gap "${MAX_GAP}" --out-dir "${OUT}" )

cat <<NOTE

==========================================
Figure and numbers, saved to disk:
    ${OUT}/m4_temporal.svg
    ${OUT}/m4_temporal.json

How to read them is pre-registered in
experiments/M4_temporal_distance/README.md. Read it before deciding what the
numbers mean, not after.
NOTE
