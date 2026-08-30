#!/usr/bin/env bash
# The three new evaluation methods, on whatever exports you point them at.
#
#   bash experiments/M_evaluation_methods/run_local.sh                  # oracle
#   bash experiments/M_evaluation_methods/run_local.sh eval/exports/H14-*.npz
#
# Runs locally, needs numpy and the standard library, takes about a minute.
# Every step is memory-capped: an unbounded run crashed this workstation on
# 2026-08-28.
#
# Read `README.md` first. Both outcomes of each method are written down there
# BEFORE the run, so a result cannot be reinterpreted after the fact.

set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

ANN_DIR="${ANN_DIR:-data/video/vidvrd/annotations/train}"
OUT="${OUT:-eval/M_methods}"
MEM_KB="${MEM_KB:-8000000}"
HOLD_OUT="${HOLD_OUT:-bird}"
mkdir -p "${OUT}"

# Default corpus: the oracle exports built from the screened clips. Pass your
# own exports as arguments to score a trained model instead.
if [[ $# -gt 0 ]]; then
    EXPORTS=("$@")
else
    shopt -s nullglob
    EXPORTS=(eval/probe/batch/*.npz)
    shopt -u nullglob
fi

if (( ${#EXPORTS[@]} == 0 )); then
    echo "No exports. Build the oracle corpus first:" >&2
    echo "  bash experiments/M_evaluation_methods/build_oracle_corpus.sh" >&2
    exit 2
fi

echo "scoring ${#EXPORTS[@]} export(s)"

# The annotation for each export, matched by file stem.
ANNS=()
for f in "${EXPORTS[@]}"; do
    stem="$(basename "${f}" .npz)"
    [[ -f "${ANN_DIR}/${stem}.json" ]] && ANNS+=("${ANN_DIR}/${stem}.json")
done

echo
echo "=========================================="
echo "M1  does the latent encode the relations?"
echo "=========================================="
if (( ${#ANNS[@]} >= 2 )); then
    ( ulimit -v "${MEM_KB}"
      python3 tools/planner/predicate_probe.py "${EXPORTS[@]}" \
          --annotation "${ANNS[@]}" --out-dir "${OUT}/M1" )
else
    echo "SKIP: M1 needs at least two clips with annotations. Within one clip"
    echo "      most predicates never change, so the probe saturates."
fi

echo
echo "=========================================="
echo "M2  a rule, or memorised objects?"
echo "=========================================="
if (( ${#ANNS[@]} >= 3 )); then
    ( ulimit -v "${MEM_KB}"
      python3 tools/planner/compositional.py "${EXPORTS[@]}" \
          --annotations "${ANN_DIR}" --hold-out "${HOLD_OUT}" \
          --out-dir "${OUT}/M2" ) || echo "  (M2 could not form a split)"
else
    echo "SKIP: M2 needs at least three clips."
fi

echo
echo "=========================================="
echo "M3  is the plan admissible, with no truth?"
echo "=========================================="
( ulimit -v "${MEM_KB}"
  python3 tools/planner/plan_validity.py "${EXPORTS[@]}" --out-dir "${OUT}/M3" )

cat <<NOTE

==========================================
Figures, all saved to disk:
    ${OUT}/M1/probe.svg
    ${OUT}/M2/compositional.svg
    ${OUT}/M3/validity.svg

How to read each one is pre-registered in
experiments/M_evaluation_methods/README.md. Read it before deciding what the
numbers mean, not after.
NOTE
