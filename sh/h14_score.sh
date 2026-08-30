#!/usr/bin/env bash
# Score the H14 exports on the workstation, and say whether the thesis
# question is answered.
#
#     bash sh/h14_score.sh
#
# Two stages, cheapest first. The screen costs seconds and the planner costs
# minutes per arm, so the screen goes first (SPEC V22).
#
# Runs entirely on the local machine and is capped so it cannot take the
# machine down: an unbounded planner run crashed this workstation once
# already, on 2026-08-28.

set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

WINDOW="${WINDOW:-8}"
BUDGET="${BUDGET:-60}"
MAX_WINDOWS="${MAX_WINDOWS:-14}"
MEM_KB="${MEM_KB:-8000000}"     # 8 GB ceiling per planner process

shopt -s nullglob
EXPORTS=(eval/exports/*H14*.npz)
shopt -u nullglob

if (( ${#EXPORTS[@]} == 0 )); then
    echo "no eval/exports/*H14*.npz found." >&2
    echo "Did the Sherlock exports get committed and pushed, and did you pull?" >&2
    exit 2
fi

echo "=========================================="
echo "stage 1  latent geometry screen (seconds)"
echo "=========================================="
# A screen, not a ranking: it separates an arbitrary code from a positional
# one, but it correlates with planner error in the WRONG direction across
# resolutions, so it cannot order the arms (SPEC V26). Near-zero here means
# the code carries no position and planning it is a waste of an afternoon.
python3 tools/planner/latent_geometry.py "${EXPORTS[@]}" \
    --csv eval/planner/h14_geometry.csv

echo
echo "=========================================="
echo "stage 2  the planner, one arm at a time"
echo "=========================================="
# One process at a time, memory-capped, window count capped. `bfs,pddl` and
# not `bfs` alone: BFS solved 0 of 14 windows on good data, because good
# annotation makes nearly every transition unique and blind search needs
# repetition to compose them. That is a statement about the search, not about
# the representation.
for NPZ in "${EXPORTS[@]}"; do
    NAME="H14-$(basename "${NPZ}" .npz)"
    echo
    echo "--- ${NAME}"
    ( ulimit -v "${MEM_KB}"
      bash tools/planner/eval_plannability.sh "${NPZ}" \
          --methods bfs,pddl \
          --window "${WINDOW}" \
          --max-windows "${MAX_WINDOWS}" \
          --budget "${BUDGET}" \
          --length-mode max \
          --name "${NAME}" ) || echo "  ${NAME} did not complete"
done

echo
echo "=========================================="
echo "stage 3  reports"
echo "=========================================="
python3 tools/planner/make_report.py eval/planner/H14-*/ --index || true

cat <<'NOTE'

Read the verdict at:  eval/planner/index.html
Charts per arm at:    eval/planner/H14-*/chart.svg

How to read it, against the oracle on the same task:

    oracle    planner error 12.0   mse_ratio 0.046   14/14 solved
    floor      9.91  -- no representation at this resolution beats this

  mse_ratio < 1     a trained FOSAE beats linear interpolation. The thesis
                    has a positive answer.
  1 <= ratio < 3    it carries real signal but does not beat the straight
                    line. A qualified answer, and the honest one to report.
  ratio >= 3, or
  few windows       the representation does not support planning. That is a
    solved          result too, and the oracle number proves it is the model
                    and not the task.
NOTE
