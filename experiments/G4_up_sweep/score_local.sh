#!/usr/bin/env bash
# G4 -- score every cell of the U by P grid and draw it. Runs locally, CPU only.
#
#   bash experiments/G4_up_sweep/score_local.sh
#
# Needs the exports pulled from Sherlock. Writes:
#
#   eval/planner/G4_geometry.csv   the cheap screen
#   eval/planner/G4-U<u>-P<p>/     one planner run per cell
#   eval/planner/G4_summary.md     the table and the pre-registered reading
#   eval/planner/G4_summary.svg    the figure, both axes
#
# Memory capped throughout: an unbounded planner run over a wide latent crashed
# this workstation on 2026-08-28.

set -eo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PY="${PY:-.venv-local/bin/python}"
[[ -x "${PY}" ]] || PY=python3

# Window 16, not 8. The crossover criterion is a steep function of window size,
# and these 88 clips were screened at window 16; scoring at 8 measures
# something the selection never promised (SPEC V37). Override with a reason.
WINDOW="${WINDOW:-16}"
BUDGET="${BUDGET:-30}"
MAX_WINDOWS="${MAX_WINDOWS:-20}"
MEM_KB="${MEM_KB:-6000000}"     # 6 GB ceiling per planner process

shopt -s nullglob
EXPORTS=(eval/exports/U*_A*_P*_catH14-winnable*.npz)
shopt -u nullglob

if (( ${#EXPORTS[@]} == 0 )); then
    cat <<'MISSING' >&2
No G4 exports found.

Expected files matching:
    eval/exports/U*_A*_P*_catH14-winnable*.npz

Run experiments/G4_up_sweep/run_sherlock.sh on Sherlock first, then push the
exports and the training summary and pull them here:

    git add -f eval/exports/*catH14-winnable*.npz eval/exports/G4_train.csv
    git commit -m "G4 exports" && git push

See the README.
MISSING
    exit 2
fi

echo "found ${#EXPORTS[@]} cell(s)"
for f in "${EXPORTS[@]}"; do echo "  ${f}"; done
if [[ ! -f eval/exports/G4_train.csv ]]; then
    echo
    echo "NOTE: eval/exports/G4_train.csv is absent, so the figure falls back"
    echo "      to round-trip box error for its reconstruction axis. The"
    echo "      training-loss axis needs that file pushed from the cluster."
fi

echo
echo "=========================================="
echo "stage 1  latent geometry screen (seconds)"
echo "=========================================="
# A screen, not a ranking: it separates an arbitrary code from a positional
# one, but it correlates with planner error in the WRONG direction across
# resolutions, so it cannot order the cells (SPEC V26). Near-zero here means
# the code carries no position and planning it wastes an afternoon.
( ulimit -v "${MEM_KB}"
  "${PY}" tools/planner/latent_geometry.py "${EXPORTS[@]}" \
      --csv eval/planner/G4_geometry.csv ) || echo "  screen did not complete"

echo
echo "=========================================="
echo "stage 2  the planner, one cell at a time"
echo "=========================================="
# `bfs,pddl` and not `bfs` alone: BFS solved 0 of 14 windows on good data,
# because good annotation makes nearly every transition unique and blind search
# needs repetition to compose them. That is a statement about the search, not
# about the representation.
for NPZ in "${EXPORTS[@]}"; do
    STEM="$(basename "${NPZ}" .npz)"
    U="$(echo "${STEM}" | sed -n 's/^U\([0-9]\+\)_A[0-9]\+_P[0-9]\+_cat.*/\1/p')"
    P="$(echo "${STEM}" | sed -n 's/^U[0-9]\+_A[0-9]\+_P\([0-9]\+\)_cat.*/\1/p')"
    if [[ -z "${U}" || -z "${P}" ]]; then
        echo "  cannot read U and P from ${STEM}, skipping"
        continue
    fi
    NAME="G4-U${U}-P${P}"

    # Spread the windows over the whole export instead of taking the first
    # twenty. The export concatenates 88 clips in order, so the default stride
    # of K-1 would score only the first three or four of them and the grid
    # would compare cells on a corner of the data.
    STRIDE="$("${PY}" - "${NPZ}" "${WINDOW}" "${MAX_WINDOWS}" <<'PYEOF'
import sys
import numpy as np
path, window, cap = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
frames = int(np.load(path)["latents"].shape[0])
usable = max(1, frames - window + 1)
print(max(window - 1, usable // max(1, cap)))
PYEOF
)" || STRIDE="$(( WINDOW - 1 ))"

    echo
    echo "--- ${NAME}  (stride ${STRIDE})"
    ( ulimit -v "${MEM_KB}"
      bash tools/planner/eval_plannability.sh "${NPZ}" \
          --methods bfs,pddl \
          --window "${WINDOW}" \
          --stride "${STRIDE}" \
          --max-windows "${MAX_WINDOWS}" \
          --budget "${BUDGET}" \
          --length-mode max \
          --name "${NAME}" 2>&1 | tail -3 ) || echo "  ${NAME} did not complete"
done

echo
echo "=========================================="
echo "stage 3  the comparison and the figure"
echo "=========================================="
( ulimit -v "${MEM_KB}"
  "${PY}" experiments/G4_up_sweep/g4_summary.py )

echo
echo "=========================================="
echo "stage 4  per-cell reports"
echo "=========================================="
"${PY}" tools/planner/make_report.py eval/planner/G4-U*-P*/ --index || true

cat <<'NOTE'

The grid, both axes:   eval/planner/G4_summary.svg
The table and reading: eval/planner/G4_summary.md
Per-cell verdicts:     eval/planner/index.html

How to read the figure. The left panel is reconstruction and the right panel
is planning, over the same cells. If the two panels look alike, training loss
ranks the cells the way the planner does and model selection can skip the
planner. If they do not, a cell can reconstruct well and plan badly, which is
the claim in EVAL.md 5.7, and the bottom panel puts those cells in the shaded
corner.
NOTE
