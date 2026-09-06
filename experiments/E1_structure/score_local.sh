#!/usr/bin/env bash
# E1 — score both arms and print the comparison. Runs locally, CPU only.
#
#   bash experiments/E1_structure/score_local.sh
#
# Needs the two exports pulled from Sherlock. Writes eval/planner/E1_summary.md
# and eval/planner/E1_summary.svg, both via tools/planner/e1_summary.py.

set -eo pipefail

# Window 16, not 8. The crossover criterion depends steeply on window size and
# these clips were screened at a larger window than 8; scoring at 8 measures
# something the selection never promised (SPEC V37). Override with WINDOW=.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PY="${PY:-.venv-local/bin/python}"
[[ -x "${PY}" ]] || PY=python3

A=$(ls eval/exports/*E1-structured*.npz 2>/dev/null | head -1 || true)
B=$(ls eval/exports/*E1-unstructured*.npz 2>/dev/null | head -1 || true)

if [[ -z "${A}" || -z "${B}" ]]; then
    cat <<'MISSING' >&2
Both E1 exports are needed and at least one is absent.

Expected something matching:
    eval/exports/*E1-structured*.npz
    eval/exports/*E1-unstructured*.npz

Run experiments/E1_structure/run_sherlock.sh first, export the two models, and
pull them here. See the README.
MISSING
    exit 1
fi

echo "structured   ${A}"
echo "unstructured ${B}"
echo

# Cheap screen first: latent geometry needs no planner and takes seconds. A
# code that does not order frames like the world will not plan, whatever its
# training data, so this can settle the question before any search runs.
echo "=== latent geometry (screen, no planning) ==="
"${PY}" tools/planner/latent_geometry.py "${A}" "${B}" || true
echo

# 6 GB and a short budget: an unbounded search over a wide latent can exhaust
# memory, and this has crashed a workstation before.
# Scoped by the callers below rather than applied to the whole script, which
# is what sh/h14_score.sh does. Pick one convention and this is it.
MEM_KB="${MEM_KB:-6000000}"
for pair in "structured:${A}" "unstructured:${B}"; do
    name="${pair%%:*}"; path="${pair#*:}"
    echo "=== planning: ${name} ==="
    # The cap is applied here, in a subshell. It used to be defined above and
    # never used, so every E1 planner run was uncapped -- the exact condition
    # that crashed this workstation once.
    ( ulimit -v "${MEM_KB}"
    bash tools/planner/eval_plannability.sh "${path}" \
        --methods bfs,pddl --window "${WINDOW:-16}" --budget 30 --name "E1-${name}" \
        2>&1 | tail -3 )
    echo
done

# The comparison lives in tools/planner/e1_summary.py, not in a heredoc here.
#
# It used to be inline, and that inline version reported the OPPOSITE of what
# the data said. It counted CSV rows rather than windows, so two planners on
# one window read as two windows and it called 80 windows 160. It then compared
# the two arms' raw `bbox_mse` medians as though they described comparable
# samples, when unstructured had reached only 4 of 58 windows — its easiest 7%,
# measured against the whole of the other arm.
#
# The module fixes both: it keys windows by (init, goal) and credits each
# window with the better of the two planners, it compares `mse_ratio` and never
# raw error across arms, and it reports solve rate first with a selection-bias
# caveat whenever an arm reached under half its windows. It also carries the
# same pre-registered 1.38x volume confound as VOLUME_CONFOUND.
#
# G1 and G4 already import it. E1 is the experiment it was extracted from and
# was the last caller still running the version it replaced.
"${PY}" tools/planner/e1_summary.py
