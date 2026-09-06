#!/usr/bin/env bash
# smoke_test.sh — Verify FOSAE install: package import, config, path utils.
# Usage: bash smoke_test.sh
# For a full training check:
#   python3 strips.py learn puzzle FirstOrderAE mnist 3 3 None None None 100
#
# The AECLASS comes FIRST. strips.py::main dispatches positionally into
# puzzle(aeclass, type, width, height, U, A, P, num_examples), so dropping it
# shifts every argument left and the run dies on `import latplan.puzzles.puzzle_3`.
# The three Nones matter for the same reason: a bare `... mnist 3 3 100` sets
# U=100, not num_examples=100, and trains a 100-unit model on the full set.
# sh/submit.sh:89 composes the same command the same way.

set -eo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

PASS=0; FAIL=0
ok()   { echo "  [PASS] $*"; PASS=$((PASS+1)); }
fail() { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }

echo "FOSAE smoke test"

python3 -c "import latplan; import latplan.model; print('  latplan:', latplan.__file__)" \
    && ok "latplan import" || fail "latplan import"

python3 -c "import config_cpu" \
    && ok "config_cpu import" || fail "config_cpu import"

python3 -c "from latplan.util.paths import DATA_DIR, OUT_DIR; print('  DATA_DIR:', DATA_DIR)" \
    && ok "paths import" || fail "paths import"

python3 -c "from latplan.puzzles.puzzle_vidvrd import PATCH_SIZE, MAX_OBJECTS, PICSIZE" \
    && ok "puzzle_vidvrd import" || fail "puzzle_vidvrd import"

python3 -c "from latplan.domains.video.actiongenome import build_dataset; from latplan.util.cache import npz_cache_path" \
    && ok "actiongenome + cache import" || fail "actiongenome + cache import"

echo ""
echo "Results: ${PASS} passed, ${FAIL} failed"
[[ ${FAIL} -eq 0 ]] && exit 0 || exit 1
