#!/usr/bin/env bash
# smoke_test.sh — Verify FOSAE install: package import, config, path utils.
# Usage: bash smoke_test.sh
# For a full training check:
#   python3 model/train/strips.py learn puzzle FirstOrderAE mnist 3 3 None None None 100
#
# The AECLASS comes FIRST. model/train/strips.py::main dispatches positionally into
# puzzle(aeclass, type, width, height, U, A, P, num_examples), so dropping it
# shifts every argument left and the run dies on `import model.loaders.3`.
# The three Nones matter for the same reason: a bare `... mnist 3 3 100` sets
# U=100, not num_examples=100, and trains a 100-unit model on the full set.
# workbench/scripts/train/submit.sh:140 composes the same command the same way.

set -eo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

PASS=0; FAIL=0
ok()   { echo "  [PASS] $*"; PASS=$((PASS+1)); }
fail() { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }

echo "FOSAE smoke test"

python3 -c "import model; import model.model; print('  model:', model.__file__)" \
    && ok "model import" || fail "model import"

python3 -c "from model import config_cpu" \
    && ok "config_cpu import" || fail "config_cpu import"

python3 -c "from model.paths import DATA_DIR, OUT_DIR; print('  DATA_DIR:', DATA_DIR)" \
    && ok "paths import" || fail "paths import"

python3 -c "from model.loaders.vidvrd import PATCH_SIZE, MAX_OBJECTS, PICSIZE" \
    && ok "model.loaders.vidvrd import" || fail "model.loaders.vidvrd import"

python3 -c "from model.loaders.cache import npz_cache_path" \
    && ok "cache import" || fail "cache import"

echo ""
echo "Results: ${PASS} passed, ${FAIL} failed"
[[ ${FAIL} -eq 0 ]] && exit 0 || exit 1
