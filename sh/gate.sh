#!/bin/bash
#
# Every check, in one command.
#
#     bash sh/gate.sh
#     bash sh/gate.sh --quick     # the cheap half only, for a fast loop
#
# "The gate passed" becomes a fact rather than a claim, because this exits
# non-zero when anything fails and names what.
#
# The tests run under .venv-local and never under bare python3. The bare
# interpreter lacks pillow and SILENTLY SKIPS the tests that need it, and a
# skipped test read as a pass for weeks while one of them had always failed.

set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY=".venv-local/bin/python"
if [[ ! -x "${PY}" ]]; then
    echo "WARNING: ${PY} is missing, falling back to python3."
    echo "         Tests needing pillow will be SKIPPED, and a skipped test"
    echo "         is not a passing test."
    PY="python3"
fi

QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

FAILED=0
declare -a BROKEN=()

run () {
    local name="$1"; shift
    printf '\n=== %s\n' "${name}"
    if "$@"; then
        printf '    ok\n'
    else
        printf '    FAILED\n'
        FAILED=$((FAILED + 1))
        BROKEN+=("${name}")
    fi
}

# The cheap half. Seconds, and it runs every time.
run "tests"              "${PY}" -m unittest discover -s tools/planner/tests
run "python 3.6"         python3 tools/check_py36.py

if (( QUICK )); then
    printf '\n==========================================\n'
    if (( FAILED == 0 )); then
        printf 'the quick half passed. Run without --quick before committing.\n'
        exit 0
    fi
    printf '%d check(s) failed: %s\n' "${FAILED}" "${BROKEN[*]}"
    exit 1
fi

# The full half. Each of these guards a failure that actually happened here.
run "documents"          python3 tools/check_docs.py
run "withdrawn numbers"  python3 tools/check_superseded.py
run "glossary"           python3 tools/check_glossary.py
run "plan consistency"   python3 tools/workplan.py check
run "unit verification"  python3 tools/workplan.py verify
run "dataset order"      python3 tools/score_datasets.py --check-order

# Headlines and liveness are conditional: one needs its script, the other needs
# exports on disk. A check that cannot run says so rather than passing.
if [[ -f tools/check_headlines.py ]]; then
    run "headlines"      python3 tools/check_headlines.py
fi
if [[ -d eval/exports ]]; then
    run "export liveness" "${PY}" tools/planner/liveness.py --exports eval/exports
else
    printf '\n=== export liveness\n    skipped: no eval/exports on this machine\n'
fi

printf '\n==========================================\n'
if (( FAILED == 0 )); then
    printf 'every check passed\n'
    exit 0
fi
printf '%d check(s) failed: %s\n' "${FAILED}" "${BROKEN[*]}"
exit 1
