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

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"

# Where the thesis repository is. The checks that read source and run tests
# operate on it, and after the working infrastructure was split out on
# 2026-09-05 the two live in separate directories. When this script sits
# inside the thesis repository itself, HERE is already the answer.
THESIS="${THESIS:-}"
if [[ -z "${THESIS}" ]]; then
    if [[ -d "${HERE}/tools/planner/tests" ]]; then
        THESIS="${HERE}"
    elif [[ -d "${HERE}/../labeled-fosae/tools/planner/tests" ]]; then
        THESIS="$(cd "${HERE}/../labeled-fosae" && pwd)"
    elif [[ -d "${HERE}/../sgg-thesis/tools/planner/tests" ]]; then
        THESIS="$(cd "${HERE}/../sgg-thesis" && pwd)"
    else
        echo "cannot find the thesis repository. Set THESIS=/path/to/it."
        exit 2
    fi
fi

# Run from the thesis repository, not from wherever this script happens to
# live. Every check below invokes its tool as a RELATIVE path (tools/*.py) and
# reads relative inputs (eval/exports), so they all resolve against the current
# directory rather than against THESIS. While the two are the same directory
# that is invisible; once the working infrastructure is split out it is not.
# The two conditional checks are the dangerous half: `[[ -f
# tools/check_headlines.py ]]` and `[[ -d eval/exports ]]` would simply be
# false in the other repository, and a check that quietly does not run reads
# exactly like a check that passed.
cd "${THESIS}"

PY="${THESIS}/.venv-local/bin/python"
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
run "tests"              "${PY}" -m unittest discover \
                             -s "${THESIS}/tools/planner/tests"
run "python 3.6"         python3 tools/check_py36.py --root "${THESIS}"

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
# A generated cross-reference block goes stale silently: the prose still reads
# as current and nothing in the document says it was machine-written. The
# review found one carrying 26 of 28 shortlisted papers while the gate was
# green, because nothing ran this.
run "generated blocks"   python3 tools/lit/render_index.py --check

# Headlines and liveness are conditional: one needs its script, the other needs
# exports on disk. A check that cannot run says so rather than passing.
if [[ -f tools/check_headlines.py ]]; then
    run "headlines"      python3 tools/check_headlines.py
else
    printf '\n=== headlines\n    skipped: tools/check_headlines.py is not here\n'
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
