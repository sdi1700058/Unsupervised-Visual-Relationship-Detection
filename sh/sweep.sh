#!/usr/bin/env bash
# sweep.sh - remove what a test run leaves behind, and nothing else.
#
# Two kinds of clutter, and both are made by running the suites:
#
#   1. The temporary directories of tempfile.mkdtemp under $TMPDIR, which
#      sh/claude_confined.sh puts at <repo>/.claude/tmp.
#   2. The __pycache__ directories under workbench/tools.
#
# The plan this came from asked for the `tmp` prefix alone. The author widened
# it to `slotid-` as well on 2026-09-18, because that prefix names 6264
# directories from the same defect in an older public fixture, and no other
# command in this project reaches them. Do not narrow it back without asking.
#
# Measured 2026-09-18, before the leaking fixtures were repaired: 8119
# directories and 44 MB in .claude/tmp, and one run of the private suite added
# 9 more. tools/verify.py layer 2 attributes a result by a diff of the tree, so
# a directory that no unit wrote is a false positive that it must discount.
#
# The boundary is the workspace, which is $SCRATCH/panos. The account belongs
# to the author's brother, and notes/docs/SANDBOX.md gives the reason. This
# script refuses every path outside it, and it refuses any path that holds a
# git directory. Run it with --dry-run to read the list before it acts.
#
#   bash sh/sweep.sh --dry-run
#   bash sh/sweep.sh
#
# Do not run it while a suite runs. A directory of a live test looks the same
# as one that a dead test left.

set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd "$here/.." && pwd -P)

# The workspace comes from the environment where the environment gives it, so
# that no account name is written into this public repository. The parent of
# the repository is the fallback, and the repository must sit inside whichever
# of the two answers.
if [ -n "${SCRATCH:-}" ] && [ -d "$SCRATCH/panos" ]; then
    workspace=$(cd "$SCRATCH/panos" && pwd -P)
else
    workspace=$(cd "$repo/.." && pwd -P)
fi

case "$repo/" in
    "$workspace"/*) ;;
    *)  echo "sweep: the repository at $repo is outside $workspace." >&2
        echo "sweep: nothing was removed." >&2
        exit 2 ;;
esac

tmp_root="$repo/.claude/tmp"
pycache_root="$repo/workbench/tools"

dry=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry=1 ;;
        -h|--help)
            sed -n '2,25p' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *)  echo "sweep: unknown argument $arg" >&2
            exit 2 ;;
    esac
done

# A path is swept only when it resolves inside the workspace and holds no git
# directory. The second test is the one that matters: workbench/.git holds
# commits that are on no remote, and $SCRATCH has no backup.
safe_to_remove() {
    local path="$1" real
    real=$(realpath -m -- "$path")
    case "$real/" in
        "$workspace"/*) ;;
        *)  echo "sweep: REFUSED, outside the workspace: $real" >&2
            return 1 ;;
    esac
    case "$real/" in
        */.git/*) echo "sweep: REFUSED, inside a git directory: $real" >&2
                  return 1 ;;
    esac
    if [ -e "$real/.git" ]; then
        echo "sweep: REFUSED, holds a git directory: $real" >&2
        return 1
    fi
    return 0
}

removed=0
refused=0
listed=0

sweep_one() {
    local path="$1"
    if ! safe_to_remove "$path"; then
        refused=$((refused + 1))
        return 0
    fi
    if [ "$dry" -eq 1 ]; then
        echo "would remove  $path"
        listed=$((listed + 1))
        return 0
    fi
    rm -r "$path"
    removed=$((removed + 1))
}

# mkdtemp names a directory from its prefix and eight characters of [a-z0-9_].
# The pattern is exact, so the named directories that sit beside them in
# .claude/tmp stay. `slotid-` is the prefix that the public loader fixture uses.
if [ -d "$tmp_root" ]; then
    while IFS= read -r path; do
        [ -n "$path" ] || continue
        sweep_one "$path"
    done < <(find "$tmp_root" -maxdepth 1 -mindepth 1 -type d \
                  -regextype posix-extended \
                  -regex '.*/(tmp|slotid-)[A-Za-z0-9_]{8}$' | sort)
fi

if [ -d "$pycache_root" ]; then
    while IFS= read -r path; do
        [ -n "$path" ] || continue
        sweep_one "$path"
    done < <(find "$pycache_root" -type d -name '__pycache__' | sort)
fi

if [ "$dry" -eq 1 ]; then
    echo "sweep: --dry-run, $listed to remove, $refused refused, 0 removed."
else
    echo "sweep: $removed removed, $refused refused."
fi

if [ "$refused" -gt 0 ]; then
    exit 1
fi
