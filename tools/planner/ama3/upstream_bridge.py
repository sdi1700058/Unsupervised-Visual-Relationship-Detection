#!/usr/bin/env python3
"""tools/planner/ama3/upstream_bridge.py — SPEC.md §T H2c.

Insert upstream latplan on `sys.path` at runtime per C18, then verify that
`latplan.util.planner` is importable.

**Where upstream lives is a property of the machine.** Until 2026-09-06 this
was an absolute literal under one home directory with no override, so every
ama3 run on the cluster failed on a path that exists on one workstation.
`install_roswell.sh`, in this same directory, already honoured
`${UPSTREAM_LATPLAN:-...}`; two files that must agree about one directory
disagreed about whether it was configurable.

`UPSTREAM_LATPLAN` in the environment wins. Without it the default is the
sibling `latplan` beside this repository, which is how both machines are laid
out, so the common case needs no variable.

Read-only per C14: this module NEVER edits upstream.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))

UPSTREAM_LATPLAN = os.environ.get(
    "UPSTREAM_LATPLAN", os.path.join(os.path.dirname(_REPO), "latplan"))


def ensure_upstream_on_path():
    """Insert upstream latplan on sys.path if not already there."""
    if UPSTREAM_LATPLAN not in sys.path:
        sys.path.insert(0, UPSTREAM_LATPLAN)


def check_upstream_planner_util():
    """Smoke: import `latplan.util.planner::setup_planner_utils`.

    Returns True on success, False on ImportError.
    """
    ensure_upstream_on_path()
    try:
        from latplan.util.planner import setup_planner_utils   # noqa: F401
        return True
    except ImportError as e:
        print(f"[upstream_bridge] ImportError: {e}", file=sys.stderr)
        return False


def upstream_dir(sub=""):
    """Return a path under the upstream latplan checkout."""
    return os.path.join(UPSTREAM_LATPLAN, sub) if sub else UPSTREAM_LATPLAN


if __name__ == "__main__":
    ok = check_upstream_planner_util()
    print("OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)
