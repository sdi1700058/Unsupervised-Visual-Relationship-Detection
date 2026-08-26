#!/usr/bin/env python3
"""tools/planner/ama3/upstream_bridge.py — SPEC.md §T H2c.

Insert `/home/panoslat/Dev/Thesis/FOSAE/latplan/` on `sys.path` at runtime
per C18, then verify that `latplan.util.planner` is importable.

Read-only per C14: this module NEVER edits upstream.
"""

import os
import sys

UPSTREAM_LATPLAN = "/home/panoslat/Dev/Thesis/FOSAE/latplan"


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
    """Return path under `/home/panoslat/Dev/Thesis/FOSAE/latplan/`."""
    return os.path.join(UPSTREAM_LATPLAN, sub) if sub else UPSTREAM_LATPLAN


if __name__ == "__main__":
    ok = check_upstream_planner_util()
    print("OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)
