#!/usr/bin/env python3
"""Catch Python 3.7+ syntax on the code path that runs on the cluster.

Sherlock pins python/3.6.1 (sh/sherlock_config.sh) because that is what
TensorFlow 1.15 needs. The workstation runs something much newer, so a
3.7-only line looks fine locally and fails only after a git push, a pull on
the cluster, a job submission and a wait. That has now happened twice:

    from __future__ import annotations          -> SyntaxError on 3.6
    subprocess.run(..., capture_output=True)    -> TypeError on 3.6

Both were found by a failed Sherlock run rather than by anything local. This
turns that into a check that costs a second:

    python3 tools/check_py36.py            # scan the cluster-side path
    python3 tools/check_py36.py --all      # scan every .py in the repo

Exits non-zero when something would break, so it also works in a hook.

Only the cluster-side path is scanned by default. The planner's scoring half
(`tools/planner/{bfs,pddl}`, `common/metrics.py`) runs on the workstation
under a modern interpreter and is deliberately out of scope.
"""

import argparse
import ast
import os
import sys

# Everything here is imported by a job running under python/3.6.1.
CLUSTER_PATHS = (
    "setup-dataset.py",
    "strips.py",
    "extract_fol.py",
    "config.py",
    "latplan",
    "tools/video",
    "tools/planner/export_latents.py",
    "tools/planner/common/encode.py",
    "tools/planner/common/decode.py",
    "tools/planner/common/windows.py",
)

# name -> the version that introduced it
LATE_ATTRS = {
    "cached_property": "3.8 (functools)",
    "prod": "3.8 (math.prod)",
}
LATE_MODULES = {
    "dataclasses": "3.7",
    "contextvars": "3.7",
    "importlib.metadata": "3.8",
}
LATE_KWARGS = {
    "capture_output": "3.7 (subprocess.run)",
    "text": "3.7 (subprocess.run)",
}
BUILTIN_GENERICS = {"list", "dict", "set", "tuple", "frozenset", "type"}


class Scanner(ast.NodeVisitor):

    def __init__(self, path):
        self.path = path
        self.findings = []

    def _report(self, node, what, since):
        self.findings.append((getattr(node, "lineno", 0), what, since))

    # -- syntax ----------------------------------------------------------
    def visit_ImportFrom(self, node):
        if node.module == "__future__":
            for a in node.names:
                if a.name == "annotations":
                    self._report(node, "from __future__ import annotations",
                                 "3.7")
        elif node.module in LATE_MODULES:
            self._report(node, f"import from {node.module}",
                         LATE_MODULES[node.module])
        for a in node.names:
            if a.name in LATE_ATTRS:
                self._report(node, f"{node.module}.{a.name}",
                             LATE_ATTRS[a.name])
        self.generic_visit(node)

    def visit_Import(self, node):
        for a in node.names:
            if a.name in LATE_MODULES:
                self._report(node, f"import {a.name}", LATE_MODULES[a.name])
        self.generic_visit(node)

    def visit_NamedExpr(self, node):          # walrus, 3.8
        self._report(node, "walrus operator :=", "3.8")
        self.generic_visit(node)

    def visit_arguments(self, node):
        if getattr(node, "posonlyargs", None):
            self._report(node, "positional-only parameters (/)", "3.8")
        self.generic_visit(node)

    # -- calls -----------------------------------------------------------
    def visit_Call(self, node):
        for kw in node.keywords:
            if kw.arg in LATE_KWARGS and _is_subprocess_run(node.func):
                self._report(node, f"{kw.arg}= on subprocess.run",
                             LATE_KWARGS[kw.arg])
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "prod":
            if isinstance(f.value, ast.Name) and f.value.id == "math":
                self._report(node, "math.prod", "3.8")
        self.generic_visit(node)

    # -- annotations -----------------------------------------------------
    def visit_AnnAssign(self, node):
        self._check_annotation(node.annotation)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        for a in list(node.args.args) + list(node.args.kwonlyargs):
            if a.annotation is not None:
                self._check_annotation(a.annotation)
        if node.returns is not None:
            self._check_annotation(node.returns)
        self.generic_visit(node)

    def _check_annotation(self, node):
        # PEP 604 unions, X | Y
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            self._report(node, "PEP 604 union in an annotation (X | Y)",
                         "3.10")
        # PEP 585 builtin generics, list[int]
        if isinstance(node, ast.Subscript):
            base = node.value
            if isinstance(base, ast.Name) and base.id in BUILTIN_GENERICS:
                self._report(node,
                             f"PEP 585 generic in an annotation ({base.id}[...])",
                             "3.9")
        for child in ast.iter_child_nodes(node):
            self._check_annotation(child)


def _is_subprocess_run(func):
    """True for subprocess.run(...) and a bare run(...) after a from-import."""
    if isinstance(func, ast.Attribute):
        return func.attr in ("run", "check_output")
    if isinstance(func, ast.Name):
        return func.id in ("run", "check_output")
    return False


def iter_py_files(targets):
    for t in targets:
        if os.path.isfile(t) and t.endswith(".py"):
            yield t
        elif os.path.isdir(t):
            for root, dirs, files in os.walk(t):
                dirs[:] = [d for d in dirs
                           if d not in (".git", "__pycache__", "venv", ".venv")]
                for f in sorted(files):
                    if f.endswith(".py"):
                        yield os.path.join(root, f)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Flag Python 3.7+ syntax on the cluster-side code path.")
    ap.add_argument("--all", action="store_true",
                    help="scan the whole repository, not only the cluster path")
    ap.add_argument("paths", nargs="*", help="override the paths to scan")
    args = ap.parse_args(argv)

    targets = args.paths or (["."] if args.all else list(CLUSTER_PATHS))
    targets = [t for t in targets if os.path.exists(t)]

    total, scanned = 0, 0
    for path in iter_py_files(targets):
        try:
            src = open(path, encoding="utf-8").read()
            tree = ast.parse(src, filename=path)
        except (OSError, SyntaxError) as e:
            print(f"{path}: could not parse: {e}")
            total += 1
            continue
        scanned += 1
        s = Scanner(path)
        s.visit(tree)
        for lineno, what, since in sorted(s.findings):
            print(f"{path}:{lineno}: {what} — needs Python {since}")
            total += 1

    if total:
        print(f"\n{total} finding{'' if total == 1 else 's'} across "
              f"{scanned} file{'' if scanned == 1 else 's'}. "
              "Sherlock runs python/3.6.1.")
        return 1
    print(f"{scanned} files scanned, nothing needs more than Python 3.6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
