#!/usr/bin/env python3
"""A resumable ledger for reviewing every artefact in the project.

**Why a ledger and not a pass.** The review covers 154 Python files, 46 shell
scripts, 32 documents and the plan, which is far more than fits in one sitting.
This session has been killed by usage limits repeatedly, so any design that
loses its place when the process dies is useless. The state lives on disk and
each artefact is recorded before the next is started.

**Four passes, each asking a different question.**

    infrastructure  does each gate actually fail when it should
    plan            does every unit and claim rest on something real
    code            does the file do what it says, is it reachable, is it tested
    documents       does every number trace to a file, every path exist

`infrastructure` and `plan` run first. They are small, they decide whether the
foundations hold, and a defect there can stop the work on its own.

    python3 tools/review.py build
    python3 tools/review.py next --which infrastructure
    python3 tools/review.py progress
    python3 tools/review.py report

Standard library only, Python 3.6 clean.
"""

import argparse
import glob
import json
import os
import subprocess
import sys

LEDGER = "notes/review_ledger.json"

# Ordered worst first, so a sort by index puts the serious findings on top.
SEVERITIES = ("high", "medium", "low", "note")

# The order the passes run in. Infrastructure and plan decide whether the
# foundations hold, so they come first and can stop the work on their own.
PASSES = ("infrastructure", "plan", "code", "documents")


def build(items, existing=None):
    """A ledger over `items`, each a `(path, pass)` pair.

    Rebuilding keeps whatever was already reviewed. New files appear as
    unreviewed and everything already done stays done, so the ledger can be
    regenerated after the tree changes without discarding the work.
    """
    done = {}
    for entry in (existing or {}).get("entries", []):
        if entry.get("status") == "reviewed":
            done[entry["path"]] = entry

    entries = []
    for path, which in items:
        if path in done:
            entries.append(done[path])
            continue
        entries.append({"path": path, "pass": which, "status": "unreviewed",
                        "verdict": "", "findings": []})
    return {"entries": entries}


def next_batch(ledger, size, which=None):
    """The next unreviewed entries, optionally from one pass only."""
    out = []
    for entry in ledger.get("entries", []):
        if entry.get("status") == "reviewed":
            continue
        if which is not None and entry.get("pass") != which:
            continue
        out.append(entry)
        if len(out) >= size:
            break
    return out


def record(ledger, path, findings, verdict):
    """Mark one artefact reviewed, with its findings.

    Refuses an unknown severity, because a severity nobody defined cannot be
    sorted or counted, and refuses an empty verdict, because an entry with no
    verdict is not a review.
    """
    if not verdict:
        raise ValueError("a verdict is required; an entry without one is not "
                         "a review")
    for finding in findings or []:
        severity = finding.get("severity")
        if severity not in SEVERITIES:
            raise ValueError("unknown severity %r; use one of %s"
                             % (severity, ", ".join(SEVERITIES)))
    for entry in ledger.get("entries", []):
        if entry["path"] == path:
            entry["status"] = "reviewed"
            entry["verdict"] = verdict
            entry["findings"] = list(findings or [])
            return entry
    raise ValueError("no entry for %r" % path)


def progress(ledger):
    """`{pass: {reviewed, total}}`, plus `_findings` counted by severity."""
    out = {}
    counts = dict((s, 0) for s in SEVERITIES)
    for entry in ledger.get("entries", []):
        row = out.setdefault(entry["pass"], {"reviewed": 0, "total": 0})
        row["total"] += 1
        if entry.get("status") == "reviewed":
            row["reviewed"] += 1
        for finding in entry.get("findings") or []:
            counts[finding["severity"]] = counts.get(finding["severity"], 0) + 1
    out["_findings"] = counts
    return out


def save(ledger, path=LEDGER):
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w") as handle:
        json.dump(ledger, handle, indent=2)


def load(path=LEDGER):
    if not os.path.isfile(path):
        return {"entries": []}
    with open(path) as handle:
        return json.load(handle)


# --------------------------------------------------------------------------
# what there is to review
# --------------------------------------------------------------------------

def _tracked(pattern):
    try:
        out = subprocess.check_output(["git", "ls-files", pattern],
                                      stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, OSError):
        return []
    return [p for p in out.decode("utf-8", "replace").split("\n") if p]


def discover():
    """Every artefact, as `(path, pass)` pairs, in the order the passes run.

    The infrastructure pass reviews checks rather than files, so its entries
    are named for the check and not for a path on disk.
    """
    items = []

    # Each gate, reviewed by planting a violation and confirming it is caught.
    for check in ("tests", "python 3.6", "documents", "withdrawn numbers",
                  "glossary", "plan consistency", "unit verification",
                  "dataset order", "headlines", "export liveness"):
        items.append(("gate: %s" % check, "infrastructure"))

    # The plan, reviewed as its parts rather than as one file.
    for part in ("units", "claims", "observations", "assumptions",
                 "questions", "combinations", "milestones", "decisions"):
        items.append(("plan: %s" % part, "plan"))

    for path in sorted(set(_tracked("*.py") + _tracked("*.sh"))):
        items.append((path, "code"))

    docs = sorted(set(glob.glob("notes/*.md") + glob.glob("notes/docs/*.md")
                      + glob.glob("notes/docs/*/*.md") + ["README.md"]))
    for path in docs:
        if os.path.isfile(path):
            items.append((path, "documents"))
    return items


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["build", "next", "progress", "report"])
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--which", default=None, choices=list(PASSES))
    ap.add_argument("--size", type=int, default=6)
    a = ap.parse_args(argv)

    ledger = load(a.ledger)

    if a.command == "build":
        ledger = build(discover(), ledger)
        save(ledger, a.ledger)
        rows = progress(ledger)
        print("%d artefact(s) to review" % len(ledger["entries"]))
        for name in PASSES:
            if name in rows:
                print("  %-15s %d" % (name, rows[name]["total"]))
        return 0

    if a.command == "next":
        # Follow the pass order unless one is named, so the foundations are
        # reviewed before the bulk.
        batch = []
        if a.which:
            batch = next_batch(ledger, a.size, a.which)
        else:
            for name in PASSES:
                batch = next_batch(ledger, a.size, name)
                if batch:
                    break
        if not batch:
            print("nothing left to review")
            return 0
        print("%s pass, %d artefact(s):" % (batch[0]["pass"], len(batch)))
        for entry in batch:
            print("  %s" % entry["path"])
        return 0

    if a.command == "progress":
        rows = progress(ledger)
        total = sum(r["total"] for k, r in rows.items() if k != "_findings")
        done = sum(r["reviewed"] for k, r in rows.items() if k != "_findings")
        for name in PASSES:
            if name not in rows:
                continue
            row = rows[name]
            share = (100.0 * row["reviewed"] / row["total"]) if row["total"] \
                else 0.0
            print("  %-15s %4d of %4d   %3.0f%%"
                  % (name, row["reviewed"], row["total"], share))
        print("  %-15s %4d of %4d" % ("TOTAL", done, total))
        found = rows.get("_findings", {})
        if any(found.values()):
            print("\n  findings: %s"
                  % ", ".join("%d %s" % (found[s], s) for s in SEVERITIES
                              if found.get(s)))
        return 0

    if a.command == "report":
        rows = []
        for entry in ledger.get("entries", []):
            for finding in entry.get("findings") or []:
                rows.append((SEVERITIES.index(finding["severity"]),
                             entry["path"], finding))
        rows.sort(key=lambda r: (r[0], r[1]))
        if not rows:
            print("no findings recorded yet")
            return 0
        for _, path, finding in rows:
            print("[%-6s] %s" % (finding["severity"], path))
            print("         %s" % finding.get("what", ""))
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
