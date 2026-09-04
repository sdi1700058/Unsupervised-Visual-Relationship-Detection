#!/usr/bin/env python3
"""Find withdrawn numbers that still read as current.

**Why this exists.** On 2026-08-30 four claims were written up as findings and
then disproved. Each was corrected where it was found. A scripted sweep the
next day showed **nine** occurrences of those figures still sitting in the
reference documents, and two of them would not have been caught by reading:

- a **duplicate annotation**, two different wordings one line apart, leaving a
  reader unsure which was authoritative;
- a **restatement three sections away** from the section that had been marked,
  quoting "14 of 14" as a current fact while discussing something else.

Marking the source does not mark the echo. So the sweep is kept rather than
re-derived, and it runs as part of the quality gate.

**Withdrawn numbers are not deleted from the documents.** Each superseded block
keeps its figures under a banner giving the specific reason they are wrong,
because how a plausible measurement misled is part of the record. This tool
only checks that the banner is there.

    python3 tools/check_superseded.py
    python3 tools/check_superseded.py --paths notes/docs/EVAL.md

Exits non-zero when an occurrence is uncovered.

Standard library only, Python 3.6 clean.
"""

import argparse
import glob
import os
import re
import sys


# value -> why it was withdrawn. The reason is printed with any hit, so a
# reader learns why rather than only that.
SUPERSEDED = {
    "0.046": "pre-review mse_ratio; the Hungarian pairing was solved against "
             "absent boxes and the oracle floor used the wrong dequantiser",
    "0.082": "single-clip mse_ratio at window 8; that clip is a 33x outlier "
             "on its linear baseline and window 8 is not its screening window",
    "9.91": "quantisation floor computed with bin centres, where the real "
            "decoder takes bin left edges (SPEC V31)",
    "13 of 14": "window counts from the superseded single-clip run at window 8",
    "14 of 14": "window counts from the superseded single-clip run at window 8",
    "0.644": "mse_ratio from a 75-frame sampling of one clip; the same clip at "
             "135 frames is unwinnable, so the figure describes the sampling",
    "101.44": "planner bbox mse understated by 3/2, because bbox_mse scored an "
              "empty third object slot and counted it in the denominator; the "
              "current pair is 20.50 and 152.17",
    "13.67": "the oracle half of the same 3/2 understatement",
    "38.1": "E2's pre-correction planner error, paired with the withdrawn 12.0",
    "1.377": "E1's mse_ratio margin, computed over window-times-method rows; "
             "counted per window it is 1.05x",
    "160/160": "E1 row count, not window count; the structured arm has 80 windows",
    "8 of 116": "E1 row count, not window count; the unstructured arm reached "
                "4 of 58 windows",
}

# A section carrying any of these is understood to have declared itself.
MARKERS = ("superseded", "pre-review", "withdrawn", "corrected", "correction",
           "wrong", "do not quote", "struck")

DEFAULT_PATHS = ("notes/REPORT.md", "notes/docs/*.md")

# Append-only records. A dated entry there quoting a figure that was current on
# that date is history, not a stale number, and rewriting it would destroy the
# record. `check_docs.py` already excludes these; the two checkers disagreeing
# about what counts as history is how CHANGES.md came to be scanned at all.
HISTORICAL = ("CHANGES.md", "AUDIT.md")

# How far after a hit a marker still counts. A table row is often annotated on
# the line below it, which is legitimate and must not be reported.
LOOKAHEAD = 2


def _heading_before(lines, index):
    """(line number, text) of the nearest heading at or above `index`."""
    for i in range(index, -1, -1):
        if lines[i].startswith("#"):
            return i, lines[i]
    return 0, "(top of file)"


def find_uncovered(paths, registry=None):
    """Occurrences of a withdrawn number with no marker in their section.

    The marker must be in the **same section** — between the nearest heading
    above the hit and shortly after the hit itself. A marker under a different
    heading does not cover it, which is the restatement case that reading
    missed.
    """
    registry = SUPERSEDED if registry is None else registry
    patterns = {v: re.compile(r"(?<![\d.])" + re.escape(v) + r"(?![\d])")
                for v in registry}

    hits = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path) as handle:
            lines = handle.read().splitlines()
        for index, line in enumerate(lines):
            for value, pattern in patterns.items():
                if not pattern.search(line):
                    continue
                start, heading = _heading_before(lines, index)
                block = "\n".join(
                    lines[start:min(index + 1 + LOOKAHEAD, len(lines))]).lower()
                if any(m in block for m in MARKERS):
                    continue
                hits.append({"path": path, "line": index + 1, "value": value,
                             "heading": heading.strip("# ").strip(),
                             "reason": registry[value],
                             "text": line.strip()[:70]})
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paths", nargs="*", default=None,
                    help="files or globs to scan; defaults to the docs")
    a = ap.parse_args(argv)

    patterns = a.paths if a.paths else list(DEFAULT_PATHS)
    paths = []
    for p in patterns:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])

    paths = [p for p in paths
             if not any(p.endswith(h) for h in HISTORICAL)]
    hits = find_uncovered(paths)
    scanned = sum(1 for p in paths if os.path.isfile(p))

    if not hits:
        print("%d files scanned, no withdrawn number reads as current."
              % scanned)
        return 0

    print("%d files scanned, %d uncovered occurrence(s):\n" % (scanned, len(hits)))
    for h in hits:
        print("  %s:%d  under '%s'" % (h["path"], h["line"], h["heading"]))
        print("    %s" % h["text"])
        print("    %s is withdrawn: %s\n" % (h["value"], h["reason"]))
    print("Add a marker to the section (%s) or update the number."
          % ", ".join(MARKERS[:3]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
