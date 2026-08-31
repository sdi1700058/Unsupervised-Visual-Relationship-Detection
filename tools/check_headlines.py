#!/usr/bin/env python3
"""Check the thesis's headline numbers against the run data they come from.

**The gap this closes.** `check_superseded.py` catches a number that was
**withdrawn**. It cannot catch one that was merely **improved on**, because
nothing marks those. Claim 1 sat in `THESIS_MAP.md` at n=10 — "6 of 10" — for a
day while n=22 — "12 of 22" — was already on disk. It was found by hand.

A number improved somewhere and not carried to where it is quoted is the same
failure as a stale document, and it is harder to see because nothing about the
old value looks wrong.

So each headline is recomputed here from `eval/planner/*/summary.csv` and the
documents are checked for the current value.

    python3 tools/check_headlines.py
    python3 tools/check_headlines.py --verbose

Exits non-zero when a document quotes a value the data no longer supports.

**A headline with no run data is skipped, not failed.** `eval/` is gitignored
and a fresh checkout has none of it, so a missing directory must not fail the
gate — it only means the check cannot speak.

Standard library only, Python 3.6 clean.
"""

import argparse
import csv
import glob
import os
import statistics
import sys


DOCS = {
    "map": ".claude/docs/THESIS_MAP.md",
    "report": ".claude/REPORT.md",
    "eval": ".claude/docs/EVAL.md",
    "status": ".claude/docs/STATUS.md",
}


def fmt_ratio(value):
    """A ratio as the documents write it."""
    return "%.2f" % value


def fmt_count(hit, total):
    """A count pair as the documents write it."""
    return "%d of %d" % (hit, total)


def quoted_in(paths, needle):
    """Is `needle` present in any of these documents?"""
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path) as handle:
            if needle in handle.read():
                return True
    return False


def _medians(pattern, column="mse_ratio", min_motion=6):
    """One median per run directory matching `pattern`, skipping empty runs."""
    out = []
    for directory in sorted(glob.glob(pattern)):
        path = os.path.join(directory, "summary.csv")
        if not os.path.isfile(path):
            continue
        with open(path) as handle:
            rows = [r for r in csv.DictReader(handle)
                    if (r.get("reachability") or "").strip() == "True"
                    and (r.get(column) or "").strip() not in ("", "None")
                    and float(r.get("moving_gt_steps") or 0) >= min_motion]
        if rows:
            out.append(statistics.median(float(r[column]) for r in rows))
    return out


def _oracle_beats_baseline():
    values = _medians("eval/planner/p3-*/")
    if len(values) < 5:
        return None
    return fmt_count(sum(1 for v in values if v < 1.0), len(values))


def _oracle_median_ratio():
    values = _medians("eval/planner/p3-*/")
    if len(values) < 5:
        return None
    return fmt_ratio(statistics.median(values))


def _trained_beats_baseline():
    values = _medians("eval/planner/w16-trained-*/")
    if len(values) < 5:
        return None
    return fmt_count(sum(1 for v in values if v < 1.0), len(values))


# name -> how to recompute it, which documents must carry it, and why it
# matters. `why` is printed on a mismatch so the reader learns what broke.
HEADLINES = {
    "oracle beats the baseline": {
        "compute": _oracle_beats_baseline,
        "docs": [DOCS["map"], DOCS["report"], DOCS["eval"]],
        "why": "Claim 1's headline count. It sat at n=10 for a day while n=22 "
               "was on disk.",
    },
    "oracle median mse_ratio": {
        "compute": _oracle_median_ratio,
        "docs": [DOCS["map"], DOCS["report"], DOCS["eval"]],
        "why": "Claim 1's headline ratio.",
    },
    "trained beats the baseline": {
        "compute": _trained_beats_baseline,
        "docs": [DOCS["map"], DOCS["report"], DOCS["eval"]],
        "why": "Claim 2's headline count.",
    },
}


def check(headlines=None):
    """Recompute each headline and see whether the documents carry it."""
    headlines = HEADLINES if headlines is None else headlines
    stale, ok, skipped = [], [], []
    for name, spec in sorted(headlines.items()):
        value = spec["compute"]()
        if value is None:
            skipped.append(name)
            continue
        if quoted_in(spec["docs"], value):
            ok.append((name, value))
        else:
            stale.append((name, value, spec.get("why", "")))
    return {"stale": stale, "ok": ok, "skipped": skipped}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    result = check()

    if a.verbose:
        for name, value in result["ok"]:
            print("  ok      %-28s %s" % (name, value))
        for name in result["skipped"]:
            print("  skip    %-28s no run data" % name)

    if not result["stale"]:
        print("%d headline(s) match the run data, %d skipped for want of data."
              % (len(result["ok"]), len(result["skipped"])))
        return 0

    print("%d headline(s) the documents do not carry:\n" % len(result["stale"]))
    for name, value, why in result["stale"]:
        print("  %s" % name)
        print("    the data now says: %s" % value)
        print("    %s\n" % why)
    print("Update the documents, or say why the older figure is the one to "
          "quote.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
