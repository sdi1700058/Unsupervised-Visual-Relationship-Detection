#!/usr/bin/env python3
"""Quality gate for the notes, the same way the code has one.

**Why the notes need a gate at all.** They are outside version control, so they
have no history, no diff, no review point and no commit message explaining a
change. Everything that keeps the code honest is absent for them, and the work
is planned from them.

An audit on 2026-08-31 found what that costs:

- **STATUS.md, whose own header calls it the current state of the project, was
  carrying a superseded headline.** `check_headlines.py` was declared to read it
  and no entry ever referenced it, so it was never read.
- **EVAL.md, the authoritative evaluation document, was carrying the same
  superseded headline**, and the check passed because two other documents had
  been updated. One fresh document was excusing two stale ones.
- Two documents declared a task **finished**. Progress is reported here; a work
  item is closed by decision, not by a document asserting it.
- Three references pointed at scripts that had moved.

    python3 tools/check_docs.py
    python3 tools/check_docs.py --verbose

Exits non-zero when any check fails. Standard library only, Python 3.6 clean.
"""

import argparse
import glob
import os
import re
import subprocess
import sys


# Documents that describe the project as it is now. These must be correct.
def live_docs():
    paths = sorted(set(glob.glob("notes/*.md") + glob.glob("notes/docs/*.md")
                       + glob.glob("experiments/*/*.md") + ["README.md"]))
    return [p for p in paths
            if os.path.isfile(p) and os.path.basename(p) not in HISTORICAL]


# Append-only records. They describe what was true when written, so a name that
# has since moved is correct history rather than a defect.
HISTORICAL = {"CHANGES.md", "AUDIT.md", "loop.md"}

# Paths a document may name although they do not exist, each with the reason.
# A bare allowlist rots; a reason can be checked by a reader.
ALLOWED_ABSENT = {
    "latplan/domains/video/vidvrd.py":
        "planned destination, SPEC.md I1 and F1, explicitly still pending",
    "latplan/domains/image/labeled_objects.py":
        "planned destination, SPEC.md I10 and F1, explicitly still pending",
    "latplan/domains/image/__init__.py":
        "planned destination, SPEC.md F1, explicitly still pending",
    "latplan/domains/video/videonet.py":
        "planned loader, SPEC.md VN2, not yet written",
    "notes/docs/GUIDE.md":
        "planned document, SPEC.md task E3, not yet written",
    "tools/workplan.py":
        "planned tool, DESIGN_WORKPLAN.md section 5, built in phase P0",
    "tools/video/screen_actiongenome.py":
        "planned screen, named in the worked example in DESIGN_WORKPLAN.md",
}

# A template naming a filename shape rather than a file. `reports/YYYY-MM-DD.md`
# is an instruction about how to name a file, not a claim that one exists.
PLACEHOLDER = re.compile(r"YYYY|MM-DD|<[a-z_]+>|\{|\*")

# Generated output. Absent because it is gitignored or was cleaned, never
# because a document is wrong.
ALLOWED_PREFIXES = ("eval/", "data/", "out/", "logs/")

PATH_PAT = re.compile(
    r"\b((?:tools|sh|experiments|latplan|eval|data|out|logs|notes)"
    r"/[A-Za-z0-9_./-]+\.(?:py|sh|md|json|csv|txt|npz|npy|yml|html|svg))")

# Standing rule: never write that a TASK is complete. Reporting that a chunk of
# implementation is finished is fine, so the pattern targets the subjects that
# name work items.
DONE_PAT = re.compile(
    r"\b(pipeline|task|evaluation|project|phase|thesis|work|dataset search)\b"
    r"[^.\n]{0,40}?\b(is|are|was|were)\s+(now\s+)?"
    r"(complete|completed|finished|settled|closed)\b", re.I)

# Lines that quote the rule rather than break it.
DONE_EXEMPT = re.compile(r"never write|forbidden|do not (write|say)|"
                         r"previously read|corrected|the correct heading|"
                         r"was not the assistant|SUPERSEDED|superseded|"
                         r"you do not decide|the user does|only the user|"
                         # Prose describing this check is not prose breaking it.
                         r"claims that a task|checks (for|three)",
                         re.I)


def dead_paths(docs=None):
    """References to repository paths that do not exist."""
    docs = live_docs() if docs is None else docs
    hits = []
    for path in docs:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for i, line in enumerate(handle, 1):
                for ref in PATH_PAT.findall(line):
                    if os.path.exists(ref):
                        continue
                    if PLACEHOLDER.search(ref):
                        continue
                    if ref in ALLOWED_ABSENT:
                        continue
                    if ref.startswith(ALLOWED_PREFIXES):
                        continue
                    hits.append({"doc": path, "line": i, "ref": ref,
                                 "text": line.strip()[:80]})
    return hits


def completion_claims(docs=None):
    """Statements that a work item is closed, which a document cannot decide."""
    docs = live_docs() if docs is None else docs
    hits = []
    for path in docs:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for i, line in enumerate(handle, 1):
                if DONE_PAT.search(line) and not DONE_EXEMPT.search(line):
                    hits.append({"doc": path, "line": i,
                                 "text": line.strip()[:90]})
    return hits


def _actual_test_count():
    """How many tests the suite really has, or None if it cannot be run."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "unittest", "discover",
             "-s", "tools/planner/tests"],
            stderr=subprocess.STDOUT).decode("utf-8", "replace")
    except subprocess.CalledProcessError as exc:
        out = exc.output.decode("utf-8", "replace")
    except OSError:
        return None
    found = re.search(r"^Ran (\d+) tests?", out, re.M)
    return int(found.group(1)) if found else None


# Only a claim about the WHOLE suite is checkable. "20 tests" written beside
# one feature is a component count and says nothing about the total.
SUITE_MARKER = re.compile(r"unittest discover|test suite|tests pass|"
                          r"tests?, OK|suite is at", re.I)

# A dated row records what was true on that date. Correct history, not a stale
# number, and rewriting it would destroy the record.
DATED = re.compile(r"\b20\d\d-\d\d-\d\d\b")


# "not supplied" and "measured as nothing" are different. Conflating them made
# this function run the whole suite as a subprocess from inside that same
# suite, which never terminates.
UNSET = object()


def stale_counts(docs=None, actual=UNSET):
    """Claims about the size of the whole suite that disagree with it."""
    docs = live_docs() if docs is None else docs
    actual = _actual_test_count() if actual is UNSET else actual
    if actual is None:
        return []
    pat = re.compile(r"\b(\d{2,4})\s+tests\b")
    hits = []
    for path in docs:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for i, line in enumerate(handle, 1):
                # "123 -> 162 tests" is a progression; a dated row is history.
                if "->" in line or "→" in line or DATED.search(line):
                    continue
                if not SUITE_MARKER.search(line):
                    continue
                for claim in pat.findall(line):
                    if int(claim) != actual:
                        hits.append({"doc": path, "line": i, "claim": int(claim),
                                     "actual": actual,
                                     "text": line.strip()[:80]})
    return hits


# Terms that describe the working arrangement rather than what the code does.
# Each carries why it matters, because a bare blocklist rots.
# "autonomous" is deliberately absent: it is a research term of art here, as in
# "autonomous driving", and flagging it would produce noise for ever.
CONTEXT_TERMS = {
    "the next loop": "describes a working arrangement, not what the code does",
    "each loop": "same",
    "every loop": "same",
    "a cold loop": "same",
    "unattended": "describes how the work is supervised, not what runs",
}

# Where the terms are tolerated: superseded scripts kept only as a record of
# what produced which output directory.
VOCAB_EXEMPT_DIRS = ("sh/deprecated/",)

# A blocklist has to be able to name what it blocks, and its own tests have to
# be able to exercise it. Both are self-reference, not violations.
VOCAB_EXEMPT_FILES = ("tools/check_docs.py",
                      "tools/planner/tests/test_check_docs.py")


def _tracked_source():
    """Files under version control, which is what a reader can see."""
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "*.py", "*.sh", "*.md"],
            stderr=subprocess.STDOUT).decode("utf-8", "replace")
    except (subprocess.CalledProcessError, OSError):
        return []
    return [p for p in out.split("\n")
            if p and not p.startswith(VOCAB_EXEMPT_DIRS)
            and p not in VOCAB_EXEMPT_FILES]


def working_context_vocabulary(paths=None):
    """Tracked source that describes the working arrangement.

    **This check supplies its own scope.** Only files under version control are
    in scope, because only those can be read by anyone else, and the notes are
    deliberately excluded from version control. An earlier version intersected
    the caller's list with the tracked set, which made it pass vacuously when
    handed the notes — a check that always passes is worse than one that is
    occasionally noisy.
    """
    paths = _tracked_source() if paths is None else paths
    hits = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            for i, line in enumerate(handle, 1):
                low = line.lower()
                for term, why in CONTEXT_TERMS.items():
                    if term in low:
                        hits.append({"doc": path, "line": i, "ref": term,
                                     "text": line.strip()[:80], "why": why})
    return hits


CHECKS = (
    ("tracked files describing the working arrangement",
     working_context_vocabulary,
     "The repository is public. Source describes behaviour; how the work is "
     "supervised is not the code's business."),
    ("references to files that do not exist", dead_paths,
     "A document that names a moved script sends the next reader to a path "
     "that is not there."),
    ("claims that a task is complete", completion_claims,
     "A document cannot close a work item. Reporting what a chunk of "
     "implementation produced is fine; declaring the task done is not."),
    ("test counts that disagree with the suite", stale_counts,
     "A stale count is a document describing a repository that no longer "
     "exists, and it is the cheapest possible thing to keep true."),
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    docs = live_docs()
    failed = 0
    for title, fn, why in CHECKS:
        # One check supplies its own scope: it reads the tracked tree, not the
        # notes. Handing it `docs` is what made it pass vacuously.
        hits = fn() if fn is working_context_vocabulary else fn(docs)
        if not hits:
            if a.verbose:
                print("  ok    %s" % title)
            continue
        failed += 1
        print("\n%d %s:" % (len(hits), title))
        print("  %s\n" % why)
        for h in hits[:20]:
            detail = h.get("ref") or ("claims %s, suite has %s"
                                      % (h.get("claim"), h.get("actual")))
            print("  %s:%d" % (h["doc"], h["line"]))
            print("      %s" % detail)
            print("      %s" % h["text"])
        if len(hits) > 20:
            print("  ... and %d more" % (len(hits) - 20))

    if not failed:
        print("%d documents checked, all %d checks pass."
              % (len(docs), len(CHECKS)))
        return 0
    print("\n%d of %d document checks failed." % (failed, len(CHECKS)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
