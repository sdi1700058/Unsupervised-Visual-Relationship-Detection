#!/usr/bin/env python3
"""Every term used in a special sense is defined once, in the glossary.

**Why this check exists.** Two terms entered this project with no definition.
"At window" appeared in a report before it meant anything to a reader, and
"corpus" was introduced as a synonym for dataset and never declared. Vocabulary
that arrives undefined produces documents only their author can read, and the
author of several of these documents does not remember writing them.

The check has two halves.

The first is a **banned list**: words this project used loosely and replaced
with something exact. Each one names the word to use instead, so a failure
tells the reader what to write rather than only what not to write. A banned
word passes only when the glossary itself defines it, which is how a decision
to readmit a word is recorded rather than argued.

The second is **coverage**: the glossary must actually parse and hold terms. A
check that cannot find its input reports failure rather than success, because a
gate that passes vacuously is worse than no gate. That is not hypothetical
here: `check_docs.py` once passed vacuously for weeks because one of its checks
was handed an empty scope.

    python3 tools/check_glossary.py
    python3 tools/check_glossary.py --verbose

Exits non-zero when a banned term is used in a live document, or when the
glossary is missing. Standard library only, Python 3.6 clean.
"""

import argparse
import glob
import os
import re
import sys

GLOSSARY = "notes/docs/GLOSSARY.md"

# Words replaced by something exact, and what to write instead. A word arrives
# here when it is found used without a definition, so the list grows by
# evidence rather than by taste.
BANNED = {
    "corpus": "dataset",
    "corpora": "datasets",
}

# Lines that name the rule rather than break it. A glossary row saying "avoid
# corpus, use dataset" must not fail the check it documents.
EXEMPT = re.compile(r"do not (write|use|say)|avoid|banned|use instead|"
                    r"deliberately not used|replaced with|never use", re.I)

ROW = re.compile(r"^\|\s*\**`?([A-Za-z_][A-Za-z_0-9 ]*?)`?\**\s*\|\s*(.+?)\s*\|\s*$")


def terms(path=GLOSSARY):
    """`{term: definition}` read from the glossary table."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            found = ROW.match(line.strip())
            if not found:
                continue
            term = found.group(1).strip()
            definition = found.group(2).strip()
            # The header row and the separator are not terms.
            if term.lower() in ("term", "avoid", "---"):
                continue
            out[term] = definition
    return out


def live_docs():
    """Documents describing the project as it is now."""
    paths = sorted(set(glob.glob("notes/*.md") + glob.glob("notes/docs/*.md")
                       + glob.glob("experiments/*/*.md") + ["README.md"]))
    return [p for p in paths if os.path.isfile(p)]


def undefined_terms(docs, defined):
    """Banned words used in a document, unless the glossary readmits them."""
    known = set(k.lower() for k in defined)
    hits = []
    for path in docs:
        # The glossary names banned words in order to ban them.
        if os.path.basename(path) == "GLOSSARY.md":
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            for i, line in enumerate(handle, 1):
                if EXEMPT.search(line):
                    continue
                low = line.lower()
                for term, instead in BANNED.items():
                    # A readmitted word is one the glossary defines.
                    if term in known:
                        continue
                    if re.search(r"\b%s\b" % re.escape(term), low):
                        hits.append({"doc": path, "line": i, "term": term,
                                     "use_instead": instead,
                                     "text": line.strip()[:80]})
                        break
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--glossary", default=GLOSSARY)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    defined = terms(a.glossary)
    if not defined:
        print("no glossary at %s, or it defines nothing." % a.glossary)
        print("  A gate that cannot find its input reports failure rather "
              "than success.")
        return 1

    docs = live_docs()
    hits = undefined_terms(docs, defined)
    if not hits:
        print("%d term(s) defined; %d document(s) use no banned word."
              % (len(defined), len(docs)))
        return 0

    print("\n%d use(s) of a word this project replaced:\n" % len(hits))
    print("  One word for one thing. Each line names what to write instead.\n")
    for hit in hits[:25]:
        print("  %s:%d   %s -> %s"
              % (hit["doc"], hit["line"], hit["term"], hit["use_instead"]))
        if a.verbose:
            print("      %s" % hit["text"])
    if len(hits) > 25:
        print("  ... and %d more" % (len(hits) - 25))
    return 1


if __name__ == "__main__":
    sys.exit(main())
