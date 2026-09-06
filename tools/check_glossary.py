#!/usr/bin/env python3
"""Every term used in a special sense is defined once, in the glossary.

**Why this check exists.** Two terms entered this project with no definition.
"At window" appeared in a report before it meant anything to a reader, and
"corpus" was introduced as a synonym for dataset and never declared. Vocabulary
that arrives undefined produces documents only their author can read, and the
author of several of these documents does not remember writing them.

The check has three halves.

The first is a **banned list**: words this project used loosely and replaced
with something exact. Each one names the word to use instead, so a failure
tells the reader what to write rather than only what not to write. A banned
word passes only when the glossary itself defines it, which is how a decision
to readmit a word is recorded rather than argued.

The second is **names**: the same words, in the names of the files git tracks.
Prose was the only thing read for months while
`experiments/M_evaluation_methods/build_oracle_corpus.sh` sat in the tree, and
the gate reported a clean run every time. Two things hid it. Word boundaries:
`\bcorpus\b` finds nothing in `build_oracle_corpus`, because an underscore is
a word character and there is therefore no boundary in front of the word. And
scope: the file list was never looked at in the first place. A name is read
more often than a paragraph, so a word banned in the prose and permitted in
the file names is banned in only half the places it is read.

The third is **coverage**: the glossary must actually parse and hold terms,
and there must be tracked files to read names from. A check that cannot find
its input reports failure rather than success, because a gate that passes
vacuously is worse than no gate. That is not hypothetical here: `check_docs.py`
once passed vacuously for weeks because one of its checks was handed an empty
scope.

    python3 tools/check_glossary.py
    python3 tools/check_glossary.py --verbose

Exits non-zero when a banned term is used in a live document, or when the
glossary is missing. Standard library only, Python 3.6 clean.
"""

import argparse
import glob
import os
import re
import subprocess
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


# Where the definitions stop. Everything below this heading lists words the
# project refuses to use, and those rows are not definitions.
#
# Reading them as definitions made this check pass vacuously on its first real
# run: `corpus` appeared in the refusal table, was registered as a defined
# term, and every use of it in every document was then skipped. A planted
# violation went undetected, which is precisely the failure the module's own
# docstring warns about.
STOP_HEADING = re.compile(r"^#+\s*terms deliberately not used", re.I)

# The refusal table's own header row, which is the second and structural way
# to recognise it.
#
# The heading alone was a single point of failure: renaming it -- an ordinary
# editorial act on a document that is outside version control, so it leaves no
# diff -- put the refusal rows back into the definition set, readmitted
# `corpus`, and made every use of it in every document skip again. That is the
# vacuous pass this module's docstring says already happened once. A table's
# column names are part of its structure rather than its prose, so they are
# the sturdier marker of the two and both are honoured.
STOP_TABLE_HEADER = re.compile(r"^\|\s*avoid\s*\|", re.I)


def terms(path=GLOSSARY):
    """`{term: definition}` read from the glossary table.

    Only the definition table. The refusal table below it names words in order
    to ban them, so treating its rows as definitions would readmit exactly the
    words the check exists to catch.
    """
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if (STOP_HEADING.match(line.strip())
                    or STOP_TABLE_HEADER.match(line.strip())):
                break
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


def tracked_names():
    """Every path git tracks, or an empty list when git cannot be asked.

    Tracked files only. Data on disk is not this check's business: a directory
    of exports named years ago is not a document anyone edits, and renaming it
    would break the loaders that read it.
    """
    try:
        out = subprocess.check_output(["git", "ls-files", "-z"],
                                      stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError):
        return []
    return [p for p in out.decode("utf-8", "replace").split("\0") if p]


def name_pattern(term):
    """A banned word in a name, bounded by anything that is not a letter.

    `\\b` is wrong here. Names join words with underscores and dashes, and an
    underscore is a word character, so `\\bcorpus\\b` never matched
    `build_oracle_corpus` -- which is exactly how that file survived this
    gate. A plain substring test is wrong in the other direction:
    `incorporate` contains `corpora`.
    """
    return re.compile(r"(?<![a-z])%s(?![a-z])" % re.escape(term))


def banned_names(paths, defined):
    """Banned words in the names of tracked files and their directories.

    Every path component is read, so a directory named once is caught as
    surely as a file. One offending component is reported once however many
    files sit under it.
    """
    known = set(k.lower() for k in defined)
    patterns = [(term, instead, name_pattern(term))
                for term, instead in sorted(BANNED.items())
                if term not in known]
    hits, seen = [], set()
    for path in paths:
        for part in path.lower().split("/"):
            for term, instead, pattern in patterns:
                if not pattern.search(part):
                    continue
                if (part, term) in seen:
                    continue
                seen.add((part, term))
                hits.append({"path": path, "name": part, "term": term,
                             "use_instead": instead})
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

    paths = tracked_names()
    if not paths:
        print("git tracks no file here, so no name was checked.")
        print("  A gate that cannot find its input reports failure rather "
              "than success.")
        return 1

    docs = live_docs()
    hits = undefined_terms(docs, defined)
    named = banned_names(paths, defined)
    if not hits and not named:
        print("%d term(s) defined; %d document(s) and %d tracked name(s) use "
              "no banned word." % (len(defined), len(docs), len(paths)))
        return 0

    print("\n%d use(s) of a word this project replaced:\n"
          % (len(hits) + len(named)))
    print("  One word for one thing. Each line names what to write instead.\n")
    for hit in hits[:25]:
        print("  %s:%d   %s -> %s"
              % (hit["doc"], hit["line"], hit["term"], hit["use_instead"]))
        if a.verbose:
            print("      %s" % hit["text"])
    if len(hits) > 25:
        print("  ... and %d more" % (len(hits) - 25))
    for hit in named[:25]:
        print("  %s   %s -> %s   (in the name, so rename the file)"
              % (hit["path"], hit["term"], hit["use_instead"]))
    if len(named) > 25:
        print("  ... and %d more name(s)" % (len(named) - 25))
    return 1


if __name__ == "__main__":
    sys.exit(main())
