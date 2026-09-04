#!/usr/bin/env python3
"""Generate the cross-reference sections from the index, in place.

The supervisors asked a question the notes could not answer: *"Ποια βρήκες; Τι
datasets χρησιμοποιούσαν;"* — which papers did you find, and what datasets did
they use. Answering it needs papers, datasets and metrics to point at each
other, and hand-written links rot.

So each document keeps a **generated block** between markers:

    <!-- generated: datasets-by-paper -->
    ...replaced on every run...
    <!-- /generated -->

Everything outside the markers is hand-written and is never touched. A document
with no markers is skipped, so adding a block is opt-in per document.

    python3 tools/lit/render_index.py
    python3 tools/lit/render_index.py --check    # fail if a block is stale

Standard library only, Python 3.6 clean.
"""

import argparse
import collections
import json
import os
import re


INDEX = "notes/lit/index.json"

START = "<!-- generated: %s -->"
END = "<!-- /generated -->"

# Which block goes in which document.
TARGETS = {
    "notes/docs/RELATED_WORK.md": "shortlist-usage",
    "notes/docs/DATASETS.md": "papers-by-dataset",
    "notes/docs/EVAL_CONSIDERED.md": "papers-by-metric",
}


def load(path=None):
    with open(path or INDEX) as handle:
        return json.load(handle)


def _realness(paper):
    r = paper.get("realness") or []
    if "real" in r:
        return "real"
    if set(r) & {"synthetic", "simulated"}:
        return "synthetic"
    return "unstated"


def shortlist_usage(index):
    """The ranked shortlist against what each paper used."""
    ranked = [p for p in index["papers"] if p.get("shortlist_rank")]
    ranked.sort(key=lambda p: p["shortlist_rank"])
    lines = ["*Generated from `notes/lit/index.json`. Do not edit by hand; run "
             "`python3 tools/lit/render_index.py`.*", "",
             "Extraction is partial by nature: many of these papers name no "
             "dataset at all, because they evaluate on in-house environments. "
             "A dash means the summary names none, not that none was checked.",
             "",
             "| # | paper | data | datasets named | metrics named |",
             "|---|---|---|---|---|"]
    for p in ranked:
        lines.append("| %d | **%s** | %s | %s | %s |"
                     % (p["shortlist_rank"], p["id"], _realness(p),
                        ", ".join(p["datasets"]) or "—",
                        ", ".join(p["metrics"]) or "—"))
    return "\n".join(lines)


def papers_by_dataset(index):
    """Every dataset the library mentions, and who used it."""
    by = collections.defaultdict(list)
    for p in index["papers"]:
        for d in p["datasets"]:
            by[d].append(p)
    lines = ["*Generated from `notes/lit/index.json`.*", "",
             "Which papers in the library use each dataset, and which metrics "
             "they report alongside it. This is the evidence for choosing "
             "datasets the nearest work already uses.", "",
             "| dataset | papers | metrics reported on it |",
             "|---|---|---|"]
    for name in sorted(by, key=lambda k: (-len(by[k]), k)):
        papers = by[name]
        metrics = sorted(set(m for p in papers for m in p["metrics"]))
        ids = ", ".join(sorted(p["id"] for p in papers))
        lines.append("| **%s** (%d) | %s | %s |"
                     % (name, len(papers), ids, ", ".join(metrics) or "—"))
    return "\n".join(lines)


def papers_by_metric(index):
    """Every metric the library reports, and on what data."""
    by = collections.defaultdict(list)
    for p in index["papers"]:
        for m in p["metrics"]:
            by[m].append(p)
    lines = ["*Generated from `notes/lit/index.json`.*", "",
             "Which papers report each metric, and on which datasets. This is "
             "the evidence for whether a metric proposed here has a precedent.",
             "",
             "| metric | papers | datasets it was reported on |",
             "|---|---|---|"]
    for name in sorted(by, key=lambda k: (-len(by[k]), k)):
        papers = by[name]
        datasets = sorted(set(d for p in papers for d in p["datasets"]))
        ids = ", ".join(sorted(p["id"] for p in papers))
        lines.append("| **%s** (%d) | %s | %s |"
                     % (name, len(papers), ids, ", ".join(datasets) or "—"))
    return "\n".join(lines)


BLOCKS = {
    "shortlist-usage": shortlist_usage,
    "papers-by-dataset": papers_by_dataset,
    "papers-by-metric": papers_by_metric,
}


def replace_block(text, name, body):
    """Swap a generated block's contents. Returns (text, found)."""
    start, end = START % name, END
    if start not in text or end not in text.split(start, 1)[1]:
        return text, False
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return head + start + "\n\n" + body + "\n\n" + end + tail, True


def render_chart(index, path):
    """Most-used datasets, split by whether the data is of the real world."""
    by = collections.defaultdict(lambda: {"real": 0, "synthetic": 0,
                                          "unstated": 0})
    for p in index["papers"]:
        for d in p["datasets"]:
            by[d][_realness(p)] += 1
    top = sorted(by.items(), key=lambda kv: -sum(kv[1].values()))[:12]
    if not top:
        return None
    w, h, pad, row = 720, 60 + 26 * len(top), 168, 26
    scale = 420.0 / max(sum(v.values()) for _, v in top)
    colour = {"real": "#2b6cb0", "synthetic": "#c05621", "unstated": "#a0aec0"}
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
             'viewBox="0 0 %d %d">' % (w, h, w, h),
             '<style>text{font-family:sans-serif}.t{font-size:14px;'
             'font-weight:bold}.l{font-size:11px}.k{font-size:11px}</style>',
             '<rect width="%d" height="%d" fill="white"/>' % (w, h),
             '<text x="12" y="22" class="t">Datasets used by the library, '
             'by whether the data is of the real world</text>']
    for i, (name, counts) in enumerate(top):
        y = 44 + i * row
        x = pad
        for kind in ("real", "synthetic", "unstated"):
            n = counts[kind]
            if not n:
                continue
            width = max(int(n * scale), 2)
            parts.append('<rect x="%d" y="%d" width="%d" height="16" '
                         'fill="%s"/>' % (x, y, width, colour[kind]))
            x += width
        parts.append('<text x="%d" y="%d" class="l" text-anchor="end">%s</text>'
                     % (pad - 8, y + 13, name))
        parts.append('<text x="%d" y="%d" class="k">%d</text>'
                     % (x + 6, y + 13, sum(counts.values())))
    legend = 44 + len(top) * row
    for j, kind in enumerate(("real", "synthetic", "unstated")):
        parts.append('<rect x="%d" y="%d" width="11" height="11" fill="%s"/>'
                     % (pad + j * 110, legend, colour[kind]))
        parts.append('<text x="%d" y="%d" class="k">%s</text>'
                     % (pad + j * 110 + 16, legend + 10, kind))
    parts.append("</svg>")
    svg = "\n".join(parts)
    with open(path, "w") as handle:
        handle.write(svg)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="do not write; fail if any block is out of date")
    a = ap.parse_args(argv)

    index = load()
    stale, written, skipped = [], [], []
    for path, name in sorted(TARGETS.items()):
        if not os.path.isfile(path):
            skipped.append("%s (absent)" % path)
            continue
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        body = BLOCKS[name](index)
        new, found = replace_block(text, name, body)
        if not found:
            skipped.append("%s (no '%s' markers)" % (path, name))
            continue
        if new == text:
            continue
        if a.check:
            stale.append(path)
        else:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(new)
            written.append(path)

    for s in skipped:
        print("  skipped %s" % s)
    if a.check:
        if stale:
            print("%d generated block(s) out of date: %s"
                  % (len(stale), ", ".join(stale)))
            return 1
        print("generated blocks are up to date")
        return 0
    for path in written:
        print("  updated %s" % path)

    chart = render_chart(index, "notes/lit/dataset_usage.svg")
    if chart:
        print("  wrote %s" % chart)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
