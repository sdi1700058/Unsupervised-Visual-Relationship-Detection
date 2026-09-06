#!/usr/bin/env python3
"""Score every dataset candidate before any of them is chosen.

**Why this exists, and it is not a tidy-up.** The author struck availability
from the dataset criteria and set its weight to zero. The bias continued
anyway, through four corrections, because it had moved upstream rather than
away: only three datasets were ever scored, and those three were the three
already sitting on disk. Every other candidate in `DATASETS_CONSIDERED.md`
reads "unscored", and an unscored dataset cannot win a comparison.

So compliance with the letter left the selection rigged. The guard is
therefore mechanical rather than another instruction:

1. Every candidate is scored before any is chosen. The inputs are readable
   from the papers, so this needs no download and no cluster.
2. A dataset selected out of score order fails the check, unless the plan
   carries a written reason. A departure is then argued rather than drifted
   into.

**A missing criterion scores `None`, never zero.** Treating a gap as a zero is
how an unmeasured dataset loses for being new, which is the same bias wearing a
different hat. An unscored candidate ranks last and is labelled unscored, which
is a statement about our knowledge and not about the dataset.

    python3 tools/score_datasets.py
    python3 tools/score_datasets.py --check-order

Standard library only, Python 3.6 clean.
"""

import argparse
import json
import os
import sys

CANDIDATES = "notes/lit/dataset_candidates.json"
OUT_DIR = "eval/datasets"

# The author's weights, set on 2026-09-04 and unchanged since. Structure is
# heaviest because it decides whether there is anything to learn, while density
# only decides whether we can measure it.
#
# Availability is absent rather than zero. A criterion weighted zero is still a
# column somebody can argue about; a criterion that does not exist is not.
WEIGHTS = {
    "structure": 0.30,
    "relations": 0.25,
    "literature": 0.20,
    "density": 0.15,
    "volume": 0.10,
}

CRITERION_MEANING = {
    "structure": "does the subject obey rules that permeate its world",
    "relations": "is the full relational ground truth available",
    "literature": "do the papers nearest this task use it",
    "density": "how densely is it annotated, per frame and per clip",
    "volume": "how many samples does it hold",
}


def score(candidate, weights=None):
    """The weighted score in 0 to 1, or None when a criterion is missing."""
    weights = WEIGHTS if weights is None else weights
    total = 0.0
    for name, weight in weights.items():
        value = candidate.get(name)
        if value is None:
            return None
        total += weight * float(value)
    return total


def rank(candidates, weights=None):
    """Candidates by score, best first, with the unscored last and marked."""
    rows = []
    for candidate in candidates:
        row = dict(candidate)
        row["score"] = score(candidate, weights)
        rows.append(row)
    scored = [r for r in rows if r["score"] is not None]
    unscored = [r for r in rows if r["score"] is None]
    scored.sort(key=lambda r: (-r["score"], r["name"]))
    unscored.sort(key=lambda r: r["name"])
    return scored + unscored


def out_of_order(plan, ranked):
    """Datasets selected below a better-ranked one, with no written reason."""
    position = {}
    for i, row in enumerate(ranked):
        position[row["name"]] = i
    selected = [d for d in plan.get("corpora", []) if d.get("selected")]
    best_unselected = None
    for i, row in enumerate(ranked):
        if row["score"] is None:
            continue
        if not any(d["name"] == row["name"] for d in selected):
            best_unselected = i
            break
    problems = []
    for dataset in selected:
        name = dataset.get("name")
        where = position.get(name)
        if where is None:
            problems.append({"dataset": name, "rank": None,
                             "reason": "not among the scored candidates"})
            continue
        if ranked[where]["score"] is None:
            problems.append({"dataset": name, "rank": where,
                             "reason": "selected while unscored"})
            continue
        if best_unselected is not None and where > best_unselected:
            if not (dataset.get("reason") or "").strip():
                problems.append({
                    "dataset": name, "rank": where,
                    "reason": "ranks below %s and carries no written reason"
                              % ranked[best_unselected]["name"]})
    return problems


def _esc(text):
    """Escape for SVG text. A raw angle bracket is invalid XML."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def render_svg(ranked):
    """One stacked bar per dataset, split into the weighted criteria."""
    order = ["structure", "relations", "literature", "density", "volume"]
    colours = {"structure": "#2b6cb0", "relations": "#2f855a",
               "literature": "#b7791f", "density": "#805ad5",
               "volume": "#c05621"}
    rows = [r for r in ranked]
    width = 760
    height = 92 + 26 * max(1, len(rows))
    pad = 168
    span = width - pad - 96
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
             'viewBox="0 0 %d %d">' % (width, height, width, height),
             '<style>text{font-family:sans-serif}.t{font-size:15px;'
             'font-weight:bold}.l{font-size:11px}.n{font-size:10px;fill:#555}'
             '</style>',
             '<rect width="%d" height="%d" fill="white"/>' % (width, height),
             '<text x="14" y="24" class="t">Dataset candidates, on the '
             'author\'s weights</text>']
    legend = "  ".join("%s %.2f" % (k, WEIGHTS[k]) for k in order)
    parts.append('<text x="14" y="42" class="n">%s. Availability is not a '
                 'criterion.</text>' % _esc(legend))
    if not rows:
        parts.append('<text x="14" y="70" class="n">No candidates scored '
                     'yet.</text>')
        parts.append("</svg>")
        return "\n".join(parts)

    y = 62
    for row in rows:
        parts.append('<text x="14" y="%d" class="l">%s</text>'
                     % (y + 11, _esc(row["name"])[:24]))
        if row["score"] is None:
            parts.append('<text x="%d" y="%d" class="n">unscored, and '
                         'therefore not comparable</text>' % (pad, y + 11))
            y += 26
            continue
        x = pad
        for name in order:
            value = row.get(name)
            if value is None:
                continue
            piece = span * WEIGHTS[name] * float(value)
            parts.append('<rect x="%.1f" y="%d" width="%.1f" height="14" '
                         'fill="%s"/>' % (x, y, max(0.0, piece),
                                          colours[name]))
            x += piece
        parts.append('<text x="%.1f" y="%d" class="l">%.3f</text>'
                     % (x + 6, y + 11, row["score"]))
        y += 26
    parts.append("</svg>")
    return "\n".join(parts)


def load_candidates(path=CANDIDATES):
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        data = json.load(handle)
    return data.get("candidates", data) if isinstance(data, dict) else data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidates", default=CANDIDATES)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--plan", default="notes/WORKPLAN.json")
    ap.add_argument("--check-order", action="store_true",
                    help="only report selections made out of score order")
    a = ap.parse_args(argv)

    candidates = load_candidates(a.candidates)
    if candidates is None:
        print("no candidates at %s." % a.candidates)
        print("  Every dataset must be scored before any is chosen. Until "
              "this file exists, the selection is decided by whatever "
              "happens to be on disk, which is the failure this guards.")
        return 2

    ranked = rank(candidates)
    scored = [r for r in ranked if r["score"] is not None]

    plan = {}
    if os.path.isfile(a.plan):
        with open(a.plan) as handle:
            plan = json.load(handle)
    problems = out_of_order(plan, ranked)

    if a.check_order:
        if not problems:
            print("%d scored candidate(s); every selection is in score order "
                  "or carries a reason." % len(scored))
            return 0
    else:
        print("%d candidate(s), %d scored\n" % (len(ranked), len(scored)))
        for i, row in enumerate(ranked, 1):
            if row["score"] is None:
                print("  %2d. %-22s unscored" % (i, row["name"]))
                continue
            print("  %2d. %-22s %.3f   %s"
                  % (i, row["name"], row["score"],
                     " ".join("%s=%.2f" % (k[:4], row[k])
                              for k in WEIGHTS if row.get(k) is not None)))
        if not os.path.isdir(a.out_dir):
            os.makedirs(a.out_dir)
        with open(os.path.join(a.out_dir, "scores.json"), "w") as handle:
            json.dump({"weights": WEIGHTS, "ranked": ranked}, handle, indent=2)
        figure = os.path.join(a.out_dir, "scores.svg")
        with open(figure, "w") as handle:
            handle.write(render_svg(ranked))
        print("\nwrote %s/scores.json and %s"
              % (a.out_dir, figure))

    if problems:
        print("\n%d selection(s) out of score order:\n" % len(problems))
        print("  A dataset may rank below another and still be chosen, but "
              "the reason has to be written down. Add a `reason` to the "
              "dataset in WORKPLAN.json.\n")
        for problem in problems:
            print("  %-22s %s" % (problem["dataset"], problem["reason"]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
