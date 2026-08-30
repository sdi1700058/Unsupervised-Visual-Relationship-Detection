#!/usr/bin/env python3
"""Is Something-Else's vocabulary more positional than VidVRD's?

**Why this question decides the dataset.** M1 measured, on VidVRD, that a
positional code recovers only geometric predicates:

    attribute        larger, taller          mean lift  +0.263    2 of 2
    configurational  left, above, behind     mean lift  -0.020    1 of 18
    coupled          chase, ride, play       mean lift  -0.139    0 of 4

A bounding box carries position and size. `ride` and `play` are not functions
of position and size, so no positional representation can express them — and
that was measured on *ground-truth* boxes, so it is a ceiling rather than a
model failure.

That result transfers into a **test a dataset must pass**: if its label
vocabulary is not a function of where the boxes are, the whole pipeline is
measuring something it cannot reach. This applies the test to Something-Else's
174 Something-Something v2 categories.

The distinction that matters, and the surprise
----------------------------------------------

Something-Else looked risky on this test. **21% of its categories are
"pretending to X" or "trying and failing to X"**, which sound like statements
about intent, and intent is exactly what a box cannot hold.

They are mostly not. *"Pretending to put something into something"* differs
from *"Putting something into something"* by whether the object **ends up
inside** the other — and containment is a relation between two boxes.
*"Pretending to poke"* differs from *"poking"* by contact, which is box
overlap. The intent is inferred by the annotator, but the **evidence** is
geometric.

Some genuinely are out of reach. *"Pretending to turn something upside down"*
turns on orientation, and an axis-aligned bounding box does not encode
rotation at all.

So the classification below asks one question per category: **is the outcome
that distinguishes it from its neighbours a function of the boxes?**

===============  ==========================================================
`geometric`      containment, contact, support, occlusion, direction of
                 motion, size change. A box pair decides it.
`partial`        a box pair narrows it but does not settle it — usually
                 because the category names a manner or a material.
`semantic`       orientation, identity, material properties, or pure
                 intent. Boxes are silent.
===============  ==========================================================

**This is a JUDGEMENT, not a measurement**, and it must not be quoted as one.
It cannot be measured without the frames, which are not obtainable yet. Its
standing rests on the fact that the equivalent judgement for VidVRD —
`screen_vidvrd.PREDICATE_TIERS`, written by hand from what the words mean —
was afterwards confirmed by M1 with a monotone ordering. The rules are all in
this file so the judgement can be argued with rather than taken on trust.

    python3 tools/video/screen_something_else.py \\
        --labels data/video/something_else/splits/labels.json --svg

Standard library only.
"""

import argparse
import json
import os
import sys

# Each rule is (tier, marker). First match wins, so order matters and the
# most specific markers come first. Every marker is a substring of the
# lower-cased category name.
RULES = (
    # --- semantic: boxes are silent -------------------------------------
    ("semantic", "upside down"),          # orientation; a bbox has no rotation
    ("semantic", "unbendable"),           # a material property
    ("semantic", "not tearable"),
    ("semantic", "that can't"),
    ("semantic", "is not"),
    ("semantic", "so it does not fall"),  # a counterfactual about stability
    ("semantic", "twist"),                # rotation again
    ("semantic", "rolls"),                # needs orientation to see rolling
    ("semantic", "spinning"),
    ("semantic", "stacked"),              # ambiguous under an axis-aligned box

    # --- partial: the boxes narrow it, the manner decides it -------------
    ("partial", "squeez"),                # deformation, only partly in a box
    ("partial", "tear"),
    ("partial", "fold"),
    ("partial", "bend"),
    ("partial", "wipe"),
    ("partial", "spread"),
    ("partial", "sprinkl"),
    ("partial", "pour"),                  # the substance is often untracked
    ("partial", "scoop"),
    ("partial", "stuff"),
    ("partial", "attach"),
    ("partial", "burying"),
    ("partial", "digging"),

    # --- geometric: a box pair decides it --------------------------------
    ("geometric", " into "),              # containment
    ("geometric", " out of "),
    ("geometric", " onto "),
    ("geometric", " off of "),
    ("geometric", " behind "),
    ("geometric", "in front of"),
    ("geometric", " next to "),
    ("geometric", "underneath"),
    ("geometric", " under "),
    ("geometric", "on top of"),
    ("geometric", "on the surface"),
    ("geometric", "on a surface"),
    ("geometric", " up"),                 # direction of motion
    ("geometric", " down"),
    ("geometric", " away"),
    ("geometric", " closer"),
    ("geometric", " across "),
    ("geometric", "falls"),
    ("geometric", "fall"),
    ("geometric", "drop"),
    ("geometric", "lift"),
    ("geometric", "push"),
    ("geometric", "pull"),
    ("geometric", "mov"),
    ("geometric", "throw"),
    ("geometric", "cover"),               # occlusion is visible in boxes
    ("geometric", "hitting"),             # contact
    ("geometric", "touch"),
    ("geometric", "poke"),
    ("geometric", "collid"),
    ("geometric", "approach"),
    ("geometric", "tilt"),                # the box aspect ratio changes
    ("geometric", "open"),
    ("geometric", "close"),
    ("geometric", "put"),
    ("geometric", "take"),
    ("geometric", "pick"),
    ("geometric", "show"),
)

TIER_ORDER = ("geometric", "partial", "semantic")
TIER_COLOUR = {"geometric": "#12805c", "partial": "#c68a12",
               "semantic": "#b3261e"}


def category_tier(name):
    """Classify one category. Returns a tier and the marker that decided it."""
    low = " " + name.lower().strip() + " "
    for tier, marker in RULES:
        if marker in low:
            return tier, marker
    return "semantic", "(no marker matched)"


def is_rule_violation(name):
    """Does this category name a rule being broken, failed or faked?

    These are what make Something-Else score well on Criterion 0: a dataset
    that labels the FAILURE of a rule is one whose world has rules.
    """
    low = name.lower()
    return any(m in low for m in
               ("pretend", "trying", "failing", "but ", "without ",
                "unsuccessful", "so nothing happens"))


def classify(categories):
    rows = []
    for c in categories:
        tier, marker = category_tier(c)
        rows.append({"category": c, "tier": tier, "marker": marker,
                     "rule_violation": is_rule_violation(c)})
    return rows


def _svg(rows, path, width=560, height=250):
    """Two stacked bars: all categories, and the rule-violation subset."""
    def counts(sub):
        return [sum(1 for r in sub if r["tier"] == t) for t in TIER_ORDER]

    groups = [("all 174 categories", rows),
              ("rule-violation subset", [r for r in rows if r["rule_violation"]])]
    out = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" '
           'font-family="sans-serif">' % (width, height)]
    out.append('<text x="8" y="18" font-size="13" fill="#444">Something-Else '
               'categories: is the outcome a function of the boxes?</text>')
    out.append('<text x="8" y="34" font-size="11" fill="#888">green geometric '
               '&#183; amber partial &#183; red semantic. A JUDGEMENT, not a '
               'measurement.</text>')
    left, span = 20, float(width - 40)
    y = 56
    for title, sub in groups:
        n = max(1, len(sub))
        c = counts(sub)
        out.append('<text x="%d" y="%d" font-size="12" fill="#333">%s '
                   '(n=%d)</text>' % (left, y + 12, title, len(sub)))
        y += 22
        x = left
        for tier, k in zip(TIER_ORDER, c):
            w = span * k / float(n)
            out.append('<rect x="%.1f" y="%d" width="%.1f" height="26" '
                       'fill="%s"/>' % (x, y, max(w, 0.5), TIER_COLOUR[tier]))
            if w > 34:
                out.append('<text x="%.1f" y="%d" font-size="11" fill="#fff" '
                           'text-anchor="middle">%d</text>'
                           % (x + w / 2.0, y + 18, k))
            x += w
        y += 44
    out.append('</svg>')
    with open(path, "w") as f:
        f.write("".join(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels",
                    default="data/video/something_else/splits/labels.json")
    ap.add_argument("--svg", action="store_true", help="also write a figure")
    ap.add_argument("--list", action="store_true", help="print every category")
    a = ap.parse_args(argv)

    if not os.path.isfile(a.labels):
        raise SystemExit(
            "%s missing. Fetch it once:\n"
            "  curl -o %s https://raw.githubusercontent.com/joaanna/"
            "something_else/master/code/dataset_splits/compositional/labels.json"
            % (a.labels, a.labels))

    with open(a.labels) as f:
        categories = list(json.load(f))
    rows = classify(categories)
    viol = [r for r in rows if r["rule_violation"]]

    def table(title, sub):
        print("\n%s (n=%d)" % (title, len(sub)))
        for t in TIER_ORDER:
            k = sum(1 for r in sub if r["tier"] == t)
            print("   %-11s %3d  %4.0f%%" % (t, k, 100.0 * k / max(1, len(sub))))

    table("all categories", rows)
    table("rule-violation subset -- what Criterion 0 wants", viol)

    if a.list:
        print()
        for t in TIER_ORDER:
            print("--- %s" % t)
            for r in rows:
                if r["tier"] == t:
                    print("   %-72s %s" % (r["category"][:72], r["marker"]))

    if a.svg:
        out = os.path.join("eval", "datasets")
        if not os.path.isdir(out):
            os.makedirs(out)
        p = os.path.join(out, "something_else_tiers.svg")
        _svg(rows, p)
        print("\nwrote %s" % p)

    print("\nJUDGEMENT, not a measurement. The rules are in RULES in this file "
          "so they\ncan be argued with. The equivalent judgement for VidVRD "
          "was afterwards\nconfirmed by M1 with a monotone ordering, which is "
          "why this one is worth\nmaking at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
