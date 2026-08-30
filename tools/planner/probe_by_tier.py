#!/usr/bin/env python3
"""Break an M1 probe result down by predicate tier, and draw it.

The aggregate M1 number hides the thing worth knowing. On the oracle corpus
the overall lift is **-0.003**, which reads as "the code carries nothing" —
but per predicate it splits cleanly, and the split confirms a judgement made
independently months earlier.

`tools/video/screen_vidvrd.py` classifies every VidVRD predicate into three
tiers **by hand**, from what the words mean:

    attribute        a property of the pair, carrying no motion: larger, taller
    configurational  where the two sit: left, above, behind
    coupled          one object's motion against the other's: chase, ride, play

That table was a *judgement*. M1 is a *measurement*. They agree, and the
ordering is monotone::

    attribute        n= 2   mean lift  +0.263    2 of 2 readable
    configurational  n=18   mean lift  -0.020    1 of 18
    coupled          n= 4   mean lift  -0.139    0 of 4

**Only geometric predicates are recoverable from a positional code**, which is
exactly what a positional code should be able to do and nothing more. So M1
finds signal precisely where signal ought to be, which is a validation of the
probe rather than only a negative about the data.

The tension this exposes
------------------------

Criterion 0 says prefer **coupled** predicates, because rule-governed motion is
what FOSAE shone on in blocksworld. M1 says coupled predicates are the **least**
recoverable of the three. Both are right, and the gap between them is the
thesis's central difficulty rather than a contradiction.

A hypothesis, tested and falsified
----------------------------------

If coupled relations are about motion, a per-frame code cannot express them and
the transition should. Tested by giving the probe `[z_t | z_{t+1} - z_t]`
instead of `z_t`::

    state only            coupled -0.139   attribute +0.263
    state + one delta     coupled -0.132   attribute +0.146

**No rescue.** Coupled predicates stay unreadable, and the attribute
predicates get *worse*, because doubling the feature width adds noise without
adding signal. So the failure is not that the code is per-frame.

The reading that survives: `ride`, `play` and `follow` are **semantic**, not
geometric. A dog playing and a dog fighting can have identical box
trajectories, so no function of boxes alone separates them — over one frame or
over many. *(measured; only a one-step delta was tried, so a longer temporal
window remains untested.)*

    python3 tools/planner/probe_by_tier.py eval/probe/M1-oracle-corpus/probe.json
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_VIDEO = os.path.join(os.path.dirname(_HERE), "video")
if _VIDEO not in sys.path:
    sys.path.insert(0, _VIDEO)

TIER_ORDER = ("attribute", "configurational", "coupled")
TIER_COLOUR = {"attribute": "#12805c",
               "configurational": "#1f6feb",
               "coupled": "#b3261e"}


def lift(row):
    """Probe score minus the harder of the two controls."""
    if row.get("ridge") is None:
        return None
    base = max(row.get("prior") or 0.0, row.get("shuffled") or 0.0)
    return row["ridge"] - base


def by_tier(rows, tier_fn):
    out = {}
    for r in rows:
        v = lift(r)
        if v is None:
            continue
        out.setdefault(tier_fn(r["predicate"]), []).append((r["predicate"], v))
    for t in out:
        out[t].sort(key=lambda p: -p[1])
    return out


def _svg(groups, path, width=780, row_h=19):
    """A bar per predicate, grouped and coloured by tier, zero in the middle."""
    total = sum(len(v) for v in groups.values())
    if not total:
        return
    height = 60 + total * row_h + 26 * len(groups)
    left, mid = 210, 210 + (width - 210 - 60) / 2.0
    half = (width - 210 - 60) / 2.0
    span = max([abs(v) for g in groups.values() for _, v in g] + [0.1])

    out = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" '
           'font-family="sans-serif">' % (width, height)]
    out.append('<text x="8" y="18" font-size="13" fill="#444">M1 lift per '
               'predicate, grouped by tier. Right of the line = the linear '
               'probe beats its best control.</text>')
    out.append('<line x1="%.1f" y1="30" x2="%.1f" y2="%d" stroke="#999" '
               'stroke-width="1"/>' % (mid, mid, height - 10))

    y = 44
    for tier in TIER_ORDER:
        g = groups.get(tier)
        if not g:
            continue
        mean = sum(v for _, v in g) / len(g)
        out.append('<text x="8" y="%d" font-size="12" font-weight="600" '
                   'fill="%s">%s  (n=%d, mean %+0.3f)</text>'
                   % (y + 12, TIER_COLOUR[tier], tier, len(g), mean))
        y += 24
        for name, v in g:
            w = half * (abs(v) / span)
            x = mid if v >= 0 else mid - w
            label = name if len(name) <= 24 else name[:23] + "…"
            out.append('<text x="20" y="%d" font-size="11" fill="#333">%s</text>'
                       % (y + 12, label.replace("&", "&amp;").replace("<", "&lt;")))
            out.append('<rect x="%.1f" y="%d" width="%.1f" height="13" '
                       'fill="%s" opacity="%.2f"/>'
                       % (x, y, max(w, 0.6), TIER_COLOUR[tier],
                          0.85 if v >= 0 else 0.45))
            out.append('<text x="%.1f" y="%d" font-size="10" fill="#666">'
                       '%+0.3f</text>'
                       % (width - 54, y + 12, v))
            y += row_h
        y += 6
    out.append('</svg>')
    with open(path, "w") as f:
        f.write("".join(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("probe_json", help="probe.json written by predicate_probe")
    ap.add_argument("--out", default=None, help="where to write the SVG")
    a = ap.parse_args(argv)

    from screen_vidvrd import predicate_tier

    with open(a.probe_json) as f:
        doc = json.load(f)
    rows = doc.get("per_predicate", [])
    groups = by_tier(rows, predicate_tier)

    print("%-18s %4s %10s %10s  %s"
          % ("tier", "n", "mean lift", "max lift", "readable (lift > 0.10)"))
    print("-" * 74)
    for tier in TIER_ORDER:
        g = groups.get(tier)
        if not g:
            continue
        vals = [v for _, v in g]
        good = [n for n, v in g if v > 0.10]
        print("%-18s %4d %+10.3f %+10.3f  %s"
              % (tier, len(g), sum(vals) / len(vals), max(vals),
                 ", ".join(good) or "-"))

    out = a.out or os.path.join(os.path.dirname(a.probe_json), "by_tier.svg")
    _svg(groups, out)
    print("\nwrote %s" % out)

    print("\nThe tier table in tools/video/screen_vidvrd.py was written by "
          "judgement.\nThis is the measurement. Monotone agreement means the "
          "judgement holds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
