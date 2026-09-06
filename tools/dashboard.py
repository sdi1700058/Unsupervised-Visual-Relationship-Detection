#!/usr/bin/env python3
"""One image that answers "where is this thesis", in ten seconds.

**Why an image and not a document.** The author has autonomy-blindness by
design: with work running unattended they are easily left with no idea what is
happening unless they can see the bottom line in bulk, in a humanly
understandable way. Four documents already exist and none of them answers the
question at a glance.

Four panels, top to bottom.

1. **Milestone bars.** Magnitude, so one hue light to dark, never a rainbow.
   M2 prints the trained count first, because that is the supervisor's
   requirement and the number the author called the important one.
2. **The grid matrix.** Datasets down, methods across, each cell split on the
   diagonal with the oracle half lower-left and the trained half upper-right.
   The four states use the reference **status** palette, which is reserved and
   never reused for a series. On a light surface two of those steps fall below
   3:1, so the palette's own mitigation applies here: every state carries a
   **glyph and a legend**, and colour never carries meaning alone.
3. **The findings strip.** The claims, each with its confidence mark.
4. **The blocked strip.** What waits on the author.

**Dark mode is chosen, not flipped.** The dark steps are declared against the
dark surface rather than derived by inverting the light ones.

    python3 tools/dashboard.py
    python3 tools/dashboard.py --out eval/dashboard.svg

Standard library only, Python 3.6 clean.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tools import grid as grid_mod                          # noqa: E402
from tools import workplan                                  # noqa: E402


# The reference status palette, unchanged. These four are reserved: a status
# colour never impersonates a series, and a series colour never means a state.
STATE_FILL = {
    grid_mod.SCORED: "#0ca30c",
    grid_mod.COLLAPSED: "#d03b3b",
    grid_mod.CANNOT_RUN: "#fab219",
    grid_mod.NEVER_RUN: None,   # drawn with the .empty class
}

# The mitigation the palette documents for its own sub-3:1 steps on a light
# surface. Colour never carries the meaning by itself.
STATE_GLYPH = {
    grid_mod.SCORED: "check",
    grid_mod.COLLAPSED: "cross",
    grid_mod.CANNOT_RUN: "bar",
    grid_mod.NEVER_RUN: "empty",
}

STATE_MARK = {
    grid_mod.SCORED: "✓",
    grid_mod.COLLAPSED: "✕",
    grid_mod.CANNOT_RUN: "—",
    grid_mod.NEVER_RUN: "",
}

TIER_MARK = {"universal": "universal", "comparative": "comparative",
             "existential": "existential", "scoped": "scoped"}


def _clip(text, limit):
    """Cut on a word boundary, and say that a cut happened.

    Cutting mid-word produced "from a wider usa" on the first render, which
    reads as a typo rather than as a truncation.
    """
    text = str(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return (cut or text[:limit]) + "..."


def _esc(text):
    """Escape for an SVG text node. A raw angle bracket is invalid XML."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _style():
    """Tokens for both surfaces, as classes rather than custom properties.

    **Literal fills, not `var()`.** A standalone SVG is opened by whatever the
    reader has, and many renderers do not resolve CSS custom properties. This
    file rendered entirely invisible on the first attempt for exactly that
    reason: every fill was `var(--ink)`, every one resolved to nothing, and the
    tests all passed because the document was still well-formed XML.

    A renderer that also ignores the media query still gets a correct light
    rendering, because the light values are the plain declarations and the dark
    ones are the override. The dark steps are chosen against the dark surface
    rather than derived by inverting the light ones.
    """
    return (
        "<style>\n"
        "  text { font-family: sans-serif; fill: #0b0b0b; }\n"
        "  .h { font-size: 17px; font-weight: bold; }\n"
        "  .s { font-size: 12px; font-weight: bold; }\n"
        "  .l { font-size: 11px; fill: #52514e; }\n"
        "  .m { font-size: 10px; fill: #52514e; }\n"
        "  .g { font-size: 11px; fill: #ffffff; font-weight: bold; }\n"
        "  .bg { fill: #fcfcfb; }\n"
        "  .track { fill: #e4e4e0; }\n"
        "  .bar { fill: #2a78d6; }\n"
        "  .empty { fill: #f0efec; }\n"
        "  .rule { stroke: #d6d5d1; }\n"
        "  @media (prefers-color-scheme: dark) {\n"
        "    text { fill: #ffffff; }\n"
        "    .l, .m { fill: #c3c2b7; }\n"
        "    .g { fill: #ffffff; }\n"
        "    .bg { fill: #1a1a19; }\n"
        "    .track { fill: #33322f; }\n"
        "    .bar { fill: #3987e5; }\n"
        "    .empty { fill: #2a2a28; }\n"
        "    .rule { stroke: #3d3c39; }\n"
        "  }\n"
        "</style>"
    )


def _panel_milestones(plan, x, y, width):
    """Five bars. Magnitude, so one hue and a recessive track."""
    rows = workplan.milestone_progress(plan)
    parts = ['<text x="%d" y="%d" class="h">Milestones</text>' % (x, y)]
    yy = y + 22
    label_x = x + width - 60
    span = label_x - (x + 300) - 12
    for row in rows:
        target = row.get("target") or 0
        have = row.get("have") or 0
        frac = (float(have) / target) if target else 0.0
        frac = max(0.0, min(1.0, frac))
        parts.append('<text x="%d" y="%d" class="s">%s</text>'
                     % (x, yy + 11, _esc(row["id"])))
        parts.append('<text x="%d" y="%d" class="l">%s</text>'
                     % (x + 30, yy + 11, _esc(_clip(row["title"], 44))))
        # A 4px rounded data-end anchored to the baseline, on a recessive
        # track. The track shows the whole, so a short bar still reads.
        parts.append('<rect x="%d" y="%d" width="%d" height="10" rx="4" '
                     'class="track"/>' % (x + 300, yy + 2, span))
        if frac > 0:
            parts.append('<rect x="%d" y="%d" width="%.1f" height="10" '
                         'rx="4" class="bar"/>'
                         % (x + 300, yy + 2, max(4.0, span * frac)))
        parts.append('<text x="%d" y="%d" class="l">%d of %d</text>'
                     % (label_x, yy + 11, have, target))
        if row.get("beside"):
            parts.append('<text x="%d" y="%d" class="m">%d %s</text>'
                         % (x + 30, yy + 24, len(row["beside"]),
                            _esc(_clip(row.get("beside_label", ""), 70))))
            yy += 14
        yy += 26
    return parts, yy


def _panel_grid(plan, x, y, width):
    """The matrix. Status colours, and a glyph in every cell."""
    cells = grid_mod.enumerate_cells(plan)
    datasets = [d.get("name") for d in plan.get("datasets_for_grid") or []]
    methods = plan.get("methods_for_grid") or []
    parts = ['<text x="%d" y="%d" class="h">The grid: dataset by method'
             '</text>' % (x, y),
             '<text x="%d" y="%d" class="m">Each cell splits on the diagonal. '
             'Lower left is the oracle, upper right is the trained run.</text>'
             % (x, y + 16)]
    if not datasets or not methods:
        parts.append('<text x="%d" y="%d" class="l">No grid declared yet.'
                     '</text>' % (x, y + 40))
        return parts, y + 56

    by_key = {}
    for cell in cells:
        by_key[(cell["dataset"], cell["method"], cell["source"])] = cell

    left = x + 130
    size = 30
    gap = 2
    top = y + 92
    for i, method in enumerate(methods):
        cx = left + i * (size + gap) + size / 2.0
        parts.append('<text x="%.1f" y="%d" class="m" '
                     'transform="rotate(-40 %.1f %d)">%s</text>'
                     % (cx, top - 6, cx, top - 6, _esc(method)[:18]))
    for j, dataset in enumerate(datasets):
        cy = top + j * (size + gap)
        parts.append('<text x="%d" y="%d" class="l">%s</text>'
                     % (x, cy + 20, _esc(dataset)[:16]))
        for i, method in enumerate(methods):
            cx = left + i * (size + gap)
            for source, corner in (("oracle", "lower"), ("trained", "upper")):
                cell = by_key.get((dataset, method, source))
                state = cell["state"] if cell else grid_mod.NEVER_RUN
                fill = STATE_FILL[state]
                if corner == "lower":
                    points = "%d,%d %d,%d %d,%d" % (cx, cy, cx, cy + size,
                                                    cx + size, cy + size)
                else:
                    points = "%d,%d %d,%d %d,%d" % (cx, cy, cx + size, cy,
                                                    cx + size, cy + size)
                if fill:
                    parts.append('<polygon points="%s" fill="%s"/>'
                                 % (points, fill))
                else:
                    parts.append('<polygon points="%s" class="empty"/>'
                                 % points)
                mark = STATE_MARK[state]
                if mark:
                    mx = cx + (9 if corner == "lower" else 21)
                    my = cy + (23 if corner == "lower" else 13)
                    parts.append('<text x="%d" y="%d" class="g">%s</text>'
                                 % (mx, my, _esc(mark)))
    bottom = top + len(datasets) * (size + gap) + 20

    # The legend is always present. Identity is never colour alone.
    lx = x
    for state in (grid_mod.SCORED, grid_mod.COLLAPSED, grid_mod.CANNOT_RUN,
                  grid_mod.NEVER_RUN):
        swatch = STATE_FILL[state]
        if swatch:
            parts.append('<rect x="%d" y="%d" width="11" height="11" rx="2" '
                         'fill="%s"/>' % (lx, bottom, swatch))
        else:
            parts.append('<rect x="%d" y="%d" width="11" height="11" rx="2" '
                         'class="empty"/>' % (lx, bottom))
        mark = STATE_MARK[state]
        if mark:
            parts.append('<text x="%d" y="%d" class="g" '
                         'style="font-size:9px">%s</text>'
                         % (lx + 2, bottom + 9, _esc(mark)))
        parts.append('<text x="%d" y="%d" class="m">%s</text>'
                     % (lx + 16, bottom + 9, _esc(state)))
        lx += 22 + 8 * len(state)
    return parts, bottom + 26


def _panel_findings(plan, x, y, width):
    """The claims, each with the confidence its evidence licenses."""
    parts = ['<text x="%d" y="%d" class="h">What the evidence currently says'
             '</text>' % (x, y)]
    yy = y + 22
    claims = plan.get("claims") or []
    if not claims:
        parts.append('<text x="%d" y="%d" class="l">No claims registered.'
                     '</text>' % (x, yy))
        return parts, yy + 20
    for claim in claims:
        try:
            e = workplan.claim_strength(claim, plan)
            tier = workplan.wording_tier(claim, e) or "inconclusive"
        except Exception:                   # noqa: BLE001 - a board must render
            e, tier = 0.0, "unknown"
        parts.append('<text x="%d" y="%d" class="s">%s</text>'
                     % (x, yy, _esc(claim.get("id", "?"))))
        parts.append('<text x="%d" y="%d" class="l">%s</text>'
                     % (x + 30, yy, _esc(claim.get("asserts", ""))[:78]))
        parts.append('<text x="%d" y="%d" class="m">%s, strength %.2f</text>'
                     % (x + width - 190, yy, _esc(TIER_MARK.get(tier, tier)),
                        e))
        yy += 19
    return parts, yy + 8


def _panel_blocked(plan, x, y, width):
    """What waits on the author, and nothing else."""
    parts = ['<text x="%d" y="%d" class="h">Waiting on you</text>' % (x, y)]
    yy = y + 22
    open_questions = [q for q in (plan.get("questions") or [])
                      if not (q.get("answered") or "").strip()]
    if not open_questions:
        parts.append('<text x="%d" y="%d" class="l">Nothing. Every question '
                     'has an answer recorded.</text>' % (x, yy))
        return parts, yy + 20
    for question in open_questions:
        parts.append('<text x="%d" y="%d" class="s">%s</text>'
                     % (x, yy, _esc(question.get("id", "?"))))
        parts.append('<text x="%d" y="%d" class="l">%s</text>'
                     % (x + 34, yy, _esc(question.get("asks", ""))[:88]))
        yy += 19
    return parts, yy + 8


def render(plan):
    """The whole dashboard, as SVG text."""
    width = 980
    x = 24
    parts = []
    body = []

    block, y = _panel_milestones(plan, x, 40, width - 48)
    body += block
    body.append('<line x1="%d" y1="%d" x2="%d" y2="%d" class="rule"/>'
                % (x, y + 4, width - x, y + 4))

    block, y = _panel_grid(plan, x, y + 30, width - 48)
    body += block
    body.append('<line x1="%d" y1="%d" x2="%d" y2="%d" class="rule"/>'
                % (x, y + 4, width - x, y + 4))

    block, y = _panel_findings(plan, x, y + 30, width - 48)
    body += block
    body.append('<line x1="%d" y1="%d" x2="%d" y2="%d" class="rule"/>'
                % (x, y + 4, width - x, y + 4))

    block, y = _panel_blocked(plan, x, y + 30, width - 48)
    body += block

    height = int(y + 30)
    parts.append('<svg xmlns="http://www.w3.org/2000/svg" width="%d" '
                 'height="%d" viewBox="0 0 %d %d">' % (width, height, width,
                                                       height))
    parts.append(_style())
    parts.append('<rect width="%d" height="%d" class="bg"/>'
                 % (width, height))
    parts += body
    parts.append("</svg>")
    return "\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", default="notes/WORKPLAN.json")
    ap.add_argument("--out", default="eval/dashboard.svg")
    a = ap.parse_args(argv)

    if not os.path.isfile(a.plan):
        print("no plan at %s" % a.plan)
        return 2
    with open(a.plan) as handle:
        plan = json.load(handle)

    directory = os.path.dirname(a.out)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(a.out, "w") as handle:
        handle.write(render(plan))
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
