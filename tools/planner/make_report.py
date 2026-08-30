#!/usr/bin/env python3
"""Turn a planner run into one page you can read in a minute.

A `summary.csv` is not a result anyone can absorb. With the loop running
unattended the bottom line has to be visible without reading a table, so this
writes a single self-contained HTML page: the verdict first, then the numbers
that support it, then a chart, then the per-window detail for anyone who wants
it.

    python3 tools/planner/make_report.py eval/planner/H15-pddl-150010
    python3 tools/planner/make_report.py eval/planner/*/ --index

Standard library only — no matplotlib, no numpy, no network. The chart is
inline SVG written by hand, so the page opens anywhere and the script runs on
Sherlock's Python 3.6 as readily as here.
"""

import argparse
import csv
import glob
import html
import math
import os
import sys


def _num(row, key):
    """A CSV cell as a float, or None when it is blank or unparseable."""
    value = (row.get(key) or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _median(values):
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _missing_columns(rows, needed=("moving_gt_steps", "mse_ratio", "bbox_mse")):
    """Which expected columns the CSV does not have.

    A missing column is not the same as a column full of zeros, and the report
    used to conflate them: a winning planner whose CSV lacked `moving_gt_steps`
    was described as "none of them carry motion". Say which it is.
    """
    if not rows:
        return list(needed)
    return [c for c in needed if c not in rows[0]]


def summarise(rows, min_motion=6):
    """Reduce a run to the numbers worth stating, and a verdict.

    Only windows the planner solved **and** whose annotated boxes actually
    move are scored: a window where nothing moves has a near-zero baseline, so
    its ratio is noise rather than signal (`EVAL.md` §4.8).
    """
    missing = _missing_columns(rows)
    solved = [r for r in rows if (r.get("reachability") or "").strip() == "True"]
    scored = [r for r in solved
              if (_num(r, "moving_gt_steps") or 0) >= min_motion
              and _num(r, "bbox_mse") is not None]

    ratio = _median([_num(r, "mse_ratio") for r in scored
                     if _num(r, "mse_ratio") is not None])
    beats = sum(1 for r in scored
                if (r.get("beats_baseline") or "").strip() == "True")

    out = {
        "windows": len(rows),
        "solved": len(solved),
        "scored": len(scored),
        "ratio": ratio,
        "beats": beats,
        "planner_mse": _median([_num(r, "bbox_mse") for r in scored
                                if _num(r, "bbox_mse") is not None]),
        "baseline_mse": _median([_num(r, "baseline_mse") for r in scored
                                 if _num(r, "baseline_mse") is not None]),
        "planner_iou": _median([_num(r, "bbox_iou") for r in scored
                                if _num(r, "bbox_iou") is not None]),
        "baseline_iou": _median([_num(r, "baseline_iou") for r in scored
                                 if _num(r, "baseline_iou") is not None]),
        "fallbacks": sum(int(_num(r, "decode_fallbacks") or 0) for r in scored),
        "rows": rows,
        "scored_rows": scored,
    }

    if not solved:
        out["verdict"] = ("No window was solved. The action schema does not "
                          "connect the two frames at all.")
    elif missing:
        out["verdict"] = (
            "Cannot judge this run: the summary is missing the column(s) %s. "
            "That is a different thing from a run with no signal, and the two "
            "used to be reported identically." % ", ".join(missing))
    elif not scored:
        out["verdict"] = ("Windows solved, but none of them carry motion, so "
                          "nothing measurable came out of this run.")
    elif ratio is None:
        out["verdict"] = "Windows scored, but no ratio could be formed."
    elif ratio < 1.0:
        out["verdict"] = (
            "The planner **beats** linear interpolation: mse_ratio %.3f, "
            "which is %.1f times better, on %d of %d scored windows."
            % (ratio, 1.0 / ratio, beats, len(scored)))
    else:
        out["verdict"] = (
            "The planner **loses** to linear interpolation: mse_ratio %.2f. "
            "It wins %d of %d scored windows."
            % (ratio, beats, len(scored)))
    return out


def _bars(scored, width=760, height=210):
    """Per-window planner error against the baseline, on a log scale.

    Errors span four orders of magnitude across a run, so a linear axis shows
    one bar and thirteen slivers.
    """
    if not scored:
        return "<p><em>No scored windows to plot.</em></p>"

    pairs = []
    for r in scored:
        p, b = _num(r, "bbox_mse"), _num(r, "baseline_mse")
        if p is None or b is None:
            continue
        pairs.append((r.get("init", "?"), max(p, 1e-3), max(b, 1e-3)))
    if not pairs:
        return "<p><em>No scored windows to plot.</em></p>"

    top = max(max(p, b) for _, p, b in pairs)
    bottom = min(min(p, b) for _, p, b in pairs)
    lo, hi = math.log10(max(bottom, 1e-3)), math.log10(top)
    span = max(hi - lo, 0.5)
    plot_h = height - 40
    slot = width / float(len(pairs))
    bar = max(4, min(18, slot / 3.0))

    def y_of(v):
        frac = (math.log10(v) - lo) / span
        return 10 + plot_h * (1.0 - frac)

    svg = ['<svg viewBox="0 0 %d %d" class="chart" role="img" '
           'aria-label="planner error against baseline per window">' % (width, height)]
    svg.append('<line x1="0" y1="%d" x2="%d" y2="%d" class="axis"/>'
               % (10 + plot_h, width, 10 + plot_h))
    for i, (init, p, b) in enumerate(pairs):
        x = i * slot + slot / 2.0
        svg.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                   'class="base"><title>window %s baseline %.1f</title></rect>'
                   % (x - bar, y_of(b), bar, 10 + plot_h - y_of(b),
                      html.escape(str(init)), b))
        svg.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                   'class="plan"><title>window %s planner %.1f</title></rect>'
                   % (x, y_of(p), bar, 10 + plot_h - y_of(p),
                      html.escape(str(init)), p))
        svg.append('<text x="%.1f" y="%d" class="tick">%s</text>'
                   % (x, height - 12, html.escape(str(init))))
    svg.append('</svg>')
    return "".join(svg)


CSS = """
:root{--bg:#fff;--fg:#16181d;--mut:#5b6270;--line:#e3e6ec;--plan:#1f6feb;
--base:#c9ced8;--good:#12805c;--bad:#b3261e;--card:#f7f8fa}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e9ecf1;
--mut:#9aa3b2;--line:#2a2f38;--base:#3b414d;--card:#1b1e24}}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1.25rem;background:var(--bg);color:var(--fg);
font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:60rem;margin:0 auto}
h1{font-size:1.5rem;margin:0 0 .25rem}
.sub{color:var(--mut);margin:0 0 1.5rem;font-size:.9rem}
.verdict{padding:1rem 1.25rem;border-radius:10px;background:var(--card);
border-left:4px solid var(--mut);margin:0 0 1.5rem;font-size:1.05rem}
.verdict.win{border-left-color:var(--good)}
.verdict.lose{border-left-color:var(--bad)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
gap:.75rem;margin:0 0 1.75rem}
.card{background:var(--card);border-radius:10px;padding:.85rem 1rem}
.card .k{color:var(--mut);font-size:.78rem;text-transform:uppercase;
letter-spacing:.04em}
.card .v{font-size:1.45rem;font-weight:650;margin-top:.15rem}
.chart{width:100%;height:auto;margin:.5rem 0 .25rem}
.axis{stroke:var(--line);stroke-width:1}
.plan{fill:var(--plan)}.base{fill:var(--base)}
.tick{fill:var(--mut);font-size:9px;text-anchor:middle}
.key{color:var(--mut);font-size:.85rem;margin:0 0 2rem}
.key b{color:var(--plan)}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:.88rem;min-width:44rem}
th,td{padding:.5rem .7rem;text-align:right;border-bottom:1px solid var(--line);
white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600;font-size:.76rem;text-transform:uppercase;
letter-spacing:.04em}
tr:last-child td{border-bottom:0}
td.win{color:var(--good);font-weight:600}
td.lose{color:var(--bad)}
footer{color:var(--mut);font-size:.82rem;margin-top:2rem}
a{color:var(--plan)}
"""


def _cards(s):
    def card(label, value):
        return ('<div class="card"><div class="k">%s</div>'
                '<div class="v">%s</div></div>'
                % (html.escape(label), html.escape(str(value))))

    fmt = lambda v, f="%.2f": "n/a" if v is None else f % v
    return "".join([
        card("windows solved", "%d / %d" % (s["solved"], s["windows"])),
        card("scored", s["scored"]),
        card("mse ratio", fmt(s["ratio"], "%.3f")),
        card("planner error", fmt(s["planner_mse"])),
        card("baseline", fmt(s["baseline_mse"])),
        card("IoU", "%s / %s" % (fmt(s["planner_iou"], "%.3f"),
                                 fmt(s["baseline_iou"], "%.3f"))),
    ])


def _table(rows):
    cols = [("init", "from"), ("goal", "to"), ("reachability", "solved"),
            ("plan_length", "plan"), ("moving_gt_steps", "moves"),
            ("bbox_mse", "planner"), ("baseline_mse", "baseline"),
            ("mse_ratio", "ratio"), ("bbox_iou", "IoU"),
            ("decode_fallbacks", "fallbacks")]
    out = ['<div class="wrap"><table><thead><tr>']
    out += ['<th>%s</th>' % html.escape(lbl) for _, lbl in cols]
    out.append('</tr></thead><tbody>')
    for r in rows:
        out.append('<tr>')
        for key, _ in cols:
            raw = (r.get(key) or "").strip()
            cls = ""
            if key == "mse_ratio" and raw:
                try:
                    cls = ' class="win"' if float(raw) < 1 else ' class="lose"'
                except ValueError:
                    cls = ""
            if key in ("bbox_mse", "baseline_mse", "mse_ratio", "bbox_iou"):
                try:
                    raw = "%.3f" % float(raw) if raw else "—"
                except ValueError:
                    pass
            out.append('<td%s>%s</td>' % (cls, html.escape(raw or "—")))
        out.append('</tr>')
    out.append('</tbody></table></div>')
    return "".join(out)


def build(run_dir):
    """Write `report.html` inside a planner run directory. Returns its path."""
    csv_path = os.path.join(run_dir, "summary.csv")
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None

    s = summarise(rows)
    name = os.path.basename(os.path.normpath(run_dir))
    css_class = "win" if (s["ratio"] is not None and s["ratio"] < 1) else \
                ("lose" if s["ratio"] is not None else "")

    verdict = html.escape(s["verdict"]).replace("**", "")
    chart = _bars(s["scored_rows"])
    body = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>%s — planner report</title><style>%s</style></head><body><main>
<h1>%s</h1>
<p class="sub">%d windows &middot; below 1 the planner beat a straight line
drawn between the two endpoint frames</p>
<p class="verdict %s">%s</p>
<div class="grid">%s</div>
%s
<p class="key"><b>Blue</b> is the planner, grey is the linear baseline.
Log scale, since errors span orders of magnitude. Shorter is better.</p>
%s
<footer>Generated by <code>tools/planner/make_report.py</code> from
<code>%s</code>. Only windows that solved and whose boxes actually move are
scored.</footer>
</main></body></html>""" % (
        html.escape(name), CSS, html.escape(name), s["windows"], css_class,
        verdict, _cards(s), chart, _table(rows),
        html.escape(csv_path))

    out = os.path.join(run_dir, "report.html")
    with open(out, "w") as fh:
        fh.write(body)

    # The chart is also written on its own, because an image file beside the
    # data is what actually gets looked at later. SVG opens in any browser or
    # image viewer and needs no matplotlib, so this works in a bare sandbox and
    # on Sherlock's Python 3.6 alike. `chart` was built once, above.
    if chart.startswith("<svg"):
        standalone = chart.replace(
            '<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
        style = ('<style>.axis{stroke:#c9ced8}.plan{fill:#1f6feb}'
                 '.base{fill:#c9ced8}.tick{fill:#5b6270;font:9px sans-serif;'
                 'text-anchor:middle}'
                 'text.title{fill:#16181d;font:600 12px sans-serif;'
                 'text-anchor:start}</style>')
        title = ('<text x="4" y="9" class="title">%s &#8212; planner (blue) vs '
                 'baseline, log scale</text>' % html.escape(name))
        standalone = standalone.replace('>', '>' + style + title, 1)
        with open(os.path.join(run_dir, "chart.svg"), "w") as fh:
            fh.write(standalone)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Write a readable HTML report for a planner run.")
    ap.add_argument("dirs", nargs="+", help="planner run directories")
    ap.add_argument("--index", action="store_true",
                    help="also write eval/planner/index.html linking them all")
    args = ap.parse_args(argv)

    targets = []
    for d in args.dirs:
        if os.path.exists(os.path.join(d, "summary.csv")):
            targets.append(d)
        else:
            targets.extend(os.path.dirname(p) for p in
                           sorted(glob.glob(os.path.join(d, "*", "summary.csv"))))
    if not targets:
        print("no planner run with a summary.csv found", file=sys.stderr)
        return 1

    made = []
    for d in targets:
        path = build(d)
        if path:
            with open(os.path.join(d, "summary.csv")) as fh:
                s = summarise(list(csv.DictReader(fh)))
            made.append((d, path, s))
            print("%-44s %s" % (os.path.basename(os.path.normpath(d)),
                                s["verdict"].replace("**", "")))

    if args.index and made:
        made.sort(key=lambda m: (m[2]["ratio"] is None, m[2]["ratio"] or 0))
        items = []
        for d, path, s in made:
            ratio = "n/a" if s["ratio"] is None else "%.3f" % s["ratio"]
            cls = "win" if (s["ratio"] is not None and s["ratio"] < 1) else "lose"
            items.append(
                '<tr><td><a href="%s">%s</a></td><td class="%s">%s</td>'
                '<td>%d/%d</td><td>%d</td></tr>'
                % (html.escape(os.path.relpath(path, "eval/planner")),
                   html.escape(os.path.basename(os.path.normpath(d))),
                   cls, ratio, s["solved"], s["windows"], s["beats"]))
        index = ("""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Planner runs</title><style>%s</style></head><body><main>
<h1>Planner runs</h1><p class="sub">Best ratio first. Below 1 beats the
straight-line baseline.</p><div class="wrap"><table><thead><tr><th>run</th>
<th>mse ratio</th><th>solved</th><th>wins</th></tr></thead><tbody>%s
</tbody></table></div></main></body></html>""" % (CSS, "".join(items)))
        os.makedirs("eval/planner", exist_ok=True)
        with open("eval/planner/index.html", "w") as fh:
            fh.write(index)
        print("\nwrote eval/planner/index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
