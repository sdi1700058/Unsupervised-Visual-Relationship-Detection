#!/usr/bin/env python3
"""Summarise G1 and state what the comparison supports.

G1 asks two questions at once:

  1. Does a FOSAE model plan on clips it never saw as well as on clips it did?
  2. Is any margin between the two larger than the spread between three
     training runs of one configuration?

**Why this is a module and not a heredoc.** `tools/planner/e1_summary.py`
records the reason: the first E1 summary lived inline in a shell script, it
compared two arms that had reached very different fractions of their windows,
and it printed the opposite of what the data said. A module can be run against
synthetic rows before it ever sees real ones.

The four house rules it inherits from `e1_summary.py`:

1. **Solve rate leads.** Reaching a goal does not depend on how far the boxes
   move, so it is the axis least confounded by clip difficulty.
2. **Compare `mse_ratio`, never raw `bbox_mse` across arms.** `SPEC.md` V38
   measured that raw planner error is dominated by how non-linear a clip
   happens to be. Raw error appears in the table, marked as not comparable.
3. **An arm reaching under half its windows has its errors labelled
   optimistically biased.** The windows it reached are its easiest.
4. **Count distinct windows, not CSV rows.** A row is one window scored by one
   method, and two methods run here. Row counting doubled every E1 figure once.

And one rule of its own:

5. **A window key is `(clip, init, goal)`, not `(init, goal)`.** Each arm is a
   directory of 18 per-clip exports, so frame index 0 exists 18 times over. The
   E1 key would collapse 18 distinct windows into one.

Python 3.6 clean, standard library only.
"""

import csv
import os
import statistics as st


# Below this fraction of windows reached, an arm's error median describes a
# self-selected sample and must be labelled as such.
SELECTION_FLOOR = 0.5

SEEDS = (1, 2, 3)
ARMS = ("seen18", "test18")

LABEL = {"seen18": "in-sample (seen18)", "test18": "held out (test18)"}


def read(seed, arm, root="eval/planner"):
    """One seed of one arm, or None when the run is absent."""
    path = os.path.join(root, "G1-seed%d-%s" % (seed, arm), "summary.csv")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    return summarise_rows(rows)


def _window_of(row):
    """The window a row scores.

    The `export` column carries the per-clip export stem, so it identifies the
    clip. Without it every clip's frame 0 would be the same window.
    """
    return (row.get("export"), row.get("init"), row.get("goal"))


def _best(rows):
    """The better of a window's rows, by mse_ratio, lowest first."""
    def key(r):
        value = (r.get("mse_ratio") or "").strip()
        return float(value) if value not in ("", "None") else float("inf")
    return min(rows, key=key)


def summarise_rows(rows):
    """The numbers one seed of one arm contributes, counted per window."""
    by_window = {}
    for r in rows:
        by_window.setdefault(_window_of(r), []).append(r)

    windows = len(by_window)
    solved_windows, live = [], []
    for key, group in by_window.items():
        reached = [r for r in group
                   if (r.get("reachability") or "").strip().lower() == "true"]
        if not reached:
            continue
        solved_windows.append(key)
        scoreable = [r for r in reached
                     if r.get("moving_gt_steps")
                     and float(r["moving_gt_steps"]) >= 6
                     and r.get("bbox_mse")]
        if scoreable:
            live.append(_best(scoreable))

    ratios = [float(r["mse_ratio"]) for r in live
              if (r.get("mse_ratio") or "").strip() not in ("", "None")]
    return {
        "windows": windows,
        "solved": len(solved_windows),
        "scored": len(live),
        "solve_rate": (float(len(solved_windows)) / windows) if windows else 0.0,
        "ratio": st.median(ratios) if ratios else None,
        "mse": st.median([float(r["bbox_mse"]) for r in live]) if live else None,
        "base": st.median([float(r["baseline_mse"]) for r in live])
                if live else None,
        "iou": st.median([float(r["bbox_iou"]) for r in live if r["bbox_iou"]])
               if live else None,
        "beats": sum(1 for r in live if r.get("beats_baseline") == "True"),
    }


def _spread(values):
    """(median, lowest, highest, highest - lowest), or None for an empty list."""
    if not values:
        return None
    return (st.median(values), min(values), max(values), max(values) - min(values))


def combine(per_seed):
    """One arm across its seeds. `per_seed` holds only the seeds that ran."""
    live = [a for a in per_seed if a is not None]
    if not live:
        return None
    rates = _spread([a["solve_rate"] for a in live])
    ratios = _spread([a["ratio"] for a in live if a["ratio"] is not None])
    mses = _spread([a["mse"] for a in live if a["mse"] is not None])
    bases = _spread([a["base"] for a in live if a["base"] is not None])
    ious = _spread([a["iou"] for a in live if a["iou"] is not None])
    return {
        "seeds": len(live),
        "windows": sum(a["windows"] for a in live),
        "solved": sum(a["solved"] for a in live),
        "scored": sum(a["scored"] for a in live),
        "beats": sum(a["beats"] for a in live),
        "rate": rates,
        "ratio": ratios,
        "mse": mses,
        "base": bases,
        "iou": ious,
        "per_seed": live,
    }


def reading(seen, test):
    """What the comparison supports, as (verdict, list of caveats).

    `seen` is the in-sample arm and `test` the held-out one. Both are the
    output of `combine`.

    The rule, fixed before the run: a held-out drop counts only when it is
    larger than the spread between the three training runs.
    """
    caveats = []
    if seen is None or test is None:
        return ("**No reading.** One arm produced no summary at all.", caveats)

    if seen["solved"] == 0 and test["solved"] == 0:
        return ("**Reading: no window was solved in either arm.** The likeliest "
                "cause is that the model did not train at 70 clips, which says "
                "nothing about held-out performance. Record the cost and give "
                "the next attempt more epochs and more memory. One failure is a "
                "cost, not a verdict.", caveats)

    drop = seen["rate"][0] - test["rate"][0]
    noise = max(seen["rate"][3], test["rate"][3])

    if seen["seeds"] < 3 or test["seeds"] < 3:
        caveats.append("Only %d of 3 seeds landed in the in-sample arm and %d "
                       "of 3 in the held-out arm, so the noise figure rests on "
                       "fewer runs than the design asked for and understates "
                       "the true spread."
                       % (seen["seeds"], test["seeds"]))

    if noise == 0:
        caveats.append("**The seed spread is exactly zero**, so the comparison "
                       "against noise above carries no weight. Either one seed "
                       "landed, or every seed produced the same solve rate to "
                       "the window. Check the independence report at the top of "
                       "`score_local.sh` before reading anything into the "
                       "margin.")

    if test["solved"] == 0:
        verdict = ("**Reading: the held-out arm solved no window, while the "
                   "in-sample arm reached %.0f%% of its own.** The operator set "
                   "learned on 70 clips does not reach states taken from clips "
                   "the model never saw, so the code is clip specific. That is "
                   "a measured negative with a named cause. The next experiment "
                   "asks whether more clips or fewer bits changes it, and "
                   "assumes neither."
                   % (100 * seen["rate"][0]))
    elif drop > noise:
        verdict = ("**Reading: the held-out arm loses more than the seeds "
                   "explain.** In-sample reaches %.0f%% of its windows and "
                   "held out reaches %.0f%%, a drop of %.0f points against a "
                   "seed spread of %.0f points. Part of every earlier score was "
                   "memorisation. Relabel the earlier headlines as in-sample "
                   "and report the held-out number beside them."
                   % (100 * seen["rate"][0], 100 * test["rate"][0],
                      100 * drop, 100 * noise))
    elif drop < -noise:
        verdict = ("**Reading: the held-out arm scores BETTER than the "
                   "in-sample arm, by more than the seeds explain** (%.0f%% "
                   "against %.0f%%, a seed spread of %.0f points). Do not read "
                   "that as generalisation. The likeliest cause is that the 18 "
                   "held-out clips are easier by chance. Re-run the split under "
                   "a second seed string before anything is written down."
                   % (100 * test["rate"][0], 100 * seen["rate"][0],
                      100 * noise))
    else:
        verdict = ("**Reading: held-out performance sits inside the seed "
                   "noise.** In-sample reaches %.0f%% of its windows and held "
                   "out reaches %.0f%%, a difference of %.0f points against a "
                   "seed spread of %.0f points. At this scale the model plans "
                   "on VidVRD clips it never saw as well as on clips it did, so "
                   "the earlier headlines can be restated as held-out numbers."
                   % (100 * seen["rate"][0], 100 * test["rate"][0],
                      100 * drop, 100 * noise))

    # Error, and only as a ratio. Raw error across different clips measures the
    # clips (V38).
    if seen["ratio"] and test["ratio"]:
        gap = test["ratio"][0] - seen["ratio"][0]
        rnoise = max(seen["ratio"][3], test["ratio"][3])
        if rnoise == 0:
            caveats.append("On `mse_ratio` the held-out arm reads %.2f against "
                           "%.2f in sample. The seed spread on that axis is "
                           "zero, so the difference is not tested against "
                           "anything and it decides nothing."
                           % (test["ratio"][0], seen["ratio"][0]))
        elif abs(gap) <= rnoise:
            caveats.append("On `mse_ratio` the two arms differ by %.2f "
                           "(%.2f held out against %.2f in sample), inside the "
                           "seed spread of %.2f. That axis agrees with the "
                           "reading above."
                           % (abs(gap), test["ratio"][0], seen["ratio"][0],
                              rnoise))
        elif gap > 0:
            caveats.append("On `mse_ratio` the held-out arm is worse by %.2f "
                           "(%.2f against %.2f), which clears the seed spread "
                           "of %.2f."
                           % (gap, test["ratio"][0], seen["ratio"][0], rnoise))
        else:
            caveats.append("On `mse_ratio` the held-out arm is BETTER by %.2f "
                           "(%.2f against %.2f), which clears the seed spread "
                           "of %.2f. Treat that as a sign the two clip sets "
                           "differ in difficulty, not as a result."
                           % (-gap, test["ratio"][0], seen["ratio"][0], rnoise))

    # The seeds half of the experiment, reported whatever the held-out half says.
    if noise >= abs(drop) and noise > 0:
        caveats.append("**The seed spread is at least as large as the "
                       "difference between the arms** (%.0f points against %.0f)."
                       " One training run per arm was never enough to support a "
                       "margin, so every single-seed headline in this project "
                       "needs three seeds before it is reported again."
                       % (100 * noise, 100 * abs(drop)))

    # Selection bias, whenever an arm reached only part of its corpus.
    for arm in ("seen18", "test18"):
        data = seen if arm == "seen18" else test
        if data["rate"][0] < SELECTION_FLOOR:
            caveats.append("The %s error figures cover only the windows that "
                           "arm could reach, %d of %d across all seeds, which "
                           "are its easiest. Its errors are **optimistically "
                           "biased**, so the gap above is understated."
                           % (LABEL[arm], data["solved"], data["windows"]))
    return (verdict, caveats)


def _cell(triple, fmt="%.2f", pct=False):
    """A `median (lowest to highest)` cell from a `_spread` triple."""
    if triple is None:
        return "n/a"
    med, lo, hi, _ = triple
    scale = 100.0 if pct else 1.0
    return "%s (%s to %s)" % (fmt % (med * scale), fmt % (lo * scale),
                              fmt % (hi * scale))


def render(seen, test):
    """The whole summary document."""
    lines = ["# G1 - held-out clips, and three seeds per arm", ""]
    if seen is None or test is None:
        lines.append("One or both arms produced no summary. Nothing to compare.")
        return "\n".join(lines) + "\n"

    lines += [
        "Rates and error figures are medians across the seeds that landed, "
        "with the lowest and the highest in brackets. Counts are totals over "
        "those seeds. Windows are counted once each, keyed by clip and by "
        "frame pair; where both planners scored a window, the better result "
        "is used.",
        "",
        "| | in-sample (seen18) | held out (test18) |",
        "|---|---|---|",
        "| seeds that landed | %d of 3 | %d of 3 |" % (seen["seeds"], test["seeds"]),
        "| **windows reached, %%** | **%s** | **%s** |"
        % (_cell(seen["rate"], "%.0f", pct=True),
           _cell(test["rate"], "%.0f", pct=True)),
        "| windows reached, count over all seeds | %d of %d | %d of %d |"
        % (seen["solved"], seen["windows"], test["solved"], test["windows"]),
        "| windows with real motion | %d | %d |" % (seen["scored"], test["scored"]),
        "| **median `mse_ratio`** (lower is better) | **%s** | **%s** |"
        % (_cell(seen["ratio"]), _cell(test["ratio"])),
        "| beats the straight line | %d | %d |" % (seen["beats"], test["beats"]),
        "| trajectory IoU | %s | %s |"
        % (_cell(seen["iou"], "%.3f"), _cell(test["iou"], "%.3f")),
        "| raw planner error *(not comparable across arms)* | %s | %s |"
        % (_cell(seen["mse"]), _cell(test["mse"])),
        "| linear baseline *(shows why)* | %s | %s |"
        % (_cell(seen["base"]), _cell(test["base"])),
        "",
        "## Seed by seed",
        "",
        "| arm | seed | windows | reached | reached % | median `mse_ratio` |",
        "|---|---|---|---|---|---|",
    ]
    for arm, data in (("seen18", seen), ("test18", test)):
        for index, one in enumerate(data["per_seed"]):
            lines.append("| %s | %d | %d | %d | %.0f%% | %s |"
                         % (LABEL[arm], index + 1, one["windows"], one["solved"],
                            100 * one["solve_rate"],
                            "n/a" if one["ratio"] is None else "%.2f" % one["ratio"]))
    lines.append("")

    verdict, caveats = reading(seen, test)
    lines.append(verdict)
    if caveats:
        lines.append("")
        for c in caveats:
            lines.append("- %s" % c)
    return "\n".join(lines) + "\n"


def _bar_panel(x0, title, pairs, fmt, note):
    """One panel of bars with a low-to-high whisker across the seeds."""
    span = max([hi for _, _, _, hi in pairs] + [1e-9])
    out = ['<text x="%d" y="30" class="t">%s</text>' % (x0, title)]
    for i, (label, value, lo, hi) in enumerate(pairs):
        y = 58 + i * 72
        width = int(255 * value / span)
        fill = "#2b6cb0" if i == 0 else "#c05621"
        out.append('<text x="%d" y="%d" class="l">%s</text>' % (x0, y - 5, label))
        out.append('<rect x="%d" y="%d" width="%d" height="24" fill="%s"/>'
                   % (x0, y, max(width, 2), fill))
        wlo = int(255 * lo / span)
        whi = int(255 * hi / span)
        if whi > wlo:
            out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" '
                       'stroke="#1a202c" stroke-width="2"/>'
                       % (x0 + wlo, y + 12, x0 + whi, y + 12))
            for tick in (wlo, whi):
                out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" '
                           'stroke="#1a202c" stroke-width="2"/>'
                           % (x0 + tick, y + 4, x0 + tick, y + 20))
        out.append('<text x="%d" y="%d" class="v">%s</text>'
                   % (x0 + max(width, whi, 2) + 8, y + 17, fmt % value))
    out.append('<text x="%d" y="%d" class="n">%s</text>' % (x0, 226, note))
    return out


def render_svg(seen, test):
    """A two-panel chart: reachability, then error as a ratio.

    Hand-written SVG so it needs no plotting library and renders anywhere. The
    black whisker on each bar is the lowest-to-highest range across the seeds,
    which is the whole point of the seeds half: a bar whose whisker overlaps
    the other bar's whisker is not a margin.
    """
    if seen is None or test is None:
        return None
    w, h, pad = 760, 300, 46

    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
             'viewBox="0 0 %d %d">' % (w, h, w, h),
             '<style>text{font-family:sans-serif}'
             '.t{font-size:15px;font-weight:bold}.l{font-size:12px}'
             '.v{font-size:12px;font-weight:bold}.n{font-size:11px;fill:#555}'
             '.c{font-size:11px;fill:#555}</style>',
             '<rect width="%d" height="%d" fill="white"/>' % (w, h)]

    parts += _bar_panel(
        pad, "windows reached",
        [("in sample (seen18)", 100 * seen["rate"][0],
          100 * seen["rate"][1], 100 * seen["rate"][2]),
         ("held out (test18)", 100 * test["rate"][0],
          100 * test["rate"][1], 100 * test["rate"][2])],
        "%.0f%%",
        "%d of %d against %d of %d, over %d and %d seeds"
        % (seen["solved"], seen["windows"], test["solved"], test["windows"],
           seen["seeds"], test["seeds"]))

    if seen["ratio"] and test["ratio"]:
        rnoise = max(seen["ratio"][3], test["ratio"][3])
        parts += _bar_panel(
            pad + 370, "median mse_ratio, lower is better",
            [("in sample (seen18)", seen["ratio"][0],
              seen["ratio"][1], seen["ratio"][2]),
             ("held out (test18)", test["ratio"][0],
              test["ratio"][1], test["ratio"][2])],
            "%.2f",
            "gap %.2f against a seed spread of %.2f"
            % (abs(test["ratio"][0] - seen["ratio"][0]), rnoise))

    parts.append('<text x="%d" y="%d" class="c">%s</text>'
                 % (pad, h - 18,
                    "Whisker = lowest to highest across seeds. One number per "
                    "window, keyed by clip. VidVRD only, window 16."))
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    seen = combine([read(s, "seen18") for s in SEEDS])
    test = combine([read(s, "test18") for s in SEEDS])
    text = render(seen, test)
    if not os.path.isdir("eval/planner"):
        os.makedirs("eval/planner")
    out = "eval/planner/G1_summary.md"
    with open(out, "w") as handle:
        handle.write(text)
    print(text)
    print("wrote %s" % out)
    svg = render_svg(seen, test)
    if svg:
        with open("eval/planner/G1_summary.svg", "w") as handle:
            handle.write(svg)
        print("wrote eval/planner/G1_summary.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
