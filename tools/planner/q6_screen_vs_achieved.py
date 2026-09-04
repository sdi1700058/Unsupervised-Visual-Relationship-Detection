#!/usr/bin/env python3
"""Does the winnability screen predict what the planner achieves on VidOR?

`tools/video/screen_vidor.py` ranks a clip by `crossover = quantisation floor /
linear baseline`, and calls the clip winnable below 1. That is a statement about
the **clip**, computed from annotation alone in source pixels over every
annotated object. The planner is scored on an oracle export, which keeps a fixed
number of objects in canvas space with absent slots, so the two are not
guaranteed to rank the same trajectories. The first VidOR clip scored ranked
12th of 315 on the screen and lost every window, which is what raised the
question.

This module pairs, per clip, the screened crossover against the median
`mse_ratio` the planner actually reached, and reports the Spearman rank
correlation between them.

**Counting.** One number per window, keyed on `(init, goal)`. A row in
`summary.csv` is one window scored by one method, and counting rows doubles
every figure once a second method runs. Where a window carries several rows the
lowest `mse_ratio` is used, matching `tools/planner/e1_summary.py`.

**Filters**, each with a reason:

- `reachability` true, case-insensitively. The writer emits both `True` and
  `false`, and an unreachable window has no plan to score.
- `mse_ratio` present and not the string `None`.
- `moving_gt_steps >= 6`. A window whose boxes barely move is won by the
  identity, so it carries no signal either way.

**Sign convention, because it inverts.** A low crossover means the clip is more
winnable, and a low `mse_ratio` means the planner did better. So the screen
ranking correctly shows up as a **positive** correlation between the two.

**Read the whole run, not a growing file.** The scorer appends one row per
window while it works, so a `summary.csv` opened mid-run holds a prefix of the
clip rather than a sample of it. Clip `0019/2775487424` read at 9 windows gave
median 3.42 with nothing beating the baseline; the same clip at 12 windows gives
2.76 with one window beating. The prefix is not wrong, it is early. This module
records how many rows each clip contributed so a number can be checked against
the file it came from.

**What the correlation can and cannot say.** The sample is about 25 clips, all
of them drawn from the winnable end of the screen, so the crossover range is
restricted by construction. A restricted range attenuates a rank correlation
toward zero, and 25 clips is a small sample. So a weak correlation here is
consistent with a screen that works over the whole corpus but cannot resolve
differences inside its own top 10%. The number is reported with its `n` and a
permutation p-value, and no verdict is attached.

Standard library only. Numpy is needed only to recompute crossover, which is
optional. Python 3.6 clean.

    python3 tools/planner/q6_screen_vs_achieved.py
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir))

RUN_GLOB = "eval/planner/vidor-*/summary.csv"
ANN_ROOT = "data/video/vidor/annotations/training"
SCREEN_LIST = "eval/vidor_winnable_w16.txt"
OUT_JSON = "eval/planner/vidor_screen_vs_achieved.json"
OUT_SVG = "eval/planner/vidor_screen_vs_achieved.svg"

# The screen that produced eval/vidor_winnable_w16.txt. Recomputing at any
# other window would compare against a different criterion, which is the
# mistake SPEC.md V37 records.
WINDOW = 16

MIN_MOVING_GT_STEPS = 6


def truthy(value):
    """`reachability` and `beats_baseline` arrive as both `True` and `false`."""
    return (value or "").strip().lower() == "true"


def number(value):
    """A float, or None for the empty string and the literal `None`."""
    text = (value or "").strip()
    if text in ("", "None", "none", "nan"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def clip_id_of(run_dir):
    """`eval/planner/vidor-0019-2775487424` -> `0019/2775487424`."""
    name = os.path.basename(run_dir.rstrip("/"))
    if name.startswith("vidor-"):
        name = name[len("vidor-"):]
    return name.replace("-", "/", 1)


def windows_of(path):
    """`{(init, goal): row}` after the filters, one row per window.

    Returns the kept windows and a count of what each filter removed, so a
    clip that contributes nothing says why.
    """
    with open(path) as handle:
        rows = list(csv.DictReader(handle))

    dropped = {"unreachable": 0, "no_ratio": 0, "static": 0}
    kept = {}
    for row in rows:
        if not truthy(row.get("reachability")):
            dropped["unreachable"] += 1
            continue
        ratio = number(row.get("mse_ratio"))
        if ratio is None:
            dropped["no_ratio"] += 1
            continue
        moving = number(row.get("moving_gt_steps"))
        if moving is None or moving < MIN_MOVING_GT_STEPS:
            dropped["static"] += 1
            continue
        key = (row.get("init"), row.get("goal"))
        previous = kept.get(key)
        if previous is None or ratio < number(previous.get("mse_ratio")):
            kept[key] = row
    return kept, dropped, len(rows)


def achieved(run_dir):
    """One clip's achieved numbers, counted per window."""
    kept, dropped, rows = windows_of(os.path.join(run_dir, "summary.csv"))
    ratios = [number(r.get("mse_ratio")) for r in kept.values()]
    floors = [number(r.get("floor_ratio")) for r in kept.values()]
    floors = [f for f in floors if f is not None]
    return {
        "clip": clip_id_of(run_dir),
        "run": run_dir,
        "rows": rows,
        "windows": len(kept),
        "dropped": dropped,
        "median_mse_ratio": st.median(ratios) if ratios else None,
        "min_mse_ratio": min(ratios) if ratios else None,
        "median_floor_ratio": st.median(floors) if floors else None,
        "windows_beating": sum(1 for r in kept.values()
                               if truthy(r.get("beats_baseline"))),
        "mse_ratios": sorted(ratios),
    }


def screened(clip_ids, window=WINDOW):
    """`{clip_id: crossover}`, recomputed from the annotation on disk."""
    from tools.video.screen_vidor import screen

    out = {}
    for clip_id in clip_ids:
        path = os.path.join(ANN_ROOT, "%s.json" % clip_id)
        if not os.path.exists(path):
            continue
        rows = screen([path], window=window)
        if rows:
            out[clip_id] = rows[0][0]
    return out


def screen_ranks(path=SCREEN_LIST):
    """`{clip_id: 1-based position}` in the most-winnable-first list."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                out.setdefault(line, len(out) + 1)
    return out


def ranks(values):
    """Ascending ranks, ties sharing their mean rank. Standard library."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = mean
        i = j + 1
    return out


def pearson(xs, ys):
    """Pearson r, or None when either side has no variance."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / float(n), sum(ys) / float(n)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    sxx = sum(v * v for v in dx)
    syy = sum(v * v for v in dy)
    if sxx <= 0 or syy <= 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(sxx * syy)


def spearman(xs, ys):
    """Pearson r on the ranks. The tie-corrected definition."""
    return pearson(ranks(xs), ranks(ys))


def permutation_p(xs, ys, trials=20000, seed=1):
    """Two-sided p for Spearman, by shuffling one side.

    `scipy` is not available and `statistics.NormalDist` is 3.8, so the
    t-approximation is out of reach. A permutation test needs neither and is
    exact in the limit, which at n around 25 is the honest option.
    """
    observed = spearman(xs, ys)
    if observed is None:
        return None
    rx, ry = ranks(xs), list(ranks(ys))
    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        rng.shuffle(ry)
        value = pearson(rx, ry)
        if value is not None and abs(value) >= abs(observed) - 1e-12:
            hits += 1
    return (hits + 1) / float(trials + 1)


def collect(run_glob=RUN_GLOB, window=WINDOW):
    """Every scored clip, paired with its screened crossover."""
    runs = sorted(os.path.dirname(p) for p in glob.glob(run_glob))
    rows = [achieved(r) for r in runs]
    scored = [r for r in rows if r["windows"] > 0]
    empty = [r for r in rows if r["windows"] == 0]

    crossovers = screened([r["clip"] for r in scored], window=window)
    listed = screen_ranks()
    paired, unscreened = [], []
    for row in scored:
        if row["clip"] not in crossovers:
            unscreened.append(row["clip"])
            continue
        row = dict(row)
        row["crossover"] = crossovers[row["clip"]]
        row["screen_rank"] = listed.get(row["clip"])
        paired.append(row)
    paired.sort(key=lambda r: r["crossover"])
    return paired, empty, unscreened, len(runs)


def analyse(paired, empty, unscreened, runs, window=WINDOW):
    xs = [r["crossover"] for r in paired]
    ys = [r["median_mse_ratio"] for r in paired]
    pooled = [v for r in paired for v in r["mse_ratios"]]

    listed = [(r["screen_rank"], r["crossover"]) for r in paired
              if r["screen_rank"]]
    agreement = None
    if len(listed) >= 3:
        agreement = spearman([a for a, _ in listed], [b for _, b in listed])

    floors = [r["median_floor_ratio"] for r in paired
              if r["median_floor_ratio"] is not None]
    floor_rho = None
    if len(floors) == len(paired) and len(paired) >= 3:
        floor_rho = spearman(xs, floors)

    # A clip with one or two surviving windows has a median made of one or two
    # numbers, and it counts as much as a clip with eighteen. Re-running the
    # correlation over the better-measured clips says whether the result rests
    # on the thin ones.
    sensitivity = []
    for floor in (1, 3, 5, 8, 10):
        subset = [r for r in paired if r["windows"] >= floor]
        if len(subset) >= 3:
            sensitivity.append({
                "min_windows": floor,
                "n": len(subset),
                "spearman": spearman([r["crossover"] for r in subset],
                                     [r["median_mse_ratio"] for r in subset]),
            })

    # Win rate is the other achieved quantity, and it does not depend on one
    # window's outlier the way a median of three can. Here a screen that works
    # shows up **negative**: a lower crossover should win more often.
    win_rate = [r["windows_beating"] / float(r["windows"]) for r in paired]

    return {
        "question": "Q6: does the VidOR winnability screen predict the "
                    "mse_ratio the planner achieves?",
        "corpus": "vidor",
        "screen_window": window,
        "filters": {
            "reachability": "true, case-insensitive",
            "mse_ratio": "present and not None",
            "moving_gt_steps": ">= %d" % MIN_MOVING_GT_STEPS,
            "unit": "distinct window keyed on (init, goal); best row per window",
        },
        "runs_found": runs,
        "probe_exports_present": len(glob.glob("eval/probe/vidor/*.npz")),
        "n": len(paired),
        "clips_with_no_surviving_window": [r["clip"] for r in empty],
        "clips_missing_from_screen": unscreened,
        "sign_convention": "positive means the screen ranks correctly: a "
                           "higher crossover goes with a worse mse_ratio",
        "spearman_crossover_vs_median_mse_ratio": spearman(xs, ys),
        "permutation_p_two_sided": permutation_p(xs, ys),
        "permutation_trials": 20000,
        "spearman_crossover_vs_win_rate": spearman(xs, win_rate),
        "spearman_crossover_vs_median_floor_ratio": floor_rho,
        "sensitivity_by_min_windows": sensitivity,
        "screen_rank_agreement": agreement,
        "crossover_range": [min(xs), max(xs)] if xs else None,
        "median_crossover": st.median(xs) if xs else None,
        "median_of_clip_median_mse_ratio": st.median(ys) if ys else None,
        "pooled_window_median_mse_ratio": st.median(pooled) if pooled else None,
        "windows_total": len(pooled),
        "windows_beating_baseline": sum(r["windows_beating"] for r in paired),
        "clips_with_any_window_beating": sum(1 for r in paired
                                            if r["windows_beating"] > 0),
        "clips_with_median_below_one": sum(1 for r in paired
                                           if r["median_mse_ratio"] < 1.0),
        "best_clip": min(paired, key=lambda r: r["median_mse_ratio"])["clip"]
                     if paired else None,
        "clips": [{
            "clip": r["clip"],
            "crossover": r["crossover"],
            "screen_rank": r["screen_rank"],
            "windows": r["windows"],
            "median_mse_ratio": r["median_mse_ratio"],
            "min_mse_ratio": r["min_mse_ratio"],
            "median_floor_ratio": r["median_floor_ratio"],
            "windows_beating": r["windows_beating"],
            "csv_rows": r["rows"],
            "dropped": r["dropped"],
        } for r in paired],
    }


def render_svg(result):
    """Screened crossover against achieved median `mse_ratio`, one dot a clip.

    Hand-written so it needs no plotting library. The y axis is logarithmic
    because `mse_ratio` spans more than a decade across clips, and the line at
    `mse_ratio = 1` is drawn because it is the only threshold that means
    anything: below it the planner beat the straight line.
    """
    clips = result["clips"]
    if not clips:
        return None
    w, h = 760, 470
    left, right, top, bottom = 80, 40, 112, 92
    pw, ph = w - left - right, h - top - bottom

    xs = [c["crossover"] for c in clips]
    ys = [c["median_mse_ratio"] for c in clips]
    x0, x1 = min(xs), max(xs)
    span = (x1 - x0) or 1.0
    x0, x1 = x0 - 0.06 * span, x1 + 0.06 * span
    ly = [math.log10(max(v, 1e-6)) for v in ys] + [0.0]
    y0, y1 = math.floor(min(ly) * 2) / 2.0, math.ceil(max(ly) * 2) / 2.0
    if y1 - y0 < 0.5:
        y1 = y0 + 0.5

    def px(v):
        return left + pw * (v - x0) / float(x1 - x0)

    def py(v):
        return top + ph * (1.0 - (math.log10(max(v, 1e-6)) - y0) / (y1 - y0))

    rho = result["spearman_crossover_vs_median_mse_ratio"]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
             'viewBox="0 0 %d %d">' % (w, h, w, h),
             '<style>text{font-family:sans-serif}'
             '.t{font-size:15px;font-weight:bold}.l{font-size:12px}'
             '.v{font-size:12px;font-weight:bold}.n{font-size:11px;fill:#555}'
             '.c{font-size:11px;fill:#555}'
             '.a{font-size:11px;fill:#555}</style>',
             '<rect width="%d" height="%d" fill="white"/>' % (w, h),
             '<text x="%d" y="28" class="t">VidOR: winnability screen against '
             'what the planner reached</text>' % left,
             '<text x="%d" y="50" class="n">Spearman %s, n=%d clips, p=%s by '
             'permutation. Positive means the screen ranked correctly.</text>'
             % (left, ("%+.3f" % rho) if rho is not None else "n/a",
                result["n"],
                ("%.3f" % result["permutation_p_two_sided"])
                if result["permutation_p_two_sided"] is not None else "n/a"),
             '<text x="%d" y="68" class="n">One dot a clip. y is the median '
             'over distinct windows, not over CSV rows: %d windows in '
             'total.</text>' % (left, result["windows_total"]),
             '<text x="%d" y="86" class="n">Every clip comes from the winnable '
             'end of the screen, so the x range is restricted by '
             'construction.</text>' % left]

    # frame and gridlines
    parts.append('<rect x="%d" y="%d" width="%d" height="%d" fill="none" '
                 'stroke="#cbd5e0"/>' % (left, top, pw, ph))
    tick = y0
    while tick <= y1 + 1e-9:
        y = top + ph * (1.0 - (tick - y0) / (y1 - y0))
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" '
                     'stroke="#edf2f7"/>' % (left, y, left + pw, y))
        parts.append('<text x="%d" y="%.1f" class="a" text-anchor="end">%s'
                     '</text>' % (left - 8, y + 4, _fmt(10 ** tick)))
        tick += 0.5
    for i in range(5):
        v = x0 + (x1 - x0) * i / 4.0
        x = px(v)
        parts.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" '
                     'stroke="#edf2f7"/>' % (x, top, x, top + ph))
        parts.append('<text x="%.1f" y="%d" class="a" text-anchor="middle">'
                     '%.2f</text>' % (x, top + ph + 18, v))

    # the only threshold that means anything
    parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e53e3e" '
                 'stroke-dasharray="5,4"/>'
                 % (left, py(1.0), left + pw, py(1.0)))
    parts.append('<text x="%d" y="%.1f" class="n" fill="#e53e3e">mse_ratio = 1'
                 ', the linear baseline</text>' % (left + 6, py(1.0) - 6))

    for clip in clips:
        x, y = px(clip["crossover"]), py(clip["median_mse_ratio"])
        won = clip["windows_beating"] > 0
        parts.append('<circle cx="%.1f" cy="%.1f" r="5" fill="%s" '
                     'stroke="white"/>'
                     % (x, y, "#2b6cb0" if won else "#a0aec0"))
        if clip["screen_rank"]:
            parts.append('<text x="%.1f" y="%.1f" class="a" '
                         'text-anchor="middle">%d</text>'
                         % (x, y - 9, clip["screen_rank"]))

    parts.append('<text x="%d" y="%d" class="l" text-anchor="middle">screened '
                 'crossover, lower is more winnable</text>'
                 % (left + pw // 2, top + ph + 40))
    parts.append('<text x="18" y="%d" class="l" transform="rotate(-90 18 %d)" '
                 'text-anchor="middle">achieved median mse_ratio</text>'
                 % (top + ph // 2, top + ph // 2))
    parts.append('<text x="%d" y="%d" class="c">Blue: at least one window beat '
                 'the baseline (%d of %d clips). Grey: none. Small number is '
                 'the clip rank on the screen.</text>'
                 % (left, h - 16, result["clips_with_any_window_beating"],
                    result["n"]))
    parts.append("</svg>")
    return "\n".join(parts)


def _fmt(value):
    if value >= 10:
        return "%.0f" % value
    if value >= 1:
        return "%.1f" % value
    return "%.2f" % value


def report(result):
    lines = ["Q6 VidOR: screen against achieved, %d clips" % result["n"]]
    rho = result["spearman_crossover_vs_median_mse_ratio"]
    lines.append("  spearman crossover vs median mse_ratio  %s (p=%s)"
                 % (("%+.3f" % rho) if rho is not None else "n/a",
                    ("%.3f" % result["permutation_p_two_sided"])
                    if result["permutation_p_two_sided"] is not None else "n/a"))
    lines.append("  spearman crossover vs win rate          %s"
                 % (("%+.3f" % result["spearman_crossover_vs_win_rate"])
                    if result["spearman_crossover_vs_win_rate"]
                    is not None else "n/a"))
    lines.append("  by min windows per clip                 %s"
                 % ", ".join("%d:%+.3f(n=%d)"
                             % (s["min_windows"], s["spearman"], s["n"])
                             for s in result["sensitivity_by_min_windows"]))
    lines.append("  spearman crossover vs median floor_ratio %s"
                 % (("%+.3f" % result["spearman_crossover_vs_median_floor_ratio"])
                    if result["spearman_crossover_vs_median_floor_ratio"]
                    is not None else "n/a"))
    lines.append("  screen rank agreement                   %s"
                 % (("%+.3f" % result["screen_rank_agreement"])
                    if result["screen_rank_agreement"] is not None else "n/a"))
    lines.append("  crossover range                         %.3f to %.3f"
                 % tuple(result["crossover_range"]))
    lines.append("  median of clip median mse_ratio         %.3f"
                 % result["median_of_clip_median_mse_ratio"])
    lines.append("  pooled window median mse_ratio          %.3f over %d windows"
                 % (result["pooled_window_median_mse_ratio"],
                    result["windows_total"]))
    lines.append("  windows beating the baseline            %d of %d"
                 % (result["windows_beating_baseline"],
                    result["windows_total"]))
    lines.append("  clips with any window beating           %d of %d"
                 % (result["clips_with_any_window_beating"], result["n"]))
    lines.append("  runs found / clips with no window       %d / %d"
                 % (result["runs_found"],
                    len(result["clips_with_no_surviving_window"])))
    return "\n".join(lines)


def self_check():
    """Four closed-form cases for the rank machinery.

    A wrong correlation would be a headline number with nothing to catch it,
    and every case here can be worked out by hand from the rank-difference
    formula, so this is a check rather than a restatement of the code.
    """
    cases = [
        (ranks([10, 20, 20, 30]), [1.0, 2.5, 2.5, 4.0], "ties share a rank"),
        (spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 1.0, "identical"),
        (spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), -1.0, "reversed"),
        # 1 - 6*4/(5*24) = 0.8
        (spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]), 0.8, "two swaps"),
        # 4.5 / sqrt(4.5*5.0)
        (spearman([1, 2, 2, 3], [1, 3, 2, 4]), 0.9486832980505138, "tied x"),
    ]
    bad = 0
    for got, want, name in cases:
        ok = (got == want if isinstance(want, list)
              else abs(got - want) < 1e-12)
        print("%-18s %s" % (name, "ok" if ok else "WRONG %r != %r"
                                                 % (got, want)))
        bad += 0 if ok else 1
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--glob", default=RUN_GLOB)
    ap.add_argument("--self-check", action="store_true",
                    help="verify the rank correlation against hand-worked cases")
    a = ap.parse_args(argv)

    if a.self_check:
        return self_check()

    paired, empty, unscreened, runs = collect(a.glob, a.window)
    if not paired:
        print("no scored VidOR clip survived the filters; looked at %s"
              % a.glob)
        return 2
    result = analyse(paired, empty, unscreened, runs, a.window)
    print(report(result))

    directory = os.path.dirname(OUT_JSON)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(OUT_JSON, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print("wrote %s" % OUT_JSON)
    svg = render_svg(result)
    if svg:
        with open(OUT_SVG, "w") as handle:
            handle.write(svg)
        print("wrote %s" % OUT_SVG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
