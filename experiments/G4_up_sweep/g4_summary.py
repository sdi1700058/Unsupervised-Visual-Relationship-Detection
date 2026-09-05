#!/usr/bin/env python3
"""Summarise the G4 grid over U and P, and draw it.

Two axes, and the whole point of the figure is that they can disagree.
`EVAL.md` 5.7 and 5.8 argue that an auto-encoding objective does not select
for plannability, so a cell can reconstruct well and plan badly. A grid that
showed only one axis would hide exactly the thing G4 is asking about.

Outputs:

    eval/planner/G4_summary.md    the table and the pre-registered reading
    eval/planner/G4_summary.svg   three panels, hand-written, no matplotlib

This is a module and not a heredoc for the reason `tools/planner/e1_summary.py`
gives at length: the first inline version of that comparison reported the
opposite of what its data said, and a module can be read, reviewed and tested.
Its `summarise_rows` is imported here rather than copied, so the per-window
best-of-two-planners rule stays in one place.

Python 3.6 clean, standard library plus numpy. numpy is optional; without it
the round-trip reconstruction column and the clip-boundary filter are dropped
and the rest still renders.
"""

import csv
import glob
import math
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools", "planner"))

from e1_summary import summarise_rows  # noqa: E402

try:
    import numpy as np
except ImportError:
    np = None


# ── the pre-registered thresholds ───────────────────────────────────────────
# Every one of these is fixed in README.md before the run. Changing one after
# reading the data turns the experiment into a search for a story.

# SPEC C17: plannability is defined only on models with val_BCE below this.
PLANNABLE_VAL_LOSS = 0.5

# A difference in median mse_ratio smaller than this is not material. Planner
# error on this task moves by more than this between reruns of one clip set.
MATERIAL = 1.5

# Spearman rank correlation between the two axes, over the cells that carry
# both. At or above the first, reconstruction ranks the cells the way planning
# does. At or below the second, the two disagree.
RHO_AGREE = 0.5
RHO_DISAGREE = 0.0

CELL_RE = re.compile(r"^U(\d+)_A(\d+)_P(\d+)_cat")
RUN_DIR_GLOB = "eval/planner/G4-U*-P*"
EXPORT_GLOB = "eval/exports/U*_A*_P*_catH14-winnable*.npz"
TRAIN_CSV = "eval/exports/G4_train.csv"


# ── reading the run ─────────────────────────────────────────────────────────
def clip_of(frame_id):
    """The clip a frame belongs to. frame_ids read '<clip>/<frame_no>'."""
    text = str(frame_id)
    return text.rsplit("/", 1)[0] if "/" in text else text


def export_facts(path):
    """Round-trip reconstruction error and the clip index, from one export.

    The round-trip error is the decoder's own error on real frames: no planner,
    no search. It is in the same canvas units as the planner's `bbox_mse`, so
    the two are directly comparable, and it is always available because it
    needs nothing but the export.
    """
    if np is None:
        return {"boxmse": None, "clips": None, "frames": None}
    # No allow_pickle. The export holds arrays and scalars only, so the pickle
    # path is not needed and refusing it keeps a hostile npz from executing.
    data = np.load(path)
    out = {"boxmse": None, "clips": None, "frames": None}
    if "latents" in data.files:
        out["frames"] = int(data["latents"].shape[0])
    if "decoded_boxes" in data.files and "gt_boxes" in data.files:
        dec = np.asarray(data["decoded_boxes"], dtype="float64")
        gt = np.asarray(data["gt_boxes"], dtype="float64")
        if dec.shape == gt.shape and dec.size:
            out["boxmse"] = float(np.mean((dec - gt) ** 2))
    if "frame_ids" in data.files:
        out["clips"] = [clip_of(f) for f in data["frame_ids"]]
    return out


def read_train_csv(path=TRAIN_CSV):
    """(U, P) -> the training row the cluster recorded, or an empty dict."""
    if not os.path.exists(path):
        return {}
    table = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            try:
                key = (int(row["U"]), int(row["P"]))
            except (KeyError, TypeError, ValueError):
                continue
            best = (row.get("best_val") or "").strip()
            try:
                best = float(best)
            except ValueError:
                best = None
            try:
                epochs = int(row.get("epochs") or 0)
            except ValueError:
                epochs = 0
            table[key] = {"best_val": best, "epochs": epochs,
                          "metric": (row.get("metric") or "").strip(),
                          "moved": (row.get("moved") or "").strip() == "True"}
    return table


def split_windows(rows, clips):
    """Split rows into (within one clip, spanning two clips).

    The window slider in `tools/planner/common/windows.py` walks a frame index
    and knows nothing about clip boundaries. The G4 export concatenates 88
    clips, so a window can start in one clip and end in the next. Such a window
    scores a cut, not motion, and its baseline and its planner error are both
    meaningless. The count is reported rather than hidden, because it is a
    property of the harness that every cell shares.
    """
    if not clips:
        return rows, []
    keep, cross = [], []
    for row in rows:
        try:
            init = int(float(row["init"]))
            goal = int(float(row["goal"]))
        except (KeyError, TypeError, ValueError):
            keep.append(row)
            continue
        if init >= len(clips) or goal >= len(clips):
            keep.append(row)
        elif clips[init] == clips[goal]:
            keep.append(row)
        else:
            cross.append(row)
    return keep, cross


def collect(root=ROOT):
    """One record per (U, P) cell that produced anything."""
    exports = {}
    for path in sorted(glob.glob(os.path.join(root, EXPORT_GLOB))):
        match = CELL_RE.match(os.path.basename(path))
        if match:
            exports[(int(match.group(1)), int(match.group(3)))] = path

    trained = read_train_csv(os.path.join(root, TRAIN_CSV))
    cells = {}
    for path in sorted(glob.glob(os.path.join(root, RUN_DIR_GLOB))):
        match = re.search(r"G4-U(\d+)-P(\d+)$", os.path.basename(path))
        summary = os.path.join(path, "summary.csv")
        if not match or not os.path.exists(summary):
            continue
        key = (int(match.group(1)), int(match.group(2)))
        with open(summary) as handle:
            rows = list(csv.DictReader(handle))
        facts = export_facts(exports[key]) if key in exports else \
            {"boxmse": None, "clips": None, "frames": None}
        kept, cross = split_windows(rows, facts["clips"])
        record = summarise_rows(kept)
        record.update({
            "u": key[0], "p": key[1], "bits": key[0] * key[1],
            "boxmse": facts["boxmse"], "frames": facts["frames"],
            "cross_rows": len(cross),
        })
        record.update(trained.get(key, {"best_val": None, "epochs": 0,
                                        "metric": "", "moved": None}))
        cells[key] = record

    # A cell that trained but was never planned still belongs in the table:
    # a blank planning column is a result, not a gap.
    for key, info in trained.items():
        if key in cells:
            continue
        cells[key] = {"u": key[0], "p": key[1], "bits": key[0] * key[1],
                      "windows": 0, "solved": 0, "scored": 0, "solve_rate": 0.0,
                      "ratio": None, "mse": None, "base": None, "iou": None,
                      "beats": 0, "boxmse": None, "frames": None,
                      "cross_rows": 0}
        cells[key].update(info)
    return [cells[k] for k in sorted(cells)]


# ── statistics, standard library only ───────────────────────────────────────
def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Rank correlation. None when there are too few points to mean anything."""
    if len(xs) < 4:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def recon_axis(cells):
    """Which reconstruction number the figure uses, and what it is called.

    Training loss when the cluster recorded it, because that is the quantity
    `EVAL.md` 5.7 makes its claim about. Round-trip box error otherwise,
    because it needs only the export and is therefore never missing.
    """
    if any(c.get("best_val") is not None for c in cells):
        return "best_val", "val loss (training)", True
    return "boxmse", "round-trip box MSE", False


# ── the pre-registered reading ──────────────────────────────────────────────
def reading(cells):
    """What the grid supports. Every threshold is fixed in README.md."""
    key, label, is_val_loss = recon_axis(cells)
    scored = [c for c in cells if c["ratio"] is not None]
    lines = []

    if not scored:
        lines.append(
            "**Reading: no cell scored a single window.** The grid says "
            "nothing about U or P. Check `G4_train.csv` first: a loss that "
            "never moved means the runs produced nothing, and the limit is "
            "then upstream of the latent shape.")
        return lines

    if is_val_loss:
        with_loss = [c for c in cells if c.get("best_val") is not None]
        plannable = [c for c in with_loss
                     if c["best_val"] < PLANNABLE_VAL_LOSS]
        if with_loss and not plannable:
            lines.append(
                "**Reading: no cell reconstructs well enough to be read.** "
                "Every one of the %d cells that reported a training loss "
                "sits at or above %.1f, and SPEC C17 defines plannability "
                "only below that. The grid is uninformative about U and P "
                "at this data volume."
                % (len(with_loss), PLANNABLE_VAL_LOSS))
            return lines

    # 1. Do the two axes agree?
    pairs = [(c[key], c["ratio"]) for c in cells
             if c.get(key) is not None and c["ratio"] is not None]
    rho = spearman([a for a, _ in pairs], [b for _, b in pairs])
    if rho is None:
        lines.append(
            "**The two axes cannot be compared.** Only %d cell(s) carry both "
            "a reconstruction number and a planning number, which is too few "
            "for a rank correlation." % len(pairs))
    elif rho >= RHO_AGREE:
        lines.append(
            "**Reconstruction ranks the cells the way planning does** "
            "(Spearman %.2f over %d cells). On this grid %s is a usable "
            "proxy for plannability, which would let model selection skip "
            "the planner. That contradicts the general argument in "
            "`EVAL.md` 5.7, so report it as a property of this grid rather "
            "than a refutation." % (rho, len(pairs), label))
    elif rho <= RHO_DISAGREE:
        lines.append(
            "**The two axes disagree** (Spearman %.2f over %d cells, on %s). "
            "A cell can reconstruct well and plan badly. This is `EVAL.md` 5.7 "
            "and the A4 argument confirmed inside this project's own data, on "
            "one grid, with one dataset held fixed. Selecting a latent shape "
            "by %s is then wrong, and the planner has to run."
            % (rho, len(pairs), label, label))
    else:
        lines.append(
            "**The two axes are weakly related** (Spearman %.2f over %d "
            "cells, on %s). Neither the proxy claim nor its denial is "
            "supported. Reconstruction loss should not be used to choose a "
            "cell, but the grid does not prove it misleads."
            % (rho, len(pairs), label))

    # 2. Does U matter, and in which direction?
    low = [c["ratio"] for c in scored if c["u"] <= 10]
    high = [c["ratio"] for c in scored if c["u"] >= 40]
    if low and high:
        lo, hi = st.median(low), st.median(high)
        if hi >= lo * MATERIAL:
            lines.append(
                "**A narrow latent plans better.** Median `mse_ratio` %.2f at "
                "U of 10 or less against %.2f at U of 40 or more, a factor of "
                "%.2f, past the %.1fx materiality bar. U=40 was never chosen "
                "and never tested; on this evidence it handicapped every "
                "earlier planning number. Rerun the decisive arms at the best "
                "U and re-open the H14 result." % (lo, hi, hi / lo, MATERIAL))
        elif lo >= hi * MATERIAL:
            lines.append(
                "**A wide latent plans better.** Median `mse_ratio` %.2f at U "
                "of 10 or less against %.2f at U of 40 or more, a factor of "
                "%.2f. Capacity binds before search width does, so extend the "
                "grid upward rather than downward." % (lo, hi, lo / hi))
        else:
            lines.append(
                "**U does not change planning quality here.** Median "
                "`mse_ratio` %.2f at U of 10 or less against %.2f at U of 40 "
                "or more, inside the %.1fx materiality bar. The latent width "
                "is not the limiting parameter, so stop tuning it."
                % (lo, hi, MATERIAL))
    else:
        lines.append(
            "**The U trend cannot be read.** Scored cells do not cover both "
            "ends of the U ladder.")

    # 3. Shape or size? Cells with equal bits but different (U, P).
    groups = {}
    for c in scored:
        groups.setdefault(c["bits"], []).append(c)
    multi = [(bits, g) for bits, g in groups.items() if len(g) >= 2]
    if multi:
        worst_bits, worst_spread = None, 1.0
        for bits, group in multi:
            ratios = [c["ratio"] for c in group]
            lowest = min(ratios)
            spread = (max(ratios) / lowest) if lowest > 0 else float("inf")
            if spread > worst_spread:
                worst_bits, worst_spread = bits, spread
        if worst_bits is not None and worst_spread >= MATERIAL:
            lines.append(
                "**The shape matters, not only the size.** At %d bits the "
                "cells differ by a factor of %.2f in `mse_ratio` while "
                "holding the bit count fixed. U and P are two parameters, not "
                "one, and U*P is not a sufficient description of the latent."
                % (worst_bits, worst_spread))
        else:
            lines.append(
                "**Only the bit count appears to matter.** Cells with equal "
                "U*P and different shapes agree to within %.2fx, under the "
                "%.1fx bar. U and P can be collapsed into one knob for the "
                "purpose of planning." % (worst_spread, MATERIAL))
    else:
        lines.append(
            "**Shape against size cannot be read.** No two scored cells share "
            "a bit count.")
    return lines


def render_markdown(cells):
    key, label, is_val_loss = recon_axis(cells)
    lines = [
        "# G4 -- the grid over U and P",
        "",
        "One dataset (VidVRD, the 88 screened clips), one configuration, one "
        "window (16). The only thing that differs between rows is the latent "
        "shape.",
        "",
    ]
    if not cells:
        lines.append("No cell produced anything. Nothing to compare.")
        return "\n".join(lines) + "\n"

    # The round-trip column is the fallback axis, so it is not repeated when
    # it IS the axis.
    head = ["U", "P", "bits"]
    if is_val_loss:
        head.append(label)
    head += ["round-trip box MSE", "epochs", "windows reached",
             "median `mse_ratio`", "beats the line"]
    lines += ["| " + " | ".join(head) + " |",
              "|" + "---|" * len(head)]
    for c in cells:
        row = [str(c["u"]), str(c["p"]), str(c["bits"])]
        if is_val_loss:
            row.append("-" if c.get("best_val") is None
                       else "%.4g" % c["best_val"])
        row += [
            "-" if c.get("boxmse") is None else "%.1f" % c["boxmse"],
            "-" if not c.get("epochs") else str(c["epochs"]),
            "%d/%d" % (c["solved"], c["windows"]),
            "-" if c["ratio"] is None else "**%.2f**" % c["ratio"],
            str(c["beats"]),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    truncated = [c for c in cells
                 if c.get("epochs") and c["epochs"] < max(
                     d.get("epochs") or 0 for d in cells)]
    if truncated:
        lines.append(
            "**Warning: %d cell(s) trained for fewer epochs than the longest.** "
            "Every cell asked for identical resources, so a short cell was cut "
            "off or crashed. Its numbers describe the budget, not the latent "
            "shape." % len(truncated))
        lines.append("")

    crossed = sum(c.get("cross_rows") or 0 for c in cells)
    if crossed:
        lines.append(
            "%d scored row(s) were dropped for spanning two clips. The window "
            "slider walks a frame index and the export concatenates 88 clips, "
            "so a window can straddle a cut. Every cell loses the same "
            "windows." % crossed)
        lines.append("")

    # A blank line between each, or Markdown runs them into one paragraph.
    for paragraph in reading(cells):
        lines.append(paragraph)
        lines.append("")
    lines.append(
        "Read `mse_ratio` and not raw planner error: below 1 the planner beat "
        "straight-line interpolation between the two given frames.")
    return "\n".join(lines) + "\n"


# ── the figure ──────────────────────────────────────────────────────────────
def _mix(lo, hi, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(lo, hi))


def _hex(rgb):
    return "#%02x%02x%02x" % rgb


BLUE_LO, BLUE_HI = (234, 242, 251), (26, 79, 139)
WARM_LO, WARM_HI = (253, 240, 227), (150, 68, 20)
GREEN = "#2f855a"
MISSING = "#eeeeee"


def _ramp(value, lo, hi, colours):
    if hi <= lo:
        t = 0.5
    else:
        t = (value - lo) / float(hi - lo)
    t = max(0.0, min(1.0, t))
    return _hex(_mix(colours[0], colours[1], t))


def _dark(value, lo, hi):
    if hi <= lo:
        return False
    return (value - lo) / float(hi - lo) > 0.55


def _heatmap(x0, y0, cw, ch, us, ps, cells, field, colours, fmt, sub):
    """One (U, P) grid. Returns (svg parts, low, high) for the legend."""
    values = [c[field] for c in cells if c.get(field) is not None]
    lo = min(values) if values else 0.0
    hi = max(values) if values else 1.0
    index = {(c["u"], c["p"]): c for c in cells}
    parts = []
    for col, u in enumerate(us):
        for row, p in enumerate(ps):
            x = x0 + col * cw
            y = y0 + (len(ps) - 1 - row) * ch
            cell = index.get((u, p))
            value = cell.get(field) if cell else None
            if value is None:
                parts.append('<rect x="%d" y="%d" width="%d" height="%d" '
                             'fill="%s" stroke="white" stroke-width="2"/>'
                             % (x, y, cw, ch, MISSING))
                parts.append('<text x="%d" y="%d" class="cm">none</text>'
                             % (x + cw // 2, y + ch // 2 + 4))
                continue
            fill = _ramp(value, lo, hi, colours)
            edge = "white"
            width = 2
            if field == "ratio" and value < 1.0:
                edge = GREEN
                width = 3
            parts.append('<rect x="%d" y="%d" width="%d" height="%d" '
                         'fill="%s" stroke="%s" stroke-width="%d"/>'
                         % (x, y, cw, ch, fill, edge, width))
            klass = "cvw" if _dark(value, lo, hi) else "cv"
            parts.append('<text x="%d" y="%d" class="%s">%s</text>'
                         % (x + cw // 2, y + ch // 2, klass, fmt % value))
            note = sub(cell)
            if note:
                parts.append('<text x="%d" y="%d" class="%s">%s</text>'
                             % (x + cw // 2, y + ch // 2 + 15,
                                "csw" if klass == "cvw" else "cs", note))
    for col, u in enumerate(us):
        parts.append('<text x="%d" y="%d" class="ax">%d</text>'
                     % (x0 + col * cw + cw // 2, y0 + len(ps) * ch + 16, u))
    for row, p in enumerate(ps):
        parts.append('<text x="%d" y="%d" class="ay">%d</text>'
                     % (x0 - 8, y0 + (len(ps) - 1 - row) * ch + ch // 2 + 4, p))
    parts.append('<text x="%d" y="%d" class="axt">U, predicate units</text>'
                 % (x0 + len(us) * cw // 2, y0 + len(ps) * ch + 34))
    parts.append('<text x="%d" y="%d" class="ayt" transform="rotate(-90 %d %d)">'
                 'P</text>'
                 % (x0 - 30, y0 + len(ps) * ch // 2,
                    x0 - 30, y0 + len(ps) * ch // 2))
    return parts, lo, hi


def _legend(x0, y0, width, lo, hi, colours, fmt):
    parts = []
    steps = 40
    step = width / float(steps)
    for i in range(steps):
        t = i / float(steps - 1)
        parts.append('<rect x="%.1f" y="%d" width="%.1f" height="10" '
                     'fill="%s"/>'
                     % (x0 + i * step, y0, step + 0.6,
                        _hex(_mix(colours[0], colours[1], t))))
    parts.append('<text x="%d" y="%d" class="lg">%s</text>'
                 % (x0, y0 + 22, fmt % lo))
    parts.append('<text x="%d" y="%d" class="lgr">%s</text>'
                 % (x0 + width, y0 + 22, fmt % hi))
    return parts


def _scatter(x0, y0, width, height, cells, key, label):
    pts = [c for c in cells
           if c.get(key) is not None and c.get("ratio") is not None]
    parts = []
    if len(pts) < 2:
        parts.append('<text x="%d" y="%d" class="n">Too few cells carry both '
                     'axes to plot them against each other.</text>'
                     % (x0, y0 + 20))
        return parts

    xs = [c[key] for c in pts]
    ys = [c["ratio"] for c in pts]
    xmax = (max(xs) * 1.15) or 1.0
    ymax = max(max(ys) * 1.15, 1.3)

    def px(v):
        return x0 + width * (v / xmax)

    def py(v):
        return y0 + height - height * (v / ymax)

    # The corner the whole experiment is about: reconstructs well, plans badly.
    # The cut is the C17 plannability threshold when the data straddle it, and
    # the median otherwise. A threshold outside the data range would draw a
    # line that separates nothing, so in that case the whole width is shaded
    # and the note says why.
    if key == "best_val" and xmax * 0.15 < PLANNABLE_VAL_LOSS < xmax * 0.9:
        xcut, cut_note = PLANNABLE_VAL_LOSS, "val loss %.1f (SPEC C17)"
    elif key == "best_val":
        xcut, cut_note = None, None
    else:
        xcut, cut_note = st.median(xs), "median %s" % label

    shade_to = px(xcut) if xcut is not None else x0 + width
    if ymax > 1.0:
        parts.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                     'fill="#fff5f5"/>'
                     % (x0, y0, shade_to - x0, py(1.0) - y0))
        parts.append('<text x="%.1f" y="%.1f" class="q">reconstructs well, '
                     'plans worse than a straight line</text>'
                     % (x0 + 8, y0 + 16))
    if xcut is None and key == "best_val":
        parts.append('<text x="%.1f" y="%.1f" class="n">every cell is under '
                     'the SPEC C17 plannability threshold of %.1f</text>'
                     % (x0 + 8, y0 + 30, PLANNABLE_VAL_LOSS))

    parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="ref"/>'
                 % (x0, py(1.0), x0 + width, py(1.0)))
    parts.append('<text x="%d" y="%.1f" class="n">mse_ratio = 1, the straight '
                 'line</text>' % (x0 + 4, py(1.0) - 5))
    if xcut is not None:
        parts.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" class="ref"/>'
                     % (px(xcut), y0, px(xcut), y0 + height))
        parts.append('<text x="%.1f" y="%d" class="n">%s</text>'
                     % (px(xcut) + 4, y0 + height - 6,
                        cut_note % xcut if "%" in cut_note else cut_note))

    parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" class="axl"/>'
                 % (x0, y0 + height, x0 + width, y0 + height))
    parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" class="axl"/>'
                 % (x0, y0, x0, y0 + height))

    for i in range(5):
        v = xmax * i / 4.0
        parts.append('<text x="%.1f" y="%d" class="ax">%s</text>'
                     % (px(v), y0 + height + 15, "%.3g" % v))
    for i in range(5):
        v = ymax * i / 4.0
        parts.append('<text x="%d" y="%.1f" class="ay">%s</text>'
                     % (x0 - 6, py(v) + 4, "%.3g" % v))

    # Nudge a label off its neighbours. Cells that plan alike land on top of
    # one another, and an unreadable label is the same as no label. Candidate
    # offsets are tried in order and the first free one wins.
    taken = []

    def free(lx, ly):
        for ox, oy in taken:
            if abs(ox - lx) < 58 and abs(oy - ly) < 11:
                return False
        return True

    for c in sorted(pts, key=lambda d: (d[key], d["ratio"])):
        cx, cy = px(c[key]), py(c["ratio"])
        fill = GREEN if c["ratio"] < 1.0 else "#1a4f8b"
        parts.append('<circle cx="%.1f" cy="%.1f" r="5" fill="%s"/>'
                     % (cx, cy, fill))
        lx, ly = cx + 8, cy + 4
        for dx, dy in ((8, 4), (8, -9), (8, 17), (-66, 4), (8, -22), (8, 30)):
            if free(cx + dx, cy + dy):
                lx, ly = cx + dx, cy + dy
                break
        taken.append((lx, ly))
        parts.append('<text x="%.1f" y="%.1f" class="pt">U%d/P%d</text>'
                     % (lx, ly, c["u"], c["p"]))

    rho = spearman(xs, ys)
    parts.append('<text x="%d" y="%d" class="axt">%s, lower is better</text>'
                 % (x0 + width // 2, y0 + height + 34, label))
    parts.append('<text x="%d" y="%d" class="ayt" '
                 'transform="rotate(-90 %d %d)">median mse_ratio</text>'
                 % (x0 - 40, y0 + height // 2, x0 - 40, y0 + height // 2))
    parts.append('<text x="%d" y="%d" class="nr">%s</text>'
                 % (x0 + width - 6, y0 + 16,
                    "Spearman rank correlation: n/a" if rho is None
                    else "Spearman rank correlation %.2f" % rho))
    return parts


def render_svg(cells):
    if not cells:
        return None
    key, label, _ = recon_axis(cells)
    us = sorted(set(c["u"] for c in cells))
    ps = sorted(set(c["p"] for c in cells))
    if not us or not ps:
        return None

    width, height = 1000, 860
    cw = max(64, min(96, 380 // max(1, len(us))))
    ch = max(56, min(84, 210 // max(1, len(ps))))
    ax, bx, top = 92, 560, 78

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
        'viewBox="0 0 %d %d">' % (width, height, width, height),
        '<style>text{font-family:sans-serif}'
        '.h{font-size:17px;font-weight:bold}'
        '.t{font-size:14px;font-weight:bold}'
        '.n{font-size:11px;fill:#555}'
        '.nr{font-size:11px;fill:#555;text-anchor:end}'
        '.q{font-size:11px;fill:#9b2c2c}'
        '.cv{font-size:12px;font-weight:bold;text-anchor:middle;fill:#1a202c}'
        '.cvw{font-size:12px;font-weight:bold;text-anchor:middle;fill:#ffffff}'
        '.cs{font-size:9px;text-anchor:middle;fill:#4a5568}'
        '.csw{font-size:9px;text-anchor:middle;fill:#e2e8f0}'
        '.cm{font-size:10px;text-anchor:middle;fill:#a0aec0}'
        '.ax{font-size:11px;text-anchor:middle;fill:#4a5568}'
        '.ay{font-size:11px;text-anchor:end;fill:#4a5568}'
        '.axt{font-size:11px;text-anchor:middle;fill:#2d3748}'
        '.ayt{font-size:11px;text-anchor:middle;fill:#2d3748}'
        '.lg{font-size:10px;fill:#4a5568}'
        '.lgr{font-size:10px;text-anchor:end;fill:#4a5568}'
        '.pt{font-size:10px;fill:#2d3748}'
        '.ref{stroke:#cbd5e0;stroke-width:1;stroke-dasharray:4 3}'
        '.axl{stroke:#a0aec0;stroke-width:1}</style>',
        '<rect width="%d" height="%d" fill="white"/>' % (width, height),
        '<text x="%d" y="30" class="h">G4 &#8212; reconstruction and planning '
        'over the U by P grid</text>' % (ax - 46),
        '<text x="%d" y="50" class="n">VidVRD, 88 screened clips, window 16. '
        'Both panels use the same cells. A cell can be light on the left and '
        'dark on the right.</text>' % (ax - 46),
    ]

    def recon_sub(cell):
        if cell.get("epochs"):
            return "%d ep" % cell["epochs"]
        return ""

    def plan_sub(cell):
        if not cell.get("windows"):
            return ""
        return "%d/%d reached" % (cell["solved"], cell["windows"])

    parts.append('<text x="%d" y="%d" class="t">reconstruction &#8212; %s</text>'
                 % (ax, top - 12, label))
    left, rlo, rhi = _heatmap(ax, top, cw, ch, us, ps, cells, key,
                              (BLUE_LO, BLUE_HI), "%.3g", recon_sub)
    parts += left
    parts += _legend(ax, top + len(ps) * ch + 48, len(us) * cw,
                     rlo, rhi, (BLUE_LO, BLUE_HI), "%.3g")

    parts.append('<text x="%d" y="%d" class="t">planning &#8212; median '
                 'mse_ratio</text>' % (bx, top - 12))
    right, plo, phi = _heatmap(bx, top, cw, ch, us, ps, cells, "ratio",
                               (WARM_LO, WARM_HI), "%.2f", plan_sub)
    parts += right
    parts += _legend(bx, top + len(ps) * ch + 48, len(us) * cw,
                     plo, phi, (WARM_LO, WARM_HI), "%.2f")
    parts.append('<text x="%d" y="%d" class="n">A green border marks a cell '
                 'that beat the straight line.</text>'
                 % (bx, top + len(ps) * ch + 84))

    scatter_top = top + len(ps) * ch + 130
    parts.append('<text x="%d" y="%d" class="t">the two axes against each '
                 'other &#8212; do they agree?</text>' % (ax, scatter_top - 14))
    parts += _scatter(ax, scatter_top, width - ax - 60,
                      height - scatter_top - 76, cells, key, label)

    parts.append('<text x="%d" y="%d" class="n">Reconstruction loss does not '
                 'select for plannability (EVAL.md 5.7). The cells in the '
                 'shaded corner are where that shows.</text>'
                 % (ax - 46, height - 26))
    parts.append('<text x="%d" y="%d" class="n">One dataset only. Treat any '
                 'ordering here as a hypothesis until a second corpus '
                 'reproduces it.</text>' % (ax - 46, height - 10))
    parts.append('</svg>')
    return "\n".join(parts)


def main():
    os.chdir(ROOT)
    cells = collect()
    text = render_markdown(cells)
    if not os.path.isdir("eval/planner"):
        os.makedirs("eval/planner")
    out = "eval/planner/G4_summary.md"
    with open(out, "w") as handle:
        handle.write(text)
    print(text)
    print("wrote %s" % out)
    svg = render_svg(cells)
    if svg:
        svg_path = "eval/planner/G4_summary.svg"
        with open(svg_path, "w") as handle:
            handle.write(svg)
        print("wrote %s" % svg_path)
    else:
        print("no cells, so no figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
