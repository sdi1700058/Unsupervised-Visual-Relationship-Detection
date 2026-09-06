#!/usr/bin/env python3
"""M4 — does latent distance track the number of actions between states?

**The quantity.** For two frames of one clip, the *frame gap* is how far apart
they sit in the video and the *latent distance* is the fewest observed
transitions that lead from the first latent to the second. A code that a
planner can use has the second grow with the first. A code that does not lets
the planner reach its goal before the window is filled, because in the latent
the goal was never far away.

`SPEC.md` V35 recorded exactly this on one clip of the H14 ablation: frames 7
apart sat 4 to 6 transitions apart in the trained latent against 7 in the
oracle. It was quoted as four medians, with no spread, on one clip, so it
could not be compared across datasets or across configurations. This turns it
into a metric.

`RELATED_WORK.md` P1 (Minimum Action Distance) learns an embedding where
distance *is* the action count, and P2 argues that a distance without the
triangle inequality *"translates to an inability to generalize and find
shortest paths"*. The number below is the quantity those two papers optimise,
measured on this project's own exports.

What is reported, and why each part is needed
---------------------------------------------

===================  =====================================================
`spearman`           rank correlation of frame gap against latent
                     distance. Below 0.30 the code carries no temporal
                     information and every other number here describes
                     noise, so it is checked first.
`pearson`            the same on the raw values, which is sensitive to the
                     curve flattening where the rank version is not.
`ratio_median`       transitions per frame of real time, and the direct
                     descendant of V35's four medians.
`ratio_q25/q75`      the **spread**. A median of 0.5 can mean every pair
                     compressed by half, or half the pairs exact and half
                     dead. Those are different codes and V35 could not
                     tell them apart.
`saturation_gap`     the frame gap beyond which the median distance stops
                     growing. Past it a planner cannot separate a long
                     horizon from a short one.
`hamming_spearman`   the same correlation against Hamming distance, which
                     needs no transition graph and so still reports when
                     the graph collapses.
===================  =====================================================

**Pairs never cross a clip boundary.** A multi-clip export concatenates clips,
and a pair spanning the join compares two frames that no amount of time
separates. The transition graph is built per clip for the same reason: a
transition seen in another clip is not one available in this one.

**This describes a code. It does not predict planner error.** `SPEC.md` V36
records what happened the last time a temporal measure was shipped as a
predictor: across 8 clips of one model it correlated +0.238 with `mse_ratio`,
the wrong sign, where the older geometry screen managed -0.452. Do not rank
two models by this.

numpy and the standard library only, so it runs under the cluster's Python
3.6. No model, no GPU and no video frames are needed.

    python3 tools/planner/m4_temporal.py eval/exports/H14-P10-150010.npz
    python3 tools/planner/m4_temporal.py vidvrd=eval/probe/batch \\
        vidor=eval/probe/vidor --max-gap 20 --out-dir eval/M4

A positional argument is a single export, a directory of exports pooled into
one group, or `label=path` to name the group in the table and the figure.
"""

import argparse
import glob
import json
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from tools.planner.onmanifold import observed_graph  # noqa: E402

# Enough gaps to see a curve flatten without leaving the shortest screened
# clips out of the table entirely.
DEFAULT_MAX_GAP = 20

# Source frames sampled per clip. Every source costs one breadth-first sweep
# of the clip's graph, and a long VidOR clip has thousands of nodes, so this
# is what keeps the whole run inside a couple of minutes.
DEFAULT_MAX_SOURCES = 120

# The median distance curve is integer-valued, so a single step of it can be
# flat while the curve still climbs. The slope is therefore measured over a
# window of gaps, and the curve counts as still climbing while it gains at
# least one transition per ten frames.
DEFAULT_SLOPE_WINDOW = 4
DEFAULT_SLOPE_FLOOR = 0.1

# Below this the frame gap and the latent distance are unrelated, and the
# ratio measures noise rather than compression.
SILENT_RHO = 0.30

# Wider than this and the ratio is not a property of the code on this dataset.
WIDE_IQR = 0.40


def split_label(arg):
    """`label=path` names a group; a bare path is named after itself."""
    if "=" in arg:
        label, path = arg.split("=", 1)
        return label, path
    base = os.path.basename(arg.rstrip(os.sep))
    if base.endswith(".npz"):
        base = base[:-4]
    return base, arg


def clip_segments(frame_ids, n_frames):
    """Contiguous `(name, start, stop)` runs, one per clip.

    An export written by `export_latents.py` carries ids shaped
    `clip/000123`, and clips are concatenated in order. Grouping by the part
    before the last slash, and only over contiguous runs, keeps a clip that
    appears twice from being merged into one impossible trajectory.
    """
    if frame_ids is None or len(frame_ids) != n_frames:
        return [("", 0, n_frames)] if n_frames else []

    names = []
    for fid in frame_ids:
        text = str(fid)
        names.append(text.rsplit("/", 1)[0] if "/" in text else text)

    segments, start = [], 0
    for i in range(1, n_frames):
        if names[i] != names[i - 1]:
            segments.append((names[start], start, i))
            start = i
    segments.append((names[start], start, n_frames))
    return segments


def frame_numbers(frame_ids, start, stop):
    """The real frame number of each row, falling back to its position.

    Sub-sampled frames sit further apart in time than in the array. Using the
    array index there would report a code as more faithful than it is, by
    exactly the sub-sampling factor.
    """
    if frame_ids is None:
        return np.arange(stop - start, dtype=np.int64)
    numbers = []
    for fid in frame_ids[start:stop]:
        tail = str(fid).rsplit("/", 1)[-1]
        try:
            numbers.append(int(tail))
        except ValueError:
            return np.arange(stop - start, dtype=np.int64)
    numbers = np.asarray(numbers, dtype=np.int64)
    # Non-increasing ids cannot be a time axis. The position is at least
    # ordered, so it is the safer reading of the same file.
    if np.any(np.diff(numbers) <= 0):
        return np.arange(stop - start, dtype=np.int64)
    return numbers


def _bfs_depths(adj, source):
    """Node -> fewest observed transitions from `source`."""
    depth = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        d = depth[node] + 1
        for nxt in adj[node]:
            if nxt not in depth:
                depth[nxt] = d
                queue.append(nxt)
    return depth


def collect_pairs(latents, frame_ids=None, max_gap=DEFAULT_MAX_GAP,
                  max_sources=DEFAULT_MAX_SOURCES):
    """Gather `(frame gap, latent distance, Hamming distance)` per pair.

    Forward pairs only, and never across a clip boundary. Within one clip a
    forward pair is always reachable, because the frames themselves are a
    path, so `unreachable` should stay at zero; it is counted rather than
    assumed, since a non-zero count would mean the graph was not built from
    the frames it is being measured against.
    """
    z = np.asarray(latents, dtype=np.int8)
    n = len(z)
    out = {"gaps": [], "dists": [], "hams": [], "unreachable": 0,
           "n_clips": 0, "n_frames": n,
           "n_distinct": len({row.tobytes() for row in z}) if n else 0}

    for _, start, stop in clip_segments(frame_ids, n):
        m = stop - start
        if m < 2:
            continue
        out["n_clips"] += 1
        seg = z[start:stop]
        frames = frame_numbers(frame_ids, start, stop)
        index, adj, _ = observed_graph(seg)
        node_of = [index[row.tobytes()] for row in seg]

        stride = max(1, (m - 1) // max_sources)
        for i in range(0, m - 1, stride):
            depth = _bfs_depths(adj, node_of[i])
            for j in range(i + 1, m):
                gap = int(frames[j] - frames[i])
                if gap <= 0:
                    continue
                if gap > max_gap:
                    break
                d = depth.get(node_of[j])
                if d is None:
                    out["unreachable"] += 1
                    continue
                out["gaps"].append(gap)
                out["dists"].append(int(d))
                out["hams"].append(int((seg[i] ^ seg[j]).sum()))
    return out


def merge_pairs(collected):
    """Pool several exports into one group, keeping the clip count honest."""
    total = {"gaps": [], "dists": [], "hams": [], "unreachable": 0,
             "n_clips": 0, "n_frames": 0, "n_distinct": 0}
    for c in collected:
        total["gaps"].extend(c["gaps"])
        total["dists"].extend(c["dists"])
        total["hams"].extend(c["hams"])
        total["unreachable"] += c["unreachable"]
        total["n_clips"] += c["n_clips"]
        total["n_frames"] += c["n_frames"]
        # Distinct latents do not add across files: two exports of the same
        # dataset share codes. The largest single export is the honest floor.
        total["n_distinct"] = max(total["n_distinct"], c["n_distinct"])
    return total


def _average_rank(values):
    """Ranks with ties averaged.

    Frame gaps are tied by construction — every pair at gap 7 shares a value —
    so ordinal ranks would order the ties arbitrarily and change the
    correlation with the order the pairs happened to be collected in.
    """
    a = np.asarray(values, dtype=np.float64)
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    return float((xc * yc).sum() / denom) if denom > 0 else None


def _spearman(x, y):
    if len(x) < 2:
        return None
    return _pearson(_average_rank(x), _average_rank(y))


def _dead(max_gap, collected, note):
    result = {"spearman": None, "pearson": None, "hamming_spearman": None,
              "ratio_median": None, "ratio_q25": None, "ratio_q75": None,
              "ratio_iqr": None, "ratio_mean": None, "ratio_sd": None,
              "exact_fraction": None, "per_gap": {}, "monotone_to": None,
              "saturation_gap": None, "plateau": None, "head_slope": None,
              "tail_slope": None, "slope_decay": None, "max_gap": max_gap,
              "note": note}
    for key in ("n_clips", "n_frames", "n_distinct", "unreachable"):
        result[key] = collected[key]
    result["n_pairs"] = len(collected["gaps"])
    return result


def summarise(collected, max_gap=DEFAULT_MAX_GAP,
              slope_window=DEFAULT_SLOPE_WINDOW,
              slope_floor=DEFAULT_SLOPE_FLOOR):
    """Turn collected pairs into the reported numbers."""
    if collected["n_distinct"] < 2:
        return _dead(max_gap, collected,
                     "dead latent: %d distinct state(s), so no distance "
                     "exists to measure" % collected["n_distinct"])
    if not collected["gaps"]:
        return _dead(max_gap, collected,
                     "no clip carried two frames inside the gap limit")

    gaps = np.asarray(collected["gaps"], dtype=np.float64)
    dists = np.asarray(collected["dists"], dtype=np.float64)
    hams = np.asarray(collected["hams"], dtype=np.float64)
    ratio = dists / gaps

    per_gap = {}
    for k in range(1, max_gap + 1):
        sel = dists[gaps == k]
        if not len(sel):
            continue
        per_gap[k] = {"median": float(np.median(sel)),
                      "q25": float(np.percentile(sel, 25)),
                      "q75": float(np.percentile(sel, 75)),
                      "mean": float(sel.mean()),
                      "n": int(len(sel))}

    ks = sorted(per_gap)
    med = dict((k, per_gap[k]["median"]) for k in ks)
    monotone_to = ks[0] if ks else None
    saturation = None
    for k in ks:
        if k + slope_window not in med:
            break
        slope = (med[k + slope_window] - med[k]) / float(slope_window)
        if slope < slope_floor:
            saturation = k
            break
        monotone_to = k + slope_window

    def _edge_slope(subset):
        if len(subset) < 2:
            return None
        span = float(subset[-1] - subset[0])
        return (med[subset[-1]] - med[subset[0]]) / span if span else None

    head = _edge_slope(ks[:slope_window + 1])
    tail = _edge_slope(ks[-(slope_window + 1):])
    decay = (tail / head) if (head not in (None, 0.0)
                              and tail is not None) else None

    q25, q75 = float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))
    return {
        "spearman": _spearman(gaps, dists),
        "pearson": _pearson(gaps, dists),
        "hamming_spearman": _spearman(gaps, hams),
        "ratio_median": float(np.median(ratio)),
        "ratio_q25": q25,
        "ratio_q75": q75,
        "ratio_iqr": q75 - q25,
        "ratio_mean": float(ratio.mean()),
        "ratio_sd": float(ratio.std()),
        "exact_fraction": float((dists == gaps).mean()),
        "per_gap": per_gap,
        "monotone_to": monotone_to,
        "saturation_gap": saturation,
        "plateau": med[ks[-1]] if ks else None,
        "head_slope": head,
        "tail_slope": tail,
        "slope_decay": decay,
        "max_gap": max_gap,
        "n_pairs": int(len(gaps)),
        "n_clips": collected["n_clips"],
        "n_frames": collected["n_frames"],
        "n_distinct": collected["n_distinct"],
        "unreachable": collected["unreachable"],
    }


def temporal_distance(latents, frame_ids=None, max_gap=DEFAULT_MAX_GAP,
                      max_sources=DEFAULT_MAX_SOURCES,
                      slope_window=DEFAULT_SLOPE_WINDOW,
                      slope_floor=DEFAULT_SLOPE_FLOOR):
    """Score one array of latents. See the module docstring for the fields."""
    return summarise(collect_pairs(latents, frame_ids, max_gap, max_sources),
                     max_gap, slope_window, slope_floor)


def read_export(path):
    """`(latents, frame_ids)`; frame_ids is None when the file omits them."""
    # allow_pickle stays off. Frame ids are stored as a unicode array, not an
    # object array, so nothing here needs to execute anything from the file.
    data = np.load(path)
    if "latents" not in data.files:
        raise SystemExit("%s: no latents" % path)
    ids = data["frame_ids"] if "frame_ids" in data.files else None
    return np.asarray(data["latents"], dtype=np.int8), ids


def score_group(paths, max_gap=DEFAULT_MAX_GAP,
                max_sources=DEFAULT_MAX_SOURCES,
                slope_window=DEFAULT_SLOPE_WINDOW,
                slope_floor=DEFAULT_SLOPE_FLOOR):
    """Pool every export in a group and score the pool as one dataset."""
    collected = []
    for p in paths:
        z, ids = read_export(p)
        collected.append(collect_pairs(z, ids, max_gap, max_sources))
    if not collected:
        raise SystemExit("no exports in group")
    return summarise(merge_pairs(collected), max_gap, slope_window,
                     slope_floor)


def verdict(result):
    """Decided in advance, in `experiments/M4_temporal_distance/README.md`.

    The order is load-bearing. A compression ratio taken over a code with no
    temporal signal measures noise, and reads exactly like a good result when
    the noise happens to sit near 1.0.
    """
    if result.get("n_distinct", 0) < 2:
        return ("SILENT: the latent has %d distinct state(s), so no distance "
                "exists to measure." % result.get("n_distinct", 0))
    rho = result.get("spearman")
    if rho is None:
        return ("SILENT: the correlation is undefined on this data, so the "
                "ratio below is not a measurement of compression.")
    if rho < SILENT_RHO:
        return ("SILENT: frame gap and latent distance correlate at only "
                "%+.3f, below the %.2f bar. The ratio measures noise, not "
                "compression, and must not be quoted as a ratio."
                % (rho, SILENT_RHO))

    tail = ""
    sat = result.get("saturation_gap")
    if sat is None:
        tail = (" The curve stays monotonic across the whole tested range, so "
                "a planner can separate a long horizon from a short one "
                "anywhere inside it.")
    else:
        tail = (" It saturates at a gap of %d frames: beyond that the latent "
                "cannot express a longer horizon." % sat)

    iqr = result.get("ratio_iqr")
    if iqr is not None and iqr >= WIDE_IQR:
        return ("SPREAD TOO WIDE to quote a median: the ratio runs %.2f to "
                "%.2f across the quartiles, an interquartile range of %.2f. "
                "On this dataset the ratio is not a property of the code. "
                "Report the quartiles, never the median alone.%s"
                % (result.get("ratio_q25", float("nan")),
                   result.get("ratio_q75", float("nan")), iqr, tail))

    ratio = result["ratio_median"]
    if ratio >= 0.95:
        head = ("TIME PRESERVED: %.2f latent transitions per frame of real "
                "time." % ratio)
    elif ratio >= 0.50:
        head = ("TIME COMPRESSED: %.2f latent transitions per frame of real "
                "time, where a faithful code gives 1.00." % ratio)
    else:
        head = ("TIME SEVERELY COMPRESSED: %.2f latent transitions per frame "
                "of real time. More than half of every horizon is lost."
                % ratio)
    return head + tail


def _esc(text):
    """SVG is XML. A raw angle bracket has shipped an unopenable figure from
    this project three times."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


# Ten colours, because a run of oracle datasets plus every trained arm reaches
# eight rows and a wrapped palette put two different curves in the same red.
PALETTE = ("#1f6feb", "#d1495b", "#2a9d8f", "#e9a03b", "#7b4bb7",
           "#5c6672", "#0b7285", "#b5179e", "#3f7d20", "#8d5524")


def write_svg(rows, path, width=880, height=470):
    """Median latent distance against frame gap, with the quartile band.

    The diagonal is a code that preserves time exactly. A curve under it
    compresses; a curve that flattens has saturated, and the frame gap where
    it flattens is the horizon a planner cannot see past.
    """
    left, top, right, bottom = 74, 46, 596, 404
    max_gap = max([r.get("max_gap") or 1 for _, r in rows] + [1])
    top_y = float(max_gap)

    def sx(k):
        return left + (right - left) * (float(k) / max_gap)

    def sy(v):
        return bottom - (bottom - top) * (min(float(v), top_y) / top_y)

    out = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" '
           'font-family="sans-serif">' % (width, height)]
    out.append('<rect width="%d" height="%d" fill="#ffffff"/>'
               % (width, height))
    out.append('<text x="14" y="22" font-size="14" fill="#222">M4 — latent '
               'distance against the frame gap between two states</text>')
    out.append('<text x="14" y="38" font-size="11" fill="#666">the dashed '
               'diagonal is one transition per frame. Below it the code '
               'compresses time; flat means it has saturated.</text>')

    # Axes and a grid every quarter of the range.
    out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333"/>'
               % (left, top, left, bottom))
    out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333"/>'
               % (left, bottom, right, bottom))
    for frac in (0.25, 0.5, 0.75, 1.0):
        v = max_gap * frac
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" '
                   'stroke="#e6e6e6"/>' % (left, sy(v), right, sy(v)))
        out.append('<text x="%d" y="%.1f" font-size="10" fill="#666" '
                   'text-anchor="end">%.0f</text>' % (left - 6, sy(v) + 3, v))
        out.append('<text x="%.1f" y="%d" font-size="10" fill="#666" '
                   'text-anchor="middle">%.0f</text>'
                   % (sx(v), bottom + 15, v))
    out.append('<text x="%.1f" y="%d" font-size="11" fill="#333" '
               'text-anchor="middle">frame gap</text>'
               % (0.5 * (left + right), bottom + 32))
    out.append('<text x="18" y="%.1f" font-size="11" fill="#333" '
               'transform="rotate(-90 18 %.1f)" text-anchor="middle">latent '
               'transitions</text>'
               % (0.5 * (top + bottom), 0.5 * (top + bottom)))
    out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#999" '
               'stroke-dasharray="4 3"/>' % (sx(0), sy(0), sx(max_gap),
                                             sy(max_gap)))

    legend_y = top
    for i, (name, r) in enumerate(rows):
        colour = PALETTE[i % len(PALETTE)]
        ks = sorted(r.get("per_gap") or {})
        if ks:
            band = ["%.1f,%.1f" % (sx(k), sy(r["per_gap"][k]["q75"]))
                    for k in ks]
            band += ["%.1f,%.1f" % (sx(k), sy(r["per_gap"][k]["q25"]))
                     for k in reversed(ks)]
            out.append('<polygon points="%s" fill="%s" opacity="0.13"/>'
                       % (" ".join(band), colour))
            line = ["%.1f,%.1f" % (sx(k), sy(r["per_gap"][k]["median"]))
                    for k in ks]
            out.append('<polyline points="%s" fill="none" stroke="%s" '
                       'stroke-width="2"/>' % (" ".join(line), colour))
            sat = r.get("saturation_gap")
            if sat is not None and sat in r["per_gap"]:
                out.append('<circle cx="%.1f" cy="%.1f" r="4" fill="none" '
                           'stroke="%s" stroke-width="2"/>'
                           % (sx(sat), sy(r["per_gap"][sat]["median"]),
                              colour))

        out.append('<rect x="%d" y="%.1f" width="10" height="10" fill="%s"/>'
                   % (right + 26, legend_y, colour))
        label = name if len(name) <= 26 else name[:25] + "…"
        out.append('<text x="%d" y="%.1f" font-size="11" fill="#222">%s</text>'
                   % (right + 42, legend_y + 9, _esc(label)))
        ratio = r.get("ratio_median")
        if ratio is None:
            detail = "no distance to measure"
        else:
            detail = ("%.2f steps per frame [%.2f-%.2f]"
                      % (ratio, r.get("ratio_q25") or 0.0,
                         r.get("ratio_q75") or 0.0))
        out.append('<text x="%d" y="%.1f" font-size="9.5" fill="#666">%s'
                   '</text>' % (right + 42, legend_y + 22, _esc(detail)))
        sat = r.get("saturation_gap")
        note = ("monotonic to gap %s" % r.get("monotone_to")
                if sat is None else "saturates at gap %d" % sat)
        if ratio is not None:
            out.append('<text x="%d" y="%.1f" font-size="9.5" fill="#666">%s'
                       '</text>' % (right + 42, legend_y + 34, _esc(note)))
        legend_y += 52

    out.append('<text x="14" y="%d" font-size="10" fill="#888">A description '
               'of a code, not a prediction of planner error (SPEC V36). The '
               'shaded band is the interquartile range.</text>'
               % (height - 14))
    out.append('</svg>')
    with open(path, "w") as fh:
        fh.write("".join(out))


def _group_paths(path):
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, "*.npz")))
        if not found:
            raise SystemExit("%s: no .npz inside" % path)
        return found
    return [path]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+",
                    help="an export, a directory of exports pooled into one "
                         "group, or label=path")
    ap.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP,
                    help="largest frame gap scored")
    ap.add_argument("--max-sources", type=int, default=DEFAULT_MAX_SOURCES,
                    help="source frames sampled per clip")
    ap.add_argument("--slope-window", type=int, default=DEFAULT_SLOPE_WINDOW,
                    help="gaps the saturation slope is measured over")
    ap.add_argument("--slope-floor", type=float, default=DEFAULT_SLOPE_FLOOR,
                    help="transitions per frame below which the curve counts "
                         "as flat")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)

    rows = []
    for arg in a.paths:
        label, path = split_label(arg)
        try:
            files = _group_paths(path)
            r = score_group(files, a.max_gap, a.max_sources, a.slope_window,
                            a.slope_floor)
        except SystemExit as exc:
            print("skip %s: %s" % (label, exc))
            continue
        r["files"] = len(files)
        rows.append((label, r))
        print("scored %-22s %4d clip(s), %6d pairs"
              % (label, r["n_clips"], r["n_pairs"]))

    if not rows:
        raise SystemExit("nothing scored")

    print("")
    print("%-24s %6s %7s %8s %8s %9s %14s %6s %10s"
          % ("group", "clips", "pairs", "spearman", "pearson", "steps/fr",
             "IQR", "exact", "saturates"))
    for label, r in rows:
        if r["ratio_median"] is None:
            print("%-24s %6d %7d %8s %8s %9s %14s %6s %10s"
                  % (label[:24], r["n_clips"], r["n_pairs"], "n/a", "n/a",
                     "n/a", "n/a", "n/a", "n/a"))
            continue
        sat = "no" if r["saturation_gap"] is None else str(r["saturation_gap"])
        print("%-24s %6d %7d %+8.3f %+8.3f %9.3f  [%.2f-%.2f]  %5.1f%% %10s"
              % (label[:24], r["n_clips"], r["n_pairs"], r["spearman"],
                 r["pearson"], r["ratio_median"], r["ratio_q25"],
                 r["ratio_q75"], 100.0 * r["exact_fraction"], sat))

    print("")
    for label, r in rows:
        print("%s: %s" % (label, verdict(r)))

    print("")
    print("A description of a code, NOT a prediction of planner error "
          "(SPEC V36).")

    if a.out_dir:
        if not os.path.isdir(a.out_dir):
            os.makedirs(a.out_dir)
        with open(os.path.join(a.out_dir, "m4_temporal.json"), "w") as fh:
            json.dump({"rows": dict((n, r) for n, r in rows),
                       "verdicts": dict((n, verdict(r)) for n, r in rows)},
                      fh, indent=2, sort_keys=True)
        write_svg(rows, os.path.join(a.out_dir, "m4_temporal.svg"))
        print("wrote %s/m4_temporal.json and m4_temporal.svg" % a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
