#!/usr/bin/env python3
"""Fill in the floor columns a planner run predates, and reconcile the pair.

`floor_ratio` is planner error over the quantisation floor, and it is the
number the paired oracle-versus-trained claim is stated in. The oracle runs
under `eval/planner/p3-*` were scored on 2026-08-30, before `harness.py`
learned to write `floor_ratio` and `quantisation_floor`, so their
`summary.csv` files stop at `temporal_order`. The trained arm under
`eval/planner/p3t-*` was scored the next morning and does carry both.

Nothing has to be re-planned to close that gap. The floor is a property of
the annotated boxes and the bin resolution alone -- no planner, no model, no
GPU -- and `bbox_mse` is already recorded per window. So the two columns can
be recomputed from what is on disk:

    quantisation_floor = round_trip_error(gt boxes of the scored frames)
    floor_ratio        = bbox_mse / quantisation_floor

The one thing that can go wrong is scoring against an export that is no
longer the export the run used. That would produce a plausible number with
nothing behind it, which is worse than a missing column. So every row is
checked first: the linear-interpolation baseline is recomputed from the
export and compared with the `baseline_mse` the run recorded. A row whose
baseline does not reproduce gets no floor and is counted as unverified.

**Two floors, and they are not the same number.** `harness.py` takes the
floor over the frames a window is scored on -- the k-2 intermediates. The
paired figure of 2026-08-31 took it over the whole clip at once and reused
that single value for every window of the clip and for both arms. The second
is the coarser instrument and it moves the oracle median by about 9%, so both
are written out and named apart: `floor_ratio` / `quantisation_floor` for the
per-window definition that the rest of the codebase means by those words, and
`clip_floor_ratio` / `clip_floor` for the whole-clip one.

Usage::

    python3 tools/planner/p1_floor.py \\
        --runs 'eval/planner/p3-*' --exports eval/exports/w16_oracle \\
        --pair 'eval/planner/p3t-*' --pair-exports eval/exports/h14_clips \\
        --out eval/planner/p1-floor

Writes the two-or-four extra columns into each run's `summary.csv` in place,
then a reconciliation JSON, a per-clip CSV and an SVG under `--out`.
"""

import argparse
import csv
import glob
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from tools.planner.common.metrics import bbox_mse                # noqa: E402
from tools.planner.common.windows import linear_interp_bboxes    # noqa: E402
from tools.planner.oracle import round_trip_error                # noqa: E402

# Where the two new per-window columns go. `eval_plannability.sh` writes them
# between `temporal_order` and `decode_fallbacks`, and a summary that puts
# them elsewhere would not diff against a freshly generated one.
AFTER = "temporal_order"
WINDOW_COLS = ("floor_ratio", "quantisation_floor")
CLIP_COLS = ("clip_floor_ratio", "clip_floor")

# The baseline recomputation runs in float64 where the harness ran in float32,
# so agreement is to about 1e-7 relative rather than exact. Anything looser
# than this is a different export, not rounding.
BASELINE_TOL = 1e-5


def read_summary(path):
    """Return (header, rows) with rows as ordered dicts of strings."""
    with open(path) as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [r for r in reader if r]
    return header, [dict(zip(header, r)) for r in rows]


def row_key(row):
    return (row.get("export"), row.get("method"),
            row.get("init"), row.get("goal"))


def insert_columns(header, name_after, new_names):
    """Return a header with `new_names` inserted after `name_after`.

    Columns already present are left where they are rather than duplicated,
    so running the tool twice is a no-op on the header.
    """
    out = list(header)
    missing = [n for n in new_names if n not in out]
    if not missing:
        return out
    at = out.index(name_after) + 1 if name_after in out else len(out)
    return out[:at] + missing + out[at:]


def window_floor(gt_boxes, init, goal):
    """Floor over the frames a window is scored on: the k-2 intermediates.

    This is what `harness.py` records, so it is what `floor_ratio` means
    everywhere else in the codebase.
    """
    return round_trip_error(gt_boxes[init + 1:goal])


def clip_floor(gt_boxes):
    """Floor over every frame of the clip at once.

    One number per clip rather than one per window. Coarser, but it is the
    definition the 2026-08-31 paired figure used, and reproducing that figure
    is the only way to tell whether its headline still holds.
    """
    return round_trip_error(gt_boxes)


def recompute_baseline(gt_boxes, init, goal):
    """The linear-interpolation error, recomputed the way the harness did.

    Used only as a fingerprint. If this matches the `baseline_mse` already in
    the summary then the export on disk is the export the run was scored
    against, and a floor taken from it belongs to the same frames.
    """
    n_mid = goal - init - 1
    if n_mid <= 0:
        return None
    init_boxes = gt_boxes[init]
    goal_boxes = gt_boxes[goal]
    scoreable = ((np.abs(init_boxes).sum(axis=-1) > 0)
                 & (np.abs(goal_boxes).sum(axis=-1) > 0))
    baseline = linear_interp_bboxes(init_boxes, goal_boxes, n_mid)
    scored = bbox_mse(baseline, gt_boxes[init + 1:goal], "hungarian", scoreable)
    return scored["mean_mse"]


def agrees(recomputed, recorded, tol=BASELINE_TOL):
    if recomputed is None or recorded in (None, ""):
        return False
    recorded = float(recorded)
    return abs(recomputed - recorded) <= tol * max(1.0, abs(recorded))


def export_path_from_log(run_dir, row):
    """The export a window was actually scored against, per its own log.

    `plan_video.py` prints the export path as the second line of the log, so
    the run records its own provenance. Preferred over guessing from a
    directory, because guessing is how a run gets scored against the wrong
    boxes.
    """
    name = "%s_%s_%s_%s.log" % (row.get("export"), row.get("method"),
                                row.get("init"), row.get("goal"))
    path = os.path.join(run_dir, "logs", name)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        for line in handle:
            if line.startswith("export "):
                return line.split(None, 1)[1].strip()
    return None


def load_gt(path, cache):
    if path not in cache:
        with np.load(path) as data:
            cache[path] = data["gt_boxes"]
    return cache[path]


def score_run(run_dir, exports_dir, cache):
    """Compute both floors for every scored row of one run.

    Returns (header, rows, report). `report` counts what could and could not
    be verified, because a silent skip is how a hole gets into a headline.
    """
    summary = os.path.join(run_dir, "summary.csv")
    header, rows = read_summary(summary)

    seen = set()
    for row in rows:
        key = row_key(row)
        if key in seen:
            raise ValueError("%s has two rows for %s; a summary must hold one "
                             "row per window per method" % (summary, key))
        seen.add(key)

    report = {"run": os.path.basename(run_dir), "rows": len(rows),
              "scored": 0, "filled": 0, "confirmed": 0,
              "unverified": [], "no_export": [], "disagreed": []}

    for row in rows:
        if not row.get("bbox_mse"):
            continue
        report["scored"] += 1

        export = export_path_from_log(run_dir, row)
        if export is None and exports_dir:
            export = os.path.join(exports_dir, "%s.npz" % row.get("export"))
        if not export or not os.path.exists(export):
            report["no_export"].append(row_key(row))
            continue

        gt = load_gt(export, cache).astype(np.float64)
        init, goal = int(row["init"]), int(row["goal"])
        if goal >= len(gt):
            report["no_export"].append(row_key(row))
            continue

        if not agrees(recompute_baseline(gt, init, goal), row.get("baseline_mse")):
            report["unverified"].append(row_key(row))
            continue

        mse = float(row["bbox_mse"])
        w_floor = window_floor(gt, init, goal)
        c_floor = clip_floor(gt)
        ratio = mse / w_floor if w_floor else None

        # A run scored after `harness.py` learned to write these columns
        # already has them. Recomputing over the top would hide a difference
        # instead of surfacing one, so an existing value is compared and kept.
        # Agreement here is the evidence that the recomputation is the same
        # arithmetic the harness did, and it is what licenses filling the
        # column in for the runs that predate it.
        if row.get("floor_ratio"):
            if ratio is not None and agrees(ratio, row["floor_ratio"]):
                report["confirmed"] += 1
            else:
                report["disagreed"].append(row_key(row))
        else:
            row["quantisation_floor"] = repr(w_floor) if w_floor else ""
            row["floor_ratio"] = repr(ratio) if ratio is not None else ""
            report["filled"] += 1

        row["clip_floor"] = repr(c_floor) if c_floor else ""
        row["clip_floor_ratio"] = repr(mse / c_floor) if c_floor else ""

    header = insert_columns(header, AFTER, WINDOW_COLS)
    # The clip-level pair goes on the end rather than beside the others: it is
    # a reconciliation aid, not part of the format `eval_plannability.sh`
    # writes, and a reader diffing the two should see the difference at the
    # end of the line instead of in the middle of it.
    header = header + [c for c in CLIP_COLS if c not in header]
    return header, rows, report


def write_summary(path, header, rows):
    with open(path, "w") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(c, "") for c in header])


def clip_of(run_dir, prefix):
    base = os.path.basename(run_dir.rstrip("/"))
    return base[len(prefix):] if base.startswith(prefix) else base


def per_clip_medians(runs, column):
    """One value per clip: the median over that clip's scored windows.

    Per clip rather than per row. Pooling rows across clips would let a clip
    with four windows outvote a clip with two, and the claim is about clips.
    """
    out = {}
    for clip, rows in runs.items():
        vals = [float(r[column]) for r in rows if r.get(column)]
        if vals:
            out[clip] = statistics.median(vals)
    return out


def clip_floors(runs):
    """The one whole-clip floor each run recorded, keyed by clip."""
    out = {}
    for clip, rows in runs.items():
        for row in rows:
            if row.get("clip_floor"):
                out[clip] = float(row["clip_floor"])
                break
    return out


def medians_against(runs, floors):
    """Per-clip medians of `bbox_mse / floors[clip]`.

    Separate from `per_clip_medians` because a paired claim can be stated
    with **one** denominator shared by both arms. The two arms' exports do not
    carry an identical object set, so each arm's own floor is not the same
    number, and dividing by different denominators is not a paired test.
    """
    out = {}
    for clip, rows in runs.items():
        floor = floors.get(clip)
        if not floor:
            continue
        vals = [float(r["bbox_mse"]) / floor for r in rows if r.get("bbox_mse")]
        if vals:
            out[clip] = statistics.median(vals)
    return out


def paired(a, b):
    clips = sorted(set(a) & set(b))
    return clips, sum(1 for c in clips if a[c] < b[c])


def _fmt(value, spec="%.4f"):
    return "n/a" if value is None else spec % value


def figure(path, clips, oracle_w, trained_w, oracle_c, trained_c, title):
    """A paired dot plot on a log axis, one row per clip, both floors shown.

    The point of drawing both definitions on one axis is that the gap between
    the filled and the hollow marker is the size of the definition choice, and
    a reader can see at a glance whether it is large enough to matter to the
    verdict.
    """
    import math

    left, top, row_h = 250, 66, 20
    x0, decade = 300.0, 118.0
    height = top + row_h * len(clips) + 74
    width = 900

    def x_of(v):
        return x0 + decade * math.log10(max(v, 1.0))

    parts = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" '
             'font-family="sans-serif">' % (width, height)]
    parts.append('<rect width="%d" height="%d" fill="#ffffff"/>' % (width, height))
    parts.append('<text x="8" y="20" font-size="14" fill="#222" '
                 'font-weight="600">%s</text>' % title)
    parts.append('<text x="8" y="38" font-size="11" fill="#888">1.0 is the best '
                 'any representation could do at this bin resolution. Log scale. '
                 'Filled marker: floor taken over the frames the window was '
                 'scored on. Hollow marker: one floor per clip, shared by both '
                 'arms.</text>')

    for power in range(0, 4):
        gx = x_of(10 ** power)
        parts.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#f0f0f0"/>'
                     % (gx, top - 12, gx, top + row_h * len(clips)))
        parts.append('<text x="%.1f" y="%d" font-size="10" fill="#888" '
                     'text-anchor="middle">%d</text>'
                     % (gx, top + row_h * len(clips) + 16, 10 ** power))
    parts.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#333" '
                 'stroke-width="1.5"/>' % (x_of(1.0), top - 12,
                                           x_of(1.0), top + row_h * len(clips)))

    for i, clip in enumerate(clips):
        y = top + row_h * i
        parts.append('<text x="8" y="%d" font-size="10" fill="#333">%s</text>'
                     % (y + 4, clip))
        parts.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#ddd" '
                     'stroke-width="1.5"/>'
                     % (x_of(oracle_w[clip]), y, x_of(trained_w[clip]), y))
        for value, colour in ((oracle_w[clip], "#12805c"),
                              (trained_w[clip], "#b3261e")):
            parts.append('<circle cx="%.1f" cy="%d" r="4" fill="%s"/>'
                         % (x_of(value), y, colour))
        for value, colour in ((oracle_c[clip], "#12805c"),
                              (trained_c[clip], "#b3261e")):
            parts.append('<circle cx="%.1f" cy="%d" r="4" fill="none" '
                         'stroke="%s" stroke-width="1.2"/>'
                         % (x_of(value), y, colour))

    base = top + row_h * len(clips)
    parts.append('<text x="8" y="%d" font-size="11" fill="#12805c" '
                 'font-weight="600">green = oracle, median %s per scored '
                 'window / %s per clip</text>'
                 % (base + 38, _fmt(statistics.median(
                     [oracle_w[c] for c in clips]), "%.2f"),
                    _fmt(statistics.median(
                        [oracle_c[c] for c in clips]), "%.2f")))
    parts.append('<text x="8" y="%d" font-size="11" fill="#b3261e" '
                 'font-weight="600">red = trained, median %s per scored '
                 'window / %s per clip</text>'
                 % (base + 54, _fmt(statistics.median(
                     [trained_w[c] for c in clips]), "%.2f"),
                    _fmt(statistics.median(
                        [trained_c[c] for c in clips]), "%.2f")))
    _, closer_w = paired(oracle_w, trained_w)
    _, closer_c = paired(oracle_c, trained_c)
    parts.append('<text x="470" y="%d" font-size="11" fill="#333">oracle closer '
                 'on %d of %d either way</text>'
                 % (base + 46, min(closer_w, closer_c), len(clips)))
    parts.append('</svg>')

    with open(path, "w") as handle:
        handle.write("".join(parts))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="eval/planner/p3-*",
                    help="glob of run directories to fill in")
    ap.add_argument("--exports", default="eval/exports/w16_oracle",
                    help="fallback export directory when a log is missing")
    ap.add_argument("--prefix", default="p3-",
                    help="run-directory prefix to strip to get the clip name")
    ap.add_argument("--pair", default=None,
                    help="glob of the arm to pair against, e.g. "
                         "'eval/planner/p3t-*'")
    ap.add_argument("--pair-exports", default="eval/exports/h14_clips")
    ap.add_argument("--pair-prefix", default="p3t-")
    ap.add_argument("--out", default="eval/planner/p1-floor")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and report without rewriting any summary")
    args = ap.parse_args(argv)

    cache = {}

    def collect(pattern, exports, prefix):
        runs, reports = {}, []
        for run_dir in sorted(glob.glob(pattern)):
            if not os.path.exists(os.path.join(run_dir, "summary.csv")):
                continue
            header, rows, report = score_run(run_dir, exports, cache)
            if not args.dry_run:
                write_summary(os.path.join(run_dir, "summary.csv"), header, rows)
            reports.append(report)
            runs[clip_of(run_dir, prefix)] = rows
        return runs, reports

    runs, reports = collect(args.runs, args.exports, args.prefix)
    if not runs:
        sys.exit("no run with a summary.csv matched %s" % args.runs)

    result = {"runs": args.runs, "reports": reports}
    oracle_w = per_clip_medians(runs, "floor_ratio")
    oracle_c = per_clip_medians(runs, "clip_floor_ratio")
    result["clips"] = len(oracle_w)
    result["windows"] = sum(r["filled"] for r in reports)
    result["median_floor_ratio"] = (statistics.median(list(oracle_w.values()))
                                    if oracle_w else None)
    result["median_clip_floor_ratio"] = (statistics.median(list(oracle_c.values()))
                                         if oracle_c else None)

    if args.pair:
        pair_runs, pair_reports = collect(args.pair, args.pair_exports,
                                          args.pair_prefix)
        result["pair_reports"] = pair_reports
        trained_w = per_clip_medians(pair_runs, "floor_ratio")
        trained_own = per_clip_medians(pair_runs, "clip_floor_ratio")
        # Both arms divided by the reference arm's whole-clip floor. This is
        # the recipe the 2026-08-31 paired figure used, and it is the only one
        # under which that figure's numbers can be checked.
        shared = clip_floors(runs)
        oracle_c = medians_against(runs, shared)
        trained_c = medians_against(pair_runs, shared)
        clips, closer_w = paired(oracle_w, trained_w)
        _, closer_own = paired(oracle_c, trained_own)
        _, closer_c = paired(oracle_c, trained_c)
        result["paired"] = {
            "clips": len(clips),
            "per_window_floor": {
                "oracle_median": statistics.median([oracle_w[c] for c in clips]),
                "trained_median": statistics.median([trained_w[c] for c in clips]),
                "oracle_closer": closer_w,
            },
            "own_clip_floor": {
                "oracle_median": statistics.median([oracle_c[c] for c in clips]),
                "trained_median": statistics.median([trained_own[c] for c in clips]),
                "oracle_closer": closer_own,
            },
            "shared_clip_floor": {
                "oracle_median": statistics.median([oracle_c[c] for c in clips]),
                "trained_median": statistics.median([trained_c[c] for c in clips]),
                "oracle_closer": closer_c,
            },
            "per_clip": [{"clip": c,
                          "oracle_floor_ratio": oracle_w[c],
                          "trained_floor_ratio": trained_w[c],
                          "oracle_clip_floor_ratio": oracle_c[c],
                          "trained_clip_floor_ratio": trained_c[c]}
                         for c in clips],
        }
        for arm in ("per_window_floor", "own_clip_floor", "shared_clip_floor"):
            block = result["paired"][arm]
            block["separation"] = block["trained_median"] / block["oracle_median"]

    if not args.dry_run:
        if not os.path.isdir(args.out):
            os.makedirs(args.out)
        with open(os.path.join(args.out, "reconciliation.json"), "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        if args.pair:
            rows = result["paired"]["per_clip"]
            with open(os.path.join(args.out, "per_clip.csv"), "w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]),
                                        lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            figure(os.path.join(args.out, "floor_ratio_paired.svg"),
                   clips, oracle_w, trained_w, oracle_c, trained_c,
                   "How far above the quantisation floor? (%d clips, "
                   "two floor definitions)" % len(clips))

    print("runs %d, clips %d, windows filled %d"
          % (len(runs), result["clips"], result["windows"]))
    all_reports = reports + result.get("pair_reports", [])
    print("floor_ratio already present and confirmed on %d windows, "
          "disagreeing on %d"
          % (sum(r["confirmed"] for r in all_reports),
             sum(len(r["disagreed"]) for r in all_reports)))
    for report in all_reports:
        if report["unverified"] or report["no_export"] or report["disagreed"]:
            print("  %s: %d unverified, %d without an export, %d disagreeing"
                  % (report["run"], len(report["unverified"]),
                     len(report["no_export"]), len(report["disagreed"])))
    print("oracle median floor_ratio      %s (floor per scored window)"
          % _fmt(result["median_floor_ratio"]))
    print("oracle median clip_floor_ratio %s (floor per clip)"
          % _fmt(result["median_clip_floor_ratio"]))
    if args.pair:
        for arm in ("per_window_floor", "own_clip_floor", "shared_clip_floor"):
            block = result["paired"][arm]
            print("%-16s n=%d  oracle %s  trained %s  separation %s  "
                  "oracle closer on %d of %d"
                  % (arm, result["paired"]["clips"],
                     _fmt(block["oracle_median"], "%.2f"),
                     _fmt(block["trained_median"], "%.2f"),
                     _fmt(block["separation"], "%.1fx"),
                     block["oracle_closer"], result["paired"]["clips"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
