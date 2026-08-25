#!/usr/bin/env python3
"""Plot the interpolation results.

Reads the summary.csv that eval_plannability.sh writes and produces three
figures under eval/planner/<model>/viz/. Each figure ships with a short
caption file that says how to read it.

    python3 tools/planner/viz_plannability.py eval/planner/<model>
"""

import argparse
import csv
import sys

import numpy as np
from pathlib import Path


def read_summary(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_true(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def write_caption(png_path, title, what, how):
    caption = png_path.with_suffix(".caption.md")
    caption.write_text(
        f"# {title}\n\n**What it shows.** {what}\n\n**How to read it.** {how}\n")


def plot_reachability(rows, out_dir):
    """How often each method found any plan at all."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({r["method"] for r in rows})
    rates = []
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        hits = sum(1 for r in subset if is_true(r["reachability"]))
        rates.append(hits / len(subset) if subset else 0.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(methods, rates)
    ax.set_ylim(0, 1)
    ax.set_ylabel("windows with a plan")
    ax.set_title("Reachability by method")
    for i, rate in enumerate(rates):
        ax.text(i, rate + 0.02, f"{rate:.0%}", ha="center")

    path = out_dir / "reachability_by_method.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    write_caption(
        path, "Reachability by method",
        "The share of windows where the planner returned any plan.",
        "A low bar means the action schema does not connect the two frames. "
        "That is a failure of the learned relations, not of the search.")
    return path


def plot_ratio(rows, out_dir):
    """The headline figure: did the planner beat the straight line?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({r["method"] for r in rows})
    series = {}
    for method in methods:
        values = [as_float(r["mse_ratio"]) for r in rows
                  if r["method"] == method and is_true(r["reachability"])]
        values = [v for v in values if v is not None]
        if values:
            series[method] = values

    fig, ax = plt.subplots(figsize=(7, 4))
    if series:
        # Outlines, not filled bars. Two methods that agree would otherwise
        # sit exactly on top of each other and one would vanish.
        lo = min(min(v) for v in series.values())
        hi = max(max(v) for v in series.values())
        edges = np.linspace(min(lo, 0.9), max(hi, 1.1), 26)

        for i, (method, values) in enumerate(series.items()):
            ax.hist(values, bins=edges, histtype="step", linewidth=2.0,
                    linestyle=["-", "--", ":"][i % 3],
                    label=f"{method} (n={len(values)})")

        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2)
        ax.text(1.0, ax.get_ylim()[1] * 0.95, " straight-line baseline",
                va="top", fontsize=9)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no successful plans to score",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("planner error / straight-line error")
    ax.set_ylabel("windows")
    ax.set_title("Did the planner beat straight-line interpolation?")

    path = out_dir / "mse_ratio.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    write_caption(
        path, "Planner error against the straight-line baseline",
        "For each window, the planner's bbox error divided by the error of "
        "drawing a straight line between the two given frames.",
        "Mass left of the dashed line means the planner placed the objects "
        "better than the trivial guess. Mass to the right means the endpoints "
        "already gave the answer away and the model added nothing.")
    return path


def plot_error_by_window(rows, out_dir):
    """Absolute error, so the ratio has a scale behind it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({r["method"] for r in rows})

    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for method in methods:
        subset = [r for r in rows
                  if r["method"] == method and is_true(r["reachability"])]
        points = [(as_float(r["init"]), as_float(r["bbox_mse"])) for r in subset]
        points = [(x, y) for x, y in points if x is not None and y is not None]
        if not points:
            continue
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points],
                marker="o", markersize=3, linewidth=1, label=method)
        plotted = True

    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no successful plans to score",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("window start frame")
    ax.set_ylabel("bbox error, squared pixels")
    ax.set_title("Where in the video the planner struggles")

    path = out_dir / "error_by_window.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    write_caption(
        path, "Error across the video",
        "Bounding box error for each window, in the order the windows appear "
        "in the video.",
        "A flat line means the model handles the whole clip evenly. Spikes "
        "point at moments the model cannot follow, usually fast motion or an "
        "object entering or leaving the frame.")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary_dir", type=Path,
                    help="eval/planner/<model>, the directory holding summary.csv")
    args = ap.parse_args(argv)

    summary_csv = args.summary_dir / "summary.csv"
    if not summary_csv.exists():
        sys.exit(f"no summary.csv in {args.summary_dir}. "
                 "Run tools/planner/eval_plannability.sh first.")

    rows = read_summary(summary_csv)
    if not rows:
        sys.exit(f"{summary_csv} has no rows")

    out_dir = args.summary_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in (plot_reachability(rows, out_dir),
                 plot_ratio(rows, out_dir),
                 plot_error_by_window(rows, out_dir)):
        print(f"wrote {path}")

    solved = [r for r in rows if is_true(r["reachability"])]
    beat = [r for r in solved if is_true(r["beats_baseline"])]
    print(f"\n{len(solved)}/{len(rows)} windows solved")
    if solved:
        print(f"{len(beat)}/{len(solved)} of those beat the straight line")
    return 0


if __name__ == "__main__":
    sys.exit(main())
