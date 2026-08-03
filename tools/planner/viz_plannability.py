#!/usr/bin/env python3
"""tools/planner/viz_plannability.py — SPEC §T H4.

Reads `eval/planner/<model_stem>/summary.csv` and writes three PNGs plus
`.caption.md` siblings under `eval/planner/<model_stem>/viz/`:

    plannability_reachability_by_route.png
        Bar chart: fraction of videos with `reachability=true` per route.

    plannability_bbox_mse_hist.png
        Histogram of per-plan bbox_mse_mean across all successful plans,
        one series per route.

    plannability_predicate_firing_heatmap.png
        (Requires per-run bfs_trace.json / plan_latents; falls back to a
         placeholder heading if the trace files are absent.)

Determinism (SPEC §V V15): same summary.csv → same PNGs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


CAPTION_HEADER = "# {title}\n\n**What.** {what}\n\n**Why.** {why}\n\n**How to read.** {how_to_read}\n"


def _load_summary(csv_path):
    """Return list of dicts, one per row."""
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _write_caption(png_path, what, why, how_to_read):
    md = png_path.with_suffix(".caption.md")
    md.write_text(CAPTION_HEADER.format(
        title=png_path.stem, what=what, why=why, how_to_read=how_to_read))


def reachability_by_route(rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_route = {}
    for r in rows:
        route = r["route"]
        reach = str(r.get("reachability", "false")).lower() == "true"
        per_route.setdefault(route, []).append(reach)

    routes = sorted(per_route)
    rates = [sum(per_route[r]) / max(1, len(per_route[r])) for r in routes]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(routes, rates, color=["#4C78A8", "#F58518", "#54A24B"][:len(routes)])
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{rate:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Reachability rate")
    ax.set_xlabel("Planner route")
    ax.set_title(f"Reachability by route (n={len(rows) // max(1, len(routes))} videos each)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    _write_caption(
        out_png,
        what="Fraction of videos for which each planner route returned a valid plan within the time budget.",
        why="Compares planning capability across routes A (upstream AMA3), B (native PDDL), and C (BFS smoke).",
        how_to_read="Higher bar = more videos solvable by that route. A route that scores 0 is not producing plans; a route that scores 1 solves every video in the eval set.",
    )
    return out_png


def bbox_mse_hist(rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_route = {}
    for r in rows:
        if str(r.get("reachability", "false")).lower() != "true":
            continue
        try:
            v = float(r.get("bbox_mse_mean", ""))
        except (TypeError, ValueError):
            continue
        per_route.setdefault(r["route"], []).append(v)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"a": "#4C78A8", "b": "#F58518", "c": "#54A24B"}
    for route, vals in sorted(per_route.items()):
        if not vals:
            continue
        ax.hist(vals, bins=20, alpha=0.6, label=f"route {route} (n={len(vals)})",
                 color=colors.get(route, "#888"))
    ax.set_xlabel("bbox_mse_mean (normalized canvas coords, CANVAS=480)")
    ax.set_ylabel("Count of successful plans")
    ax.set_title("bbox MSE distribution across successful plans")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    _write_caption(
        out_png,
        what="Per-plan mean-squared error between the plan's decoded bboxes and the ground-truth video bboxes.",
        why="Measures faithfulness: does the plan trajectory reproduce the object motion in the source video?",
        how_to_read="Left-skewed (mass near zero) = plans reconstruct the trajectory faithfully. Right-heavy tail = plans diverge from ground truth.",
    )
    return out_png


def predicate_firing_heatmap(model_stem_dir, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Scan model_stem_dir/*/route_*/plan_*/{bfs_trace.json,metrics.json} for
    # plan latents. Aggregate per-bit firing count across all plan steps.
    firing_counts = None
    n_steps = 0
    for video_dir in Path(model_stem_dir).iterdir():
        if not video_dir.is_dir():
            continue
        for route_dir in video_dir.glob("route_*"):
            for plan_dir in route_dir.glob("plan_*"):
                trace_file = plan_dir / "bfs_trace.json"
                if not trace_file.exists():
                    continue
                with trace_file.open() as f:
                    payload = json.load(f)
                latents = payload.get("plan_latents")
                if not latents:
                    continue
                arr = np.array(latents, dtype=np.int8)
                if firing_counts is None:
                    firing_counts = np.zeros(arr.shape[1], dtype=np.int64)
                firing_counts += arr.sum(axis=0)
                n_steps += arr.shape[0]

    if firing_counts is None or n_steps == 0:
        # Placeholder: no plan traces yet.
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no plan traces yet\n(needs route c bfs_trace.json)",
                 ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        _write_caption(
            out_png,
            what="Per-latent-bit firing frequency across every plan step in the eval batch.",
            why="Shows which predicates are load-bearing during planning versus predicates that stay silent.",
            how_to_read="Placeholder — this figure will populate once Route C plan traces exist. Rerun `eval_plannability.sh` first.",
        )
        return out_png

    # Reshape flat firing counts into (U, P) if the caller supplies U*P layout.
    total = firing_counts.shape[0]
    U = int(np.sqrt(total))
    while total % U != 0 and U > 1:
        U -= 1
    P = total // U
    grid = (firing_counts / n_steps).reshape(U, P)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, aspect="auto", cmap="hot")
    ax.set_xlabel(f"Predicate index (P={P})")
    ax.set_ylabel(f"Unit index (U={U})")
    ax.set_title(f"Predicate firing frequency across {n_steps} plan steps")
    fig.colorbar(im, ax=ax, label="Firing rate")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    _write_caption(
        out_png,
        what="Per-latent-bit firing rate across every plan step in the eval batch.",
        why="Shows which predicates are load-bearing during planning versus predicates that stay silent.",
        how_to_read=f"Row = predicate unit (0..{U-1}). Column = predicate index within unit (0..{P-1}). Warmer cells = predicate fires more often during plans. Solid cold rows may indicate dead units.",
    )
    return out_png


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_stem_dir", type=Path,
                    help="path to eval/planner/<model_stem>/ (contains summary.csv)")
    args = ap.parse_args(argv)

    stem = args.model_stem_dir.resolve()
    csv_path = stem / "summary.csv"
    if not csv_path.exists():
        sys.exit(f"no summary.csv at {csv_path}. Run eval_plannability.sh first.")

    viz_dir = stem / "viz"
    viz_dir.mkdir(exist_ok=True)

    rows = _load_summary(csv_path)
    print(f"[viz_plannability] read {len(rows)} rows from {csv_path}")

    p1 = reachability_by_route(rows, viz_dir / "plannability_reachability_by_route.png")
    print(f"[viz_plannability] wrote {p1}")
    p2 = bbox_mse_hist(rows, viz_dir / "plannability_bbox_mse_hist.png")
    print(f"[viz_plannability] wrote {p2}")
    p3 = predicate_firing_heatmap(stem, viz_dir / "plannability_predicate_firing_heatmap.png")
    print(f"[viz_plannability] wrote {p3}")

    print(f"[viz_plannability] all 3 PNG + 3 caption.md under {viz_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
