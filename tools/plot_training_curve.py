#!/usr/bin/env python3
"""tools/plot_training_curve.py — emit a learning-curve PNG + caption.

Tries, in order:
  1. `training_history.csv` next to the saved model (CSVLogger, future jobs)
  2. TensorBoard event files in `<model_dir>/logs/.../events.out.tfevents.*`
     (works on TIMED-OUT / killed jobs as long as the events dir exists)
  3. regex scrape of a `.err` / `.out` log file (last-resort; latplan's custom
     loop emits only a final `history: [...]` line which suffices)

Usage:
    python3 tools/plot_training_curve.py <model_dir_or_log_file> [--logy] [--output X.png]
"""

import os
import re
import sys
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_LOSS_ONLY    = re.compile(r"loss:\s*([\d\.eE+-]+)")
_VAL_LOSS     = re.compile(r"val_loss:\s*([\d\.eE+-]+)")
_HISTORY_LIST = re.compile(r"^history:\s*\[(.+)\]$")

# latplan custom bar_update format. Example:
#   "4:07:50 79 | 2026-05-24 21:20:56 status: [t] BCE   0.0729  MSE   0.0163  ...
#                                                  ... loss     10.4  loss0   0.0682 ..."
_STATUS_HEAD = re.compile(
    r"^\s*(\d+:\d+:\d+)\s+(\d+)\s+\|\s+\S+\s+\S+\s+status:\s*\[([tv])\]\s+(.*)$"
)


def _kv_pairs(rest):
    """Tokenize 'key  val  key2  val2' (key/val separated by 2+ spaces)."""
    out = {}
    for tok in re.split(r"\s{2,}", rest.strip()):
        parts = tok.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def _parse_log(path):
    """Scrape per-epoch metrics from latplan's `bar_update` `status:` lines
    AND legacy Keras `loss:`/`val_loss:` colon-format AND a final `history: [...]`.

    Returns dict of series keyed by metric name; train metrics get the raw name,
    val metrics are prefixed `val_`.
    """
    # epoch -> {'t': {...}, 'v': {...}}
    epoch_buckets = {}
    legacy_train, legacy_val = [], []
    final_history = []

    with open(path, errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")

            m_hist = _HISTORY_LIST.match(line)
            if m_hist:
                try:
                    final_history = [float(x) for x in m_hist.group(1).split(",")]
                except ValueError:
                    pass
                continue

            m_status = _STATUS_HEAD.match(line)
            if m_status:
                epoch = int(m_status.group(2))
                tag   = m_status.group(3)   # 't' or 'v'
                metrics = _kv_pairs(m_status.group(4))
                epoch_buckets.setdefault(epoch, {}).setdefault(tag, {}).update(metrics)
                continue

            # legacy Keras path (model.fit-style)
            m_loss = _LOSS_ONLY.search(line)
            m_val  = _VAL_LOSS.search(line)
            if m_loss and m_val:
                legacy_train.append(float(m_loss.group(1)))
                legacy_val.append(float(m_val.group(1)))

    # ---- assemble series ----
    series = {}
    if epoch_buckets:
        epochs = sorted(epoch_buckets.keys())
        all_train_keys = set()
        all_val_keys   = set()
        for e in epochs:
            all_train_keys.update(epoch_buckets[e].get("t", {}).keys())
            all_val_keys  .update(epoch_buckets[e].get("v", {}).keys())
        for k in all_train_keys:
            series[k]            = [epoch_buckets[e].get("t", {}).get(k, float("nan")) for e in epochs]
        for k in all_val_keys:
            series[f"val_{k}"]   = [epoch_buckets[e].get("v", {}).get(k, float("nan")) for e in epochs]
        series["_epochs"] = epochs

    if legacy_train or legacy_val:
        # only use legacy data if status-format didn't fire
        if not series:
            series["loss"]     = legacy_train
            series["val_loss"] = legacy_val
            series["_epochs"]  = list(range(1, max(len(legacy_train), len(legacy_val)) + 1))

    if final_history and "loss" not in series:
        series["loss"]    = final_history
        series["_epochs"] = list(range(1, len(final_history) + 1))

    return series


def _parse_csv(path):
    """B17 CSVLogger output: epoch,loss,val_loss[,...]"""
    import csv
    series = {}
    with open(path) as f:
        rdr = csv.DictReader(f)
        for col in rdr.fieldnames or []:
            series[col] = []
        for row in rdr:
            for k, v in row.items():
                try:
                    series.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    pass
    epochs = series.pop("epoch", None)
    if epochs:
        series["_epochs"] = [int(e) for e in epochs]
    return series


def _parse_tfevents(events_path):
    """Read scalars from a TensorBoard events file. Returns dict of series."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(events_path).Reload()
        tags = ea.Tags().get("scalars", [])
        series = {}
        epochs = None
        for t in tags:
            scs = ea.Scalars(t)
            series[t] = [s.value for s in scs]
            if epochs is None:
                epochs = [s.step for s in scs]
        if epochs:
            series["_epochs"] = epochs
        return series
    except ImportError:
        try:
            from tensorflow.python.summary.summary_iterator import summary_iterator
        except Exception as e:
            raise RuntimeError(f"no tensorboard / tf summary reader available: {e}")
        series = {}
        for ev in summary_iterator(events_path):
            for v in ev.summary.value:
                series.setdefault(v.tag, []).append(v.simple_value)
        return series


def _find_events_dir(model_dir):
    """Locate the events.out.tfevents.* file under <model_dir>/logs/<run>/."""
    import glob
    candidates = sorted(glob.glob(os.path.join(model_dir, "logs", "*", "events.out.tfevents.*")))
    return candidates[-1] if candidates else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("target", help="model_dir OR log file (.out/.err) OR events.out.tfevents.*")
    p.add_argument("--output", default=None, help="Output PNG path")
    p.add_argument("--logy",   action="store_true")
    args = p.parse_args()

    source = None
    series = {}

    if os.path.isdir(args.target):
        csv_path = os.path.join(args.target, "training_history.csv")
        if os.path.isfile(csv_path):
            series = _parse_csv(csv_path)
            source = csv_path
        if not series:
            ev = _find_events_dir(args.target)
            if ev:
                series = _parse_tfevents(ev)
                source = ev
    elif os.path.basename(args.target).startswith("events.out.tfevents"):
        series = _parse_tfevents(args.target)
        source = args.target
    else:
        series = _parse_log(args.target)
        source = args.target

    if not series:
        sys.exit(f"ERROR: no loss data extractable from {args.target!r} "
                 f"(tried csv / tfevents / log-scrape).")

    print(f"[plot_training_curve] source = {source}")
    print(f"[plot_training_curve] series = {sorted(k for k in series if k != '_epochs')}")

    epochs = series.pop("_epochs", None)
    train_loss = series.pop("loss", [])
    val_loss   = series.pop("val_loss", [])

    out_path = args.output or (
        os.path.join(args.target, "training_curve.png") if os.path.isdir(args.target)
        else args.target.rstrip(".outerr").rstrip(".") + ".curve.png")

    # Keep only "interesting" series; drop per-batch noise (loss0..lossN, val_loss0..)
    def _interesting(k):
        return not re.match(r"^(val_)?loss\d+$", k)

    extras = {k: v for k, v in series.items() if _interesting(k) and any(v)}

    n_panels = 1 + min(len(extras), 3)
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3 * n_panels), squeeze=False)

    # Panel 0 — aggregate loss
    ax = axes[0, 0]
    if train_loss:
        ax.plot(epochs or range(1, len(train_loss) + 1), train_loss, label="train loss", marker=".", ms=3, lw=1)
    if val_loss:
        ax.plot(epochs[:len(val_loss)] if epochs else range(1, len(val_loss) + 1), val_loss,
                label="val_loss", marker=".", ms=3, lw=1, alpha=0.7)
    ax.set_ylabel("loss")
    ax.set_title(f"Aggregate loss — {os.path.basename(source)}")
    if args.logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if train_loss or val_loss:
        ax.legend()

    # Pick the top-3 most informative extra metrics (prefer BCE, MSE, then alphabetical)
    priority = ["BCE", "MSE", "val_BCE", "val_MSE", "activation", "preencoder_l1"]
    chosen = [k for k in priority if k in extras][:3]
    if len(chosen) < 3:
        for k in sorted(extras.keys()):
            if k not in chosen and len(chosen) < 3:
                chosen.append(k)

    for i, k in enumerate(chosen, start=1):
        if i >= n_panels:
            break
        ax = axes[i, 0]
        v = extras[k]
        xs = (epochs[:len(v)] if epochs else range(1, len(v) + 1))
        ax.plot(xs, v, label=k, marker=".", ms=3, lw=1, color=f"C{i}")
        ax.set_ylabel(k)
        ax.grid(True, alpha=0.3)
        ax.legend()
        if args.logy and all(x > 0 for x in v if x == x):
            ax.set_yscale("log")

    axes[-1, 0].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot_training_curve] wrote {out_path}")
    print(f"[plot_training_curve] panels: aggregate + {chosen}")
    if train_loss:
        print(f"[plot_training_curve] train loss: epochs={len(train_loss)} min={min(train_loss):.4f} last={train_loss[-1]:.4f}")
    for k in chosen:
        v = extras[k]
        print(f"[plot_training_curve] {k}: epochs={len(v)} min={min(v):.4f} last={v[-1]:.4f}")


if __name__ == "__main__":
    main()
