#!/usr/bin/env python3
"""tools/plot_training_curve.py — scrape Keras per-epoch loss lines from a
training log (`.out` or `.err`) and emit a learning-curve PNG + caption.

Works on timed-out / killed jobs too — no model save required.

Usage:
    python3 tools/plot_training_curve.py logs/fosae-actiongenome-chair-*.err
    python3 tools/plot_training_curve.py logs/fosae-vidvrd-person-*.err --output curve.png
"""

import os
import re
import sys
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_EPOCH_LINE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s*$|"                              # 'Epoch N/M'
    r"^\s*\d+/\d+.*loss:\s*([\d\.eE+-]+).*?val_loss:\s*([\d\.eE+-]+)"
)
_LOSS_ONLY  = re.compile(r"loss:\s*([\d\.eE+-]+)")
_VAL_LOSS   = re.compile(r"val_loss:\s*([\d\.eE+-]+)")
_HISTORY_LIST = re.compile(r"^history:\s*\[(.+)\]$")


def parse(path):
    train_loss = []   # per-epoch train loss (or sum-of-batches for our loop)
    val_loss   = []   # per-epoch val_loss
    epoch_seen = 0
    with open(path, errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            # final 'history: [...]' dump from generate_logs — full per-epoch train loss
            m_hist = _HISTORY_LIST.match(line)
            if m_hist:
                try:
                    nums = [float(x) for x in m_hist.group(1).split(",")]
                    if not train_loss or len(nums) > len(train_loss):
                        train_loss = nums
                    continue
                except ValueError:
                    pass

            # epoch lines: '... loss: X ... val_loss: Y'
            m_loss = _LOSS_ONLY.search(line)
            m_val  = _VAL_LOSS.search(line)
            if m_loss and m_val:
                train_loss.append(float(m_loss.group(1)))
                val_loss.append(float(m_val.group(1)))
                epoch_seen += 1
                continue

            # 'Epoch N: early stopping'
            if "early stopping" in line.lower():
                pass  # informational only

    return train_loss, val_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log_path", help="Path to a .out or .err log")
    p.add_argument("--output", default=None, help="Output PNG path (default: <log>.curve.png)")
    p.add_argument("--logy",   action="store_true", help="Log-scale Y axis")
    args = p.parse_args()

    train_loss, val_loss = parse(args.log_path)
    if not train_loss and not val_loss:
        sys.exit(f"ERROR: no loss/val_loss lines found in {args.log_path!r}")

    out_path = args.output or (args.log_path.rstrip(".outerr").rstrip(".") + ".curve.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    if train_loss:
        ax.plot(range(1, len(train_loss) + 1), train_loss, label="train loss", marker=".", markersize=3, linewidth=1)
    if val_loss:
        ax.plot(range(1, len(val_loss) + 1),   val_loss,   label="val_loss",   marker=".", markersize=3, linewidth=1, alpha=0.7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if args.logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"Learning curve — {os.path.basename(args.log_path)}\n"
                 f"epochs={len(train_loss) or len(val_loss)}  "
                 f"min_train={min(train_loss) if train_loss else float('nan'):.4f}  "
                 f"min_val={min(val_loss) if val_loss else float('nan'):.4f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot_training_curve] wrote {out_path}")
    print(f"[plot_training_curve] epochs={len(train_loss) or len(val_loss)}  "
          f"min_train={min(train_loss) if train_loss else float('nan'):.4f}  "
          f"min_val={min(val_loss) if val_loss else float('nan'):.4f}")


if __name__ == "__main__":
    main()
