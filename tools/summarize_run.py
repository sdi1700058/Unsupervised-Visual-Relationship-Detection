#!/usr/bin/env python3
"""tools/summarize_run.py — extract converged val_BCE + train_BCE + activation
from a trained model's `training_history.csv`, regardless of CSV column count.

Why this exists: puzzle's CSVLogger writes MULTI-OUTPUT per-tile metrics, so
the first `val_BCE` column (header pos 7) shows only ONE of N tiles, while
the *aggregate* val_BCE (the one we care about, ~1e-7 for the verified
baseline) sits further right. Just `tail -1 ... | awk -F, '{print $7}'`
silently picks the wrong column for puzzle and the right one for video.

This script reads the header, finds ALL `val_BCE*` / `BCE*` / activation*
columns, and reports both the per-column last value AND the overall best
(min) val to give a clean single-line summary.

Usage:
    python3 tools/summarize_run.py <model_dir>
    python3 tools/summarize_run.py <model_dir> --json    # machine-readable
"""

import argparse
import csv
import json
import os
import sys


def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def summarize(model_dir):
    csv_path = os.path.join(model_dir, "training_history.csv")
    if not os.path.isfile(csv_path):
        return {"error": f"no training_history.csv at {csv_path}"}

    with open(csv_path) as f:
        rdr = csv.reader(f)
        rows = list(rdr)
    if len(rows) < 2:
        return {"error": "csv has <2 rows"}

    header = rows[0]
    last   = rows[-1]
    nrows  = len(rows) - 1

    # Collect columns by name buckets. Multi-output puzzle CSV uses suffixes
    # like `val_BCE_1`, `val_BCE_2`, etc. The aggregate is just `val_BCE`.
    def pick(prefix, want_val):
        idxs = []
        for i, name in enumerate(header):
            n = name.strip()
            if want_val:
                if not n.startswith("val_"):
                    continue
                stripped = n[4:]
            else:
                if n.startswith("val_"):
                    continue
                stripped = n
            if stripped == prefix or stripped.startswith(prefix + "_"):
                idxs.append((i, n))
        return idxs

    train_bce_idxs = pick("BCE", want_val=False)
    val_bce_idxs   = pick("BCE", want_val=True)
    train_act_idxs = pick("activation", want_val=False)
    val_act_idxs   = pick("activation", want_val=True)
    train_mse_idxs = pick("MSE", want_val=False)
    val_mse_idxs   = pick("MSE", want_val=True)

    # The "aggregate" column is the one whose name has NO suffix (i.e. exactly
    # "BCE" or "val_BCE"). Per-output columns have suffixes "_0", "_1", ...
    def find_aggregate(idxs, prefix, want_val):
        target = ("val_" + prefix) if want_val else prefix
        for i, n in idxs:
            if n.strip() == target:
                return i, n
        # Fallback: return the LAST (rightmost) index — for puzzle CSV the
        # aggregate often comes after the per-output series.
        return (idxs[-1] if idxs else (None, None))

    def col_value(i):
        return _to_float(last[i]) if (i is not None and i < len(last)) else float("nan")

    def col_min(i):
        if i is None: return float("nan")
        vals = [_to_float(r[i]) for r in rows[1:] if i < len(r)]
        finite = [v for v in vals if v == v]
        return min(finite) if finite else float("nan")

    train_bce_i, train_bce_n = find_aggregate(train_bce_idxs, "BCE", False)
    val_bce_i,   val_bce_n   = find_aggregate(val_bce_idxs,   "BCE", True)
    train_act_i, train_act_n = find_aggregate(train_act_idxs, "activation", False)
    val_act_i,   val_act_n   = find_aggregate(val_act_idxs,   "activation", True)
    train_mse_i, train_mse_n = find_aggregate(train_mse_idxs, "MSE", False)
    val_mse_i,   val_mse_n   = find_aggregate(val_mse_idxs,   "MSE", True)

    summary = {
        "model_dir":   model_dir,
        "epochs":      nrows,
        "ncols":       len(header),
        "train_BCE":   {"col": train_bce_n, "last": col_value(train_bce_i), "min": col_min(train_bce_i)},
        "val_BCE":     {"col": val_bce_n,   "last": col_value(val_bce_i),   "min": col_min(val_bce_i)},
        "train_MSE":   {"col": train_mse_n, "last": col_value(train_mse_i), "min": col_min(train_mse_i)},
        "val_MSE":     {"col": val_mse_n,   "last": col_value(val_mse_i),   "min": col_min(val_mse_i)},
        "train_activation": {"col": train_act_n, "last": col_value(train_act_i)},
        "val_activation":   {"col": val_act_n,   "last": col_value(val_act_i)},
        "n_BCE_cols":      len(train_bce_idxs),
        "n_val_BCE_cols":  len(val_bce_idxs),
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    s = summarize(args.model_dir)
    if "error" in s:
        print(s["error"], file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(s, indent=2, default=str))
        return

    print(f"{os.path.basename(args.model_dir)}")
    print(f"  epochs   = {s['epochs']}    csv cols = {s['ncols']}    "
          f"BCE cols = {s['n_BCE_cols']}  val_BCE cols = {s['n_val_BCE_cols']}")
    for k in ("train_BCE", "val_BCE", "train_MSE", "val_MSE"):
        v = s[k]
        print(f"  {k:14s}  col={v['col']!s:18s}  last={v['last']:.6g}    min={v['min']:.6g}")
    for k in ("train_activation", "val_activation"):
        v = s[k]
        print(f"  {k:14s}  col={v['col']!s:18s}  last={v['last']:.6g}")


if __name__ == "__main__":
    main()
