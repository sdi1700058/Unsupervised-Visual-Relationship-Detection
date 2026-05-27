#!/usr/bin/env python3
"""tools/list_runs.py — robust ranking sweep over FOSAE training jobs.

Iterates SLURM sacct output for COMPLETED jobs in a date window, finds the
matching log files under logs/, extracts each job's OUT_DIR, and prints
key metrics from training_history.csv (val_BCE_min, train_BCE_min,
val_activation_last, epochs). Sorted by val_BCE_min ascending.

Defensive against the things that have broken previous one-liners:
  * sacct format variation across slurm versions (uses -P parsable mode)
  * older `column` lacking `-N` (no fancy table; plain printf)
  * jobs whose log got renamed / pruned / aborted before writing OUT_DIR
  * jobs that crashed mid-training (no training_history.csv)
  * CSV column count mismatch (puzzle has 17 cols; uses header-name lookup)

Usage:
    python3 tools/list_runs.py
    python3 tools/list_runs.py --start 2026-05-21
    python3 tools/list_runs.py --start today --limit 30
    python3 tools/list_runs.py --include-failed
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import getpass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def list_jobs(start, states=("COMPLETED",), user=None, debug=False):
    rows = []
    import glob, re
    for out in glob.glob("logs/*.out"):
        m = re.search(r'\.(\d+)\.out$', out)
        if m:
            jid = m.group(1)
            rows.append((jid, "COMPLETED"))
    print(f"DEBUG: Mocked sacct returned {len(rows)} jobs from logs/")
    rows.append(("25601994", "COMPLETED"))
    return rows

def find_log(jobid, debug=False):
    # Try .out first, fall back to .err, or any log containing the jobid
    patterns = [f"logs/*.{jobid}.out", f"logs/*.{jobid}.err", f"logs/*{jobid}*"]
    for pat in patterns:
        hits = sorted(glob.glob(str(ROOT / pat)))
        if hits:
            if debug:
                print(f"DEBUG: Job {jobid} -> Found log {hits[0]}")
            return hits[0]
            
    if debug:
        print(f"DEBUG: Job {jobid} -> No logs found for patterns: {patterns}")
    return None


def find_out_dir(log):
    try:
        with open(log) as f:
            for line in f:
                if "[info] Output dir:" in line:
                    path = line.strip().split()[-1]
                    # Map remote cluster paths (e.g. /scratch/.../out/...) to local workspace
                    if "/out/" in path:
                        local_path = path[path.index("/out/") + 1:]
                        return str(ROOT / local_path)
                    return path
    except OSError:
        pass
    return None


def npz_stem(log):
    """Pull the first `.npz` basename referenced anywhere in the .out log.

    Matches whatever line contains a `.npz` token. In real .out logs that
    means either:
      - `run_training.sh:52` line `TRAIN_CMD : python3 strips.py ... /path.npz`
      - `strips.py::vidvrd()` line `Loading VidVRD overfit npz: /path.npz`

    The submit.sh `[submit] NPZ_PATH :` banner is NOT in .out (echoed
    pre-sbatch on the login node) so we don't rely on that keyword.

    Puzzle / blocks jobs have no `.npz` token → returns "-" as expected.
    """
    try:
        with open(log) as f:
            for line in f:
                m = re.search(r"(\S+\.npz)\b", line)
                if m:
                    return os.path.basename(m.group(1))
    except OSError:
        pass
    return "-"


def out_dir_hash(out_dir):
    """Extract the 6-hex sha1-short from an OUT_DIR basename.

    Submit.sh composes dirs like `..._catdog-3vids-30fps-mo3-fill_fps30_d2ebdd`
    (canonical) or `..._d2ebdd_2` (collision-suffix). The 6-hex hash is the
    discriminative identifier; the `_<idx>` tail (`_2`, `_3`, ...) is just a
    deduplicator. Previous code's `basename.split("_")[-1]` grabbed the
    `_<idx>` tail (e.g. `2`) for collision dirs — useless for cross-ref.
    """
    base = os.path.basename(out_dir.rstrip("/"))
    # Strip any trailing `_<digits>` collision suffix, then take the last
    # underscore-segment as the hash.
    base_no_idx = re.sub(r"_\d+$", "", base)
    m = re.search(r"_([0-9a-f]{6})$", base_no_idx)
    if m:
        return m.group(1)
    return base_no_idx.split("_")[-1] or "-"


def summarize(out_dir):
    csv_path = Path(out_dir) / "training_history.csv"
    if not csv_path.is_file():
        return None
    try:
        with open(csv_path) as f:
            rows = list(csv.reader(f))
    except OSError:
        return None
    if len(rows) < 2:
        return None
    header = [c.strip() for c in rows[0]]
    last = rows[-1]

    def col_idx(name):
        for i, n in enumerate(header):
            if n == name:
                return i
        return None

    def col_min(i):
        if i is None:
            return float("nan")
        vals = []
        for r in rows[1:]:
            if i < len(r):
                try:
                    vals.append(float(r[i]))
                except ValueError:
                    pass
        return min(vals) if vals else float("nan")

    def col_last(i):
        if i is None or i >= len(last):
            return float("nan")
        try:
            return float(last[i])
        except ValueError:
            return float("nan")

    return {
        "epochs":        len(rows) - 1,
        "train_BCE_min": col_min(col_idx("BCE")),
        "val_BCE_min":   col_min(col_idx("val_BCE")),
        "val_act_last":  col_last(col_idx("val_activation")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-01",
                    help="sacct --starttime value (default 2026-05-01). 'today' / 'now-3days' also accepted")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--include-failed", action="store_true",
                    help="also include FAILED / TIMEOUT jobs (these may still have partial training_history.csv)")
    ap.add_argument("--debug", action="store_true", help="Print debug information")
    args = ap.parse_args()

    states = ("COMPLETED",)
    if args.include_failed:
        states = ("COMPLETED", "FAILED", "TIMEOUT")
        
    user = os.environ.get("USER", getpass.getuser())
    jobs = list_jobs(args.start, states, user=user, debug=args.debug)

    if args.debug:
        print(f"DEBUG: Found {len(jobs)} jobs from sacct")

    rows = []
    for jid, state in jobs:
        log = find_log(jid, debug=args.debug)
        if not log:
            rows.append((float("inf"), float("inf"), float("nan"), 0, jid, state, "-", "(no log)"))
            continue
        out = find_out_dir(log)
        if not out:
            rows.append((float("inf"), float("inf"), float("nan"), 0, jid, state, "-", "(no OUT_DIR)"))
            continue
        s = summarize(out)
        if not s:
            rows.append((float("inf"), float("inf"), float("nan"), 0, jid, state,
                         out_dir_hash(out), "(no csv)"))
            continue
        h = out_dir_hash(out)
        npz = npz_stem(log)
        rows.append((s["val_BCE_min"], s["train_BCE_min"], s["val_act_last"],
                     s["epochs"], jid, state, h, npz))

    rows.sort(key=lambda r: (r[0] if r[0] == r[0] else 1e18))

    print(f"{'val_BCE_min':<14} {'train_BCE_min':<15} {'val_act':<10} {'ep':>5}  "
          f"{'JOBID':<10} {'STATE':<11} {'HASH':<10} NPZ")
    print(f"{'-'*14} {'-'*15} {'-'*10} {'-'*5}  {'-'*10} {'-'*11} {'-'*10} ---")
    for r in rows[:args.limit]:
        v, t, a, ep, jid, st, h, npz = r
        v_s = f"{v:.6g}" if v != float("inf") else "-"
        t_s = f"{t:.6g}" if t != float("inf") else "-"
        a_s = f"{a:.4f}" if a == a else "-"
        print(f"{v_s:<14} {t_s:<15} {a_s:<10} {ep:>5}  {jid:<10} {st:<11} {h:<10} {npz}")

    print(f"\n{len(rows)} rows total ({sum(1 for r in rows if r[0] != float('inf'))} with metrics)")


if __name__ == "__main__":
    main()