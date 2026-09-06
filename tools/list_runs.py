#!/usr/bin/env python3
"""Rank FOSAE training jobs by reading the scheduler.

Reads `sacct` in parsable mode for jobs in a date window, finds each job's log
under `logs/`, extracts its `OUT_DIR`, and prints the metrics from
`training_history.csv`, sorted by `val_BCE_min` ascending.

**It runs only where `sacct` runs, which is the cluster.** Off the cluster it
raises `NoScheduler` and prints nothing. That is deliberate and it is the whole
point of this module's history: until 2026-09-06 `list_jobs` did not call
`sacct` at all. It globbed `logs/*.out`, appended one hard-coded
`(jobid, "COMPLETED")` row on every invocation, and announced itself as a mock
in a debug line -- while this docstring claimed it iterated sacct output and
was defensive about format variation across Slurm versions.

The invented job id is deliberately not repeated here. `test_list_runs.py`
greps this file for it, so naming it in prose would disarm the guard.

A tool that reports which runs exist was inventing one of them and documenting
itself as authoritative. A fabricated job id in a listing is the same class of
error as a number in a document that no file supports, and worse for being
automated. So the rule here is that no answer is produced unless the scheduler
gave one.

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


class NoScheduler(RuntimeError):
    """`sacct` is not on this machine, so no job listing can be produced."""


# A job step, not a job. sacct emits `12345`, `12345.batch` and `12345.extern`
# for one submission, and counting all three would treble every total.
_STEP = re.compile(r"^\d+\.")


def parse_sacct(text):
    """`[(jobid, state)]` from sacct parsable output, steps excluded.

    The state may carry a reason, as in `CANCELLED by 12345`. Only the state
    is kept, because the reason is not one.
    """
    rows = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        jobid, _, state = line.partition("|")
        if jobid == "JobID" or _STEP.match(jobid):
            continue
        rows.append((jobid, state.split()[0] if state.split() else ""))
    return rows


def list_jobs(start, states=("COMPLETED",), user=None, debug=False,
              sacct="sacct"):
    """Jobs from the scheduler, or `NoScheduler` if there is none.

    Never invents a row. See the module docstring for why that has to be said.
    """
    cmd = [sacct, "-X", "-P", "-n", "--format=JobID,State",
           "-S", start, "-u", user or getpass.getuser()]
    if states:
        cmd += ["--state", ",".join(states)]
    if debug:
        print("DEBUG: %s" % " ".join(cmd))
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except OSError as exc:
        raise NoScheduler(
            "sacct is not available on this machine, so no job listing can be "
            "produced. This tool reads the cluster scheduler and runs only "
            "there. (%s)" % exc)
    except subprocess.CalledProcessError as exc:
        raise NoScheduler(
            "sacct failed with exit %d: %s"
            % (exc.returncode,
               exc.output.decode("utf-8", "replace").strip()[:200]))
    return parse_sacct(out.decode("utf-8", "replace"))

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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-01",
                    help="sacct --starttime value (default 2026-05-01). 'today' / 'now-3days' also accepted")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--include-failed", action="store_true",
                    help="also include FAILED / TIMEOUT jobs (these may still have partial training_history.csv)")
    ap.add_argument("--debug", action="store_true", help="Print debug information")
    ap.add_argument("--sacct", default="sacct",
                    help="scheduler command to read (default sacct)")
    args = ap.parse_args(argv)

    states = ("COMPLETED",)
    if args.include_failed:
        states = ("COMPLETED", "FAILED", "TIMEOUT")
        
    user = os.environ.get("USER", getpass.getuser())
    try:
        jobs = list_jobs(args.start, states, user=user, debug=args.debug,
                         sacct=args.sacct)
    except NoScheduler as exc:
        sys.stderr.write("%s\n" % exc)
        return 1

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
    return 0


if __name__ == "__main__":
    sys.exit(main())