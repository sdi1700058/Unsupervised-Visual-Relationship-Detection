#!/usr/bin/env python3
"""Report whether a trained latent actually carries information.

A collapsed encoder maps every state to the same binary code. The
reconstruction then cannot be better than the dataset mean, which pins BCE
near the entropy floor no matter what the optimiser does — the signature is
a whole sweep of runs landing on the same loss regardless of which
hyperparameter was moved. It shows up in autoencoding_test.png as a flat
band instead of a black-and-white grid.

This reads the dumped latents rather than the picture, so the answer comes
back as numbers over ssh:

    python3 tools/diagnose_collapse.py                 # walk out/
    python3 tools/diagnose_collapse.py out/some/run    # one model
    python3 tools/diagnose_collapse.py --limit 5

Needs numpy only. No keras, no weights — it reads all_states.csv,
training_history.csv and aux.json, all of which survive without the model.
"""

import argparse
import json
import os
import sys

import numpy as np


def _read_history(run_dir):
    """Return (first_val, best_val, last_val, n_epochs) or None."""
    path = os.path.join(run_dir, "training_history.csv")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            header = f.readline().strip().split(",")
            rows = [line.strip().split(",") for line in f if line.strip()]
    except OSError:
        return None
    if not rows:
        return None
    # The column is named val_loss on most runs; fall back to anything with
    # "val" and "loss" in it so a renamed metric still reports.
    col = None
    for i, name in enumerate(header):
        if name.strip() == "val_loss":
            col = i
            break
    if col is None:
        for i, name in enumerate(header):
            if "val" in name and "loss" in name:
                col = i
                break
    if col is None:
        return None
    vals = []
    for r in rows:
        if col < len(r):
            try:
                vals.append(float(r[col]))
            except ValueError:
                pass
    if not vals:
        return None
    return vals[0], min(vals), vals[-1], len(vals)


def _read_latents(run_dir):
    """Return the dumped binary latents as (N, bits), or None."""
    for name in ("all_states.csv", "states.csv"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                z = np.loadtxt(path, dtype=np.int8, ndmin=2)
            except (ValueError, OSError):
                continue
            if z.size:
                return z
    return None


def _read_fol(run_dir):
    """Return latent statistics from the extract_fol.py output, or None.

    Training itself only dumps all_states.csv in `dump` mode, but
    run_training.sh runs extract_fol.py by default (EXTRACT_FOL=1), and its
    predicate_analysis.json already carries the per-predicate activation
    rates. That is the same signal, and it is present on ordinary runs.
    """
    fol_dir = os.path.join(run_dir, "fol_output")
    summary_path = os.path.join(fol_dir, "predicate_analysis.json")
    if not os.path.exists(summary_path):
        return None
    try:
        with open(summary_path) as f:
            s = json.load(f)
    except (OSError, ValueError):
        return None

    n_states = s.get("total_states", 0)
    on, off = s.get("num_always_on", 0), s.get("num_always_off", 0)
    var = s.get("num_variable", 0)
    out = {
        "states": n_states,
        "bits": on + off + var,
        "dead_bits": on + off,
        "live_bits": var,
    }
    rates = s.get("predicate_activation_rates")
    if rates:
        out["mean_rate"] = float(np.asarray(rates, dtype=float).mean())

    # The per-state codes give the distinct-code count, which separates "a
    # few bits move" from "the whole latent moves".
    codes_path = os.path.join(fol_dir, "fol_predicates.json")
    if os.path.exists(codes_path):
        try:
            with open(codes_path) as f:
                blob = json.load(f)
        except (OSError, ValueError):
            blob = None
        states = None
        if isinstance(blob, dict):
            for key in ("states", "results"):
                if isinstance(blob.get(key), list):
                    states = blob[key]
                    break
        elif isinstance(blob, list):
            states = blob
        if states:
            codes = [tuple(st["latent_code"]) for st in states
                     if isinstance(st, dict) and "latent_code" in st]
            if codes:
                out["states"] = len(codes)
                out["distinct"] = len(set(codes))
    if "distinct" not in out and n_states:
        # Without the codes, bound it: a latent with no live bit has exactly
        # one reachable code.
        out["distinct"] = 1 if var == 0 else None
    if out.get("distinct") is None:
        out.pop("distinct", None)
    if "distinct" in out and out["states"]:
        out["distinct_frac"] = out["distinct"] / float(out["states"])
    return out


def _read_params(run_dir):
    path = os.path.join(run_dir, "aux.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f).get("parameters", {})
    except (OSError, ValueError):
        return {}


def describe(run_dir):
    """Return a dict of findings for one run directory."""
    out = {"dir": run_dir}

    hist = _read_history(run_dir)
    if hist is not None:
        first, best, last, n = hist
        out["first"], out["best"], out["last"], out["epochs"] = first, best, last, n
        # A run whose loss never moved is a different failure from a run that
        # learned something and then collapsed, so keep them apart.
        out["moved"] = abs(first - best) > 1e-4

    p = _read_params(run_dir)
    for k in ("U", "A", "P", "layer", "lr", "batch_size",
              "max_temperature", "min_temperature", "zerosuppress",
              "preencoder_layers", "preencoder_dimention"):
        if k in p:
            v = p[k]
            out[k] = v[0] if isinstance(v, list) and v else v

    z = _read_latents(run_dir)
    if z is None:
        fol = _read_fol(run_dir)
        if fol is not None:
            out.update(fol)
            out["source"] = "fol_output"
    else:
        n_states, n_bits = z.shape
        distinct = len(np.unique(z, axis=0))
        rate = z.mean(axis=0)
        dead = int(np.sum((rate == 0.0) | (rate == 1.0)))
        # Bits that flip on roughly half the states carry the most; a latent
        # where every bit is stuck is the collapsed case.
        out.update({
            "states": n_states,
            "bits": n_bits,
            "distinct": distinct,
            "distinct_frac": distinct / float(n_states),
            "dead_bits": dead,
            "live_bits": n_bits - dead,
            "mean_rate": float(rate.mean()),
        })
    return out


def verdict(d):
    if "distinct" not in d:
        return "no latents dumped"
    if d["distinct"] <= 1:
        return "COLLAPSED (one code for every state)"
    if d["live_bits"] == 0:
        return "COLLAPSED (every bit stuck)"
    if d["distinct_frac"] < 0.02:
        return "near-collapse"
    if d["live_bits"] < 0.1 * d["bits"]:
        return f"sparse ({d['live_bits']} of {d['bits']} bits live)"
    return "ok"


def find_runs(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        if "training_history.csv" in filenames or "all_states.csv" in filenames:
            yield dirpath


def main():
    ap = argparse.ArgumentParser(
        description="Report whether a trained latent carries information.")
    ap.add_argument("target", nargs="?", default="out",
                    help="a run directory, or a tree to walk. Default: out")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--domain", default=None,
                    help="only paths containing this string")
    args = ap.parse_args()

    if not os.path.exists(args.target):
        sys.exit(f"no such path: {args.target}")

    if os.path.isdir(args.target) and (
            os.path.exists(os.path.join(args.target, "training_history.csv"))
            or os.path.exists(os.path.join(args.target, "all_states.csv"))):
        runs = [args.target]
    else:
        runs = sorted(find_runs(args.target))
        if args.domain:
            runs = [r for r in runs if args.domain in r]

    if not runs:
        sys.exit(f"no runs with training_history.csv or all_states.csv under {args.target}")

    found = [describe(r) for r in runs]
    found.sort(key=lambda d: d.get("best", float("inf")))
    found = found[:args.limit]

    print(f"{len(runs)} run(s) found, showing {len(found)}\n")
    for d in found:
        print(d["dir"])
        if "best" in d:
            moved = "" if d.get("moved") else "   LOSS NEVER MOVED"
            print(f"  val_loss  first {d['first']:.4f}  best {d['best']:.4f}  "
                  f"last {d['last']:.4f}  over {d['epochs']} epochs{moved}")
        cfg = "  ".join(f"{k}={d[k]}" for k in
                        ("U", "A", "P", "layer", "lr", "batch_size",
                         "max_temperature", "min_temperature", "zerosuppress",
                         "preencoder_layers", "preencoder_dimention")
                        if k in d)
        if cfg:
            print(f"  cfg       {cfg}")
        if "distinct" in d:
            print(f"  latent    {d['distinct']} distinct codes over {d['states']} states "
                  f"({100 * d['distinct_frac']:.2f}%)")
            print(f"            {d['live_bits']} of {d['bits']} bits live, "
                  f"mean on-rate {d['mean_rate']:.3f}")
        print(f"  verdict   {verdict(d)}")
        print()


if __name__ == "__main__":
    main()
