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

    python3 tools/diagnose_collapse.py                 # walk out/, ranked best first
    python3 tools/diagnose_collapse.py out/some/run    # one model
    python3 tools/diagnose_collapse.py --images        # also print plot paths

Run directories are named by a parameter hash, so pairing a good number
with its pictures means matching a sha1 by eye. --collect does it for you:

    python3 tools/diagnose_collapse.py --limit 10 --collect eval/best

writes eval/best/01_val0.1216_<settings>__autoencoding_test.png and so on,
so the folder sorts best-run-first and every filename says what produced it.

Pure standard library — no numpy, no keras, no weights. It runs on a login
node with nothing activated, and reads only all_states.csv,
training_history.csv, aux.json and fol_output/, all of which outlive the
model file.
"""

import argparse
import json
import os
import re
import sys


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
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            continue
        rows = []
        try:
            with open(path) as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        rows.append(tuple(int(float(v)) for v in parts))
        except (ValueError, OSError):
            continue
        if rows:
            return rows
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
        flat = [v for row in rates for v in row] if isinstance(rates[0], list) else rates
        if flat:
            out["mean_rate"] = sum(float(v) for v in flat) / len(flat)

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
        n_states, n_bits = len(z), len(z[0])
        distinct = len(set(z))
        rate = [sum(row[i] for row in z) / float(n_states) for i in range(n_bits)]
        dead = sum(1 for r in rate if r == 0.0 or r == 1.0)
        # Bits that flip on roughly half the states carry the most; a latent
        # where every bit is stuck is the collapsed case.
        out.update({
            "states": n_states,
            "bits": n_bits,
            "distinct": distinct,
            "distinct_frac": distinct / float(n_states),
            "dead_bits": dead,
            "live_bits": n_bits - dead,
            "mean_rate": sum(rate) / float(n_bits),
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


# The pictures worth looking at first, in the order they answer questions:
# did it reconstruct at all, did it reconstruct data it was not trained on,
# and is the latent a grid or a band.
PLOT_ORDER = (
    "autoencoding_train.png",
    "autoencoding_test.png",
    "booleans_test.png",
    "render_train.png",
    "render_test.png",
    "autoencoding_test_shuffled.png",
)


def images_for(run_dir):
    """Return the run's plots, most informative first, then any others."""
    try:
        present = set(f for f in os.listdir(run_dir) if f.endswith(".png"))
    except OSError:
        return []
    ordered = [f for f in PLOT_ORDER if f in present]
    ordered += sorted(present - set(ordered))
    return [os.path.join(run_dir, f) for f in ordered]


def short_tag(d):
    """A filename-safe label naming the run's distinguishing settings."""
    bits = []
    pl = d.get("preencoder_layers")
    if pl is not None:
        pd = d.get("preencoder_dimention")
        bits.append(f"preenc{pl}x{pd}" if pl else "preencOFF")
    for key, fmt in (("U", "U{}"), ("P", "P{}"), ("layer", "layer{}"),
                     ("lr", "lr{}"), ("batch_size", "b{}"),
                     ("zerosuppress", "zs{}"), ("max_temperature", "tmax{}"),
                     ("min_temperature", "tmin{}")):
        if key in d:
            bits.append(fmt.format(d[key]))
    # The npz stem is in the directory name between "cat" and "_fps".
    base = os.path.basename(d["dir"])
    m = re.search(r"_cat(.+?)_fps", base)
    if m:
        bits.insert(0, m.group(1))
    tag = "_".join(str(b) for b in bits)
    return re.sub(r"[^A-Za-z0-9._-]", "-", tag)[:150]


def collect(found, out_dir):
    """Copy the ranked runs' plots into one flat, readable directory.

    Run directories are named by a parameter hash, so finding the pictures
    for the run that did well means matching a sha1 by eye. This writes
    01_val0.1216_<settings>_autoencoding_test.png instead.
    """
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    n_files = 0
    for rank, d in enumerate(found, 1):
        imgs = images_for(d["dir"])
        if not imgs:
            continue
        val = d.get("best")
        stem = f"{rank:02d}_val{val:.4f}" if val is not None else f"{rank:02d}_valNA"
        stem = f"{stem}_{short_tag(d)}"
        for src in imgs:
            dst = os.path.join(out_dir, f"{stem}__{os.path.basename(src)}")
            try:
                shutil.copyfile(src, dst)
                n_files += 1
            except OSError as e:
                print(f"  could not copy {src}: {e}")
        with open(os.path.join(out_dir, f"{stem}__SOURCE.txt"), "w") as f:
            f.write(d["dir"] + "\n")
    print(f"\ncopied {n_files} images into {out_dir}/")
    print("browse it sorted by name — 01_ is the best run.")


def main():
    ap = argparse.ArgumentParser(
        description="Report whether a trained latent carries information.")
    ap.add_argument("target", nargs="?", default="out",
                    help="a run directory, or a tree to walk. Default: out")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--domain", default=None,
                    help="only paths containing this string")
    ap.add_argument("--sort", choices=("loss", "distinct"), default="loss",
                    help="loss ranks by best val_loss, the usual question. "
                         "distinct ranks by how much of the latent actually "
                         "moves, which is the question the planner asks: a "
                         "model can reach a low val_loss with one code for "
                         "every state, and that model cannot plan.")
    ap.add_argument("--healthy", action="store_true",
                    help="drop runs whose verdict is collapsed or "
                         "near-collapse")
    ap.add_argument("--images", action="store_true",
                    help="print each run's plot paths")
    ap.add_argument("--collect", metavar="DIR", default=None,
                    help="copy the ranked runs' plots into DIR, renamed by "
                         "rank, val_loss and settings")
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

    if args.healthy:
        before = len(found)
        found = [d for d in found
                 if not verdict(d).startswith(("COLLAPSED", "near-collapse",
                                               "no latents"))]
        print(f"{before - len(found)} run(s) dropped as collapsed or "
              f"undumped\n")

    if args.sort == "distinct":
        # How much of the latent moves matters more than the last decimal of
        # val_loss: a run with one code for every state cannot plan, whatever
        # its loss says.
        found.sort(key=lambda d: (-d.get("distinct_frac", 0.0),
                                  d.get("best", float("inf"))))
    else:
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
        if args.images:
            for img in images_for(d["dir"]):
                print(f"  image     {img}")
        print()

    if args.collect:
        collect(found, args.collect)


if __name__ == "__main__":
    main()
