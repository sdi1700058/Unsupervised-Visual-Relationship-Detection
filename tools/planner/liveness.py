#!/usr/bin/env python3
"""Which latent exports carry information, and which collapsed.

**The failure this measures.** An export whose encoder emits one code for every
frame is dead. There is no state for a planner to search, so every downstream
metric scores it badly, and it scores badly for a reason that has nothing to do
with the property under test. A dead export read as a result says "this latent
shape plans poorly" when the truth is "this training run produced nothing".

**Measured on 2026-09-05, and the reason this module exists.** Four exports
under `eval/exports/` hold exactly one distinct latent with every bit zero:
`H14-U20_A2_P10-150010`, `H14-U40_A2_P20-150010`, and the 8,610-frame
`U20_A2_P10` and `U40_A2_P20` runs over the 88 screened clips. Four separate
pieces of work each rediscovered them by hand. The collapse does not track the
latent width: 200 bits is dead at `U20 P10` and alive at `U40 P5`, which holds
1,856 distinct codes over the same frames.

**The training signature, measured the same day.** Over the 14-cell shape grid,
five different shapes converged to a validation loss of 0.5245 plus or minus
0.0005 after 3,000 epochs, and every shape at that value exports zero bits set.
The shapes that encode anything sit between 0.198 and 0.425. Identical loss to
three decimal places across five architectures is one degenerate optimum: emit
the constant latent and let the decoder predict the mean.

That gives the `val_loss < 0.5` gate in `SPEC.md` C17 a different meaning from
the one it was given. It is not a quality threshold. It separates models that
encode something from models that collapsed, which is a stronger statement and
a cheaper test than running a planner to find out.

**The plateau value is measured here, never hard-coded.** 0.5245 is what this
dataset and this decoder produced. A different dataset has a different constant
solution, and writing this one into the source would silently mislabel it.

    python3 tools/planner/liveness.py --exports eval/exports
    python3 tools/planner/liveness.py --exports eval/exports --strict
    python3 tools/planner/liveness.py --exports eval/exports \\
        --train-csv eval/exports/G4_train.csv

`--strict` exits non-zero when anything is dead, so a scoring step can refuse
to report a grid that is mostly collapse.

Standard library plus numpy. Python 3.6 clean, because the cluster runs 3.6.1.
"""

import argparse
import csv
import glob
import json
import os

try:
    import numpy as np
except ImportError:
    np = None


# How close two validation losses must be to count as the same plateau. The
# five collapsed shapes agreed to within 0.0009 of each other, and the nearest
# live shape sat 0.10 away, so anything in this range separates them cleanly.
PLATEAU_TOLERANCE = 0.002

# A plateau needs at least this many runs. One value cannot agree with itself,
# and calling a single run a plateau would flag every ordinary result.
PLATEAU_MIN_RUNS = 2


def export_facts(path):
    """`{frames, bits, distinct, ones, dead}` for one export.

    Never raises. A survey over a directory has to survive one bad file,
    because the alternative is that a single truncated npz hides the state of
    every other export.
    """
    out = {"path": path, "name": os.path.basename(path), "frames": None,
           "bits": None, "distinct": None, "ones": None, "dead": False}
    if np is None:
        out["error"] = "numpy is not available"
        return out
    try:
        # No allow_pickle. These arrays are plain, so the pickle path buys
        # nothing and refusing it keeps a hostile npz from executing.
        data = np.load(path)
        if "latents" not in data.files:
            out["error"] = "no latents array"
            return out
        latents = np.asarray(data["latents"])
    except Exception as exc:                      # noqa: BLE001 - see docstring
        out["error"] = str(exc)
        return out
    if latents.ndim < 2 or latents.shape[0] == 0:
        out["error"] = "latents is not a frame-by-bit array"
        return out
    flat = latents.reshape(latents.shape[0], -1)
    out["frames"] = int(flat.shape[0])
    out["bits"] = int(flat.shape[1])
    out["distinct"] = int(len(np.unique(flat, axis=0)))
    out["ones"] = int(flat.sum())
    out["dead"] = out["distinct"] <= 1
    return out


def survey(directory, pattern="*.npz"):
    """Every export under `directory`, dead ones first.

    Dead first because they are the finding. Sorting by name would bury them,
    and this survey exists precisely so that nobody has to go looking.
    """
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    rows = [export_facts(p) for p in paths]
    rows.sort(key=lambda r: (not r["dead"], r["name"]))
    return rows


def read_train_csv(path):
    """`[{u, p, best_val, epochs}]` from a training summary, or an empty list."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as handle:
        for raw in csv.DictReader(handle):
            try:
                rows.append({"u": int(raw["u"]), "p": int(raw["p"]),
                             "best_val": float(raw["best_val"]),
                             "epochs": int(raw.get("epochs") or 0)})
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def plateau_shapes(rows, tolerance=PLATEAU_TOLERANCE,
                   min_runs=PLATEAU_MIN_RUNS):
    """The `(u, p)` shapes sitting on the largest shared loss value.

    The plateau is found in the data rather than compared against a constant.
    The largest cluster of runs whose best validation loss agrees within
    `tolerance` is the degenerate solution, because a real optimum depends on
    the architecture and a constant-output solution does not.
    """
    values = sorted(rows, key=lambda r: r["best_val"])
    best = []
    for i, anchor in enumerate(values):
        group = [r for r in values[i:]
                 if r["best_val"] - anchor["best_val"] <= tolerance]
        if len(group) > len(best):
            best = group
    if len(best) < min_runs:
        return set()
    return set((r["u"], r["p"]) for r in best)


def _esc(text):
    """Escape for SVG text. A raw angle bracket is invalid XML."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def render_svg(rows, limit=24):
    """A bar per export, its height the distinct-code count, dead ones marked."""
    shown = rows[:limit]
    width, height = 760, 330
    pad, base = 60, 240
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
             'viewBox="0 0 %d %d">' % (width, height, width, height),
             '<style>text{font-family:sans-serif}.t{font-size:15px;'
             'font-weight:bold}.l{font-size:9px}.n{font-size:11px;fill:#555}'
             '</style>',
             '<rect width="%d" height="%d" fill="white"/>' % (width, height)]
    dead = [r for r in rows if r["dead"]]
    parts.append('<text x="%d" y="26" class="t">Latent exports: distinct codes '
                 'per export</text>' % pad)
    parts.append('<text x="%d" y="46" class="n">%d of %d carry one code and '
                 'are dead. A dead export has no state to plan over, so its '
                 'score measures the training run and not the latent shape.'
                 '</text>' % (pad, len(dead), len(rows)))
    if not shown:
        parts.append('<text x="%d" y="%d" class="n">No exports found.</text>'
                     % (pad, base))
        parts.append("</svg>")
        return "\n".join(parts)

    counts = [r["distinct"] or 0 for r in shown]
    top = max(counts) or 1
    step = max(1, (width - 2 * pad) // len(shown))
    for i, row in enumerate(shown):
        value = row["distinct"] or 0
        # Square root, so a 3000-code export does not flatten every other bar
        # into the axis. The number is printed, so the scale only has to rank.
        bar = int(170 * (float(value) / top) ** 0.5)
        x = pad + i * step
        fill = "#c05621" if row["dead"] else "#2b6cb0"
        parts.append('<rect x="%d" y="%d" width="%d" height="%d" fill="%s"/>'
                     % (x, base - bar, max(2, step - 4), max(1, bar), fill))
        parts.append('<text x="%d" y="%d" class="l">%s</text>'
                     % (x, base - bar - 4, _esc(value)))
        label = row["name"].replace(".npz", "")[:14]
        parts.append('<text x="%d" y="%d" class="l" transform="rotate(52 %d %d)"'
                     '>%s</text>' % (x, base + 12, x, base + 12, _esc(label)))
    parts.append('<text x="%d" y="%d" class="n">Bar height is the square root '
                 'of the distinct-code count, so the small ones stay visible. '
                 'The count itself is printed above each bar.</text>'
                 % (pad, height - 12))
    parts.append("</svg>")
    return "\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exports", default="eval/exports",
                    help="directory of npz exports to survey")
    ap.add_argument("--out-dir", default=None,
                    help="where to write liveness.json and liveness.svg")
    ap.add_argument("--train-csv", default=None,
                    help="a training summary, to look for a loss plateau")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when any export is dead")
    a = ap.parse_args(argv)

    if not os.path.isdir(a.exports):
        print("no such directory: %s" % a.exports)
        return 2

    rows = survey(a.exports)
    dead = [r for r in rows if r["dead"]]
    unreadable = [r for r in rows if r.get("error")]

    print("%d export(s) in %s" % (len(rows), a.exports))
    for row in rows:
        if row.get("error"):
            print("  %-52s unreadable: %s" % (row["name"], row["error"]))
            continue
        print("  %-52s frames=%-6d bits=%-5d distinct=%-6d%s"
              % (row["name"], row["frames"], row["bits"], row["distinct"],
                 "   DEAD" if row["dead"] else ""))
    print("\n%d of %d dead, %d unreadable"
          % (len(dead), len(rows), len(unreadable)))

    report = {"exports": a.exports, "total": len(rows), "dead": len(dead),
              "unreadable": len(unreadable), "rows": rows}

    if a.train_csv:
        train = read_train_csv(a.train_csv)
        shapes = plateau_shapes(train)
        report["plateau_shapes"] = sorted("U%d P%d" % s for s in shapes)
        if shapes:
            values = [r["best_val"] for r in train if (r["u"], r["p"]) in shapes]
            report["plateau_value"] = sum(values) / len(values)
            print("\nloss plateau at %.4f over %d shape(s): %s"
                  % (report["plateau_value"], len(shapes),
                     ", ".join(report["plateau_shapes"])))
            print("  A shared loss across different shapes is one degenerate "
                  "solution, not a coincidence. Check these against the dead "
                  "list above.")

    if a.out_dir:
        if not os.path.isdir(a.out_dir):
            os.makedirs(a.out_dir)
        with open(os.path.join(a.out_dir, "liveness.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        figure = os.path.join(a.out_dir, "liveness.svg")
        with open(figure, "w") as handle:
            handle.write(render_svg(rows))
        print("wrote %s and %s"
              % (os.path.join(a.out_dir, "liveness.json"), figure))

    if a.strict and dead:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
