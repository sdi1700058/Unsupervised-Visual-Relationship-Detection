#!/usr/bin/env python3
"""M3 — is the plan physically admissible, judged without any ground truth?

**The bottleneck this removes.** Every other metric in this project needs
densely annotated in-between frames. `mse_ratio` compares the plan against the
real frames; the quantisation floor needs the real boxes; M1 needs human
relation labels. That requirement is what restricts the whole thesis to 88
screened VidVRD clips out of 800, and it is why VideoNet — 37 domains of
rule-governed skill video, and no bounding boxes at all — cannot currently be
used.

M3 asks a different question. Not *"does the plan match the real frames"* but
*"could this sequence have happened at all"*. Four ways a decoded plan betrays
itself, none of which need a reference:

===================  ======================================================
`teleport_rate`      an object moves further in one step than anything
                     observed in the training transitions
`flicker_rate`       an object appears or vanishes mid-plan, which no real
                     object does within a short window
`malformed_rate`     `x2 < x1` or `y2 < y1`: not a box
`offcanvas_rate`     the box leaves the frame it was rasterised onto
===================  ======================================================

`validity` is the fraction of object-steps that trip none of them.

**The thresholds are measured, never chosen.** `motion_model` reads the
displacement distribution off the export's own consecutive real frames and
takes a high percentile. A constant would only encode a guess about how fast
dogs run.

The control that decides whether this is worth reporting
--------------------------------------------------------

A validity score means nothing on its own — a metric that calls everything
admissible is useless. `discrimination` scores a real trajectory against a
**scrambled** one drawn from the same frames. The separation between them is
the measure's resolving power, and it belongs beside every validity number.

If a scrambled trajectory scores as well as a real one, M3 is not measuring
anything on that data and should be reported as silent, not as a pass.

What a low score does and does not mean
---------------------------------------

A low validity is a claim about the **decoded plan**, so it can come from the
planner, from the latent, or from the decoder. It does not by itself say the
representation is bad. Its value is that it is available where nothing else
is, and that a plan failing it is definitely wrong regardless of what the real
frames held.

numpy and the standard library only, so it runs on Sherlock's Python 3.6.

    python3 tools/planner/plan_validity.py eval/exports/oracle-150010.npz
    python3 tools/planner/plan_validity.py eval/exports/*.npz --out-dir eval/validity
"""

import argparse
import json
import os
import sys

import numpy as np


def _present(boxes):
    """All-zero is the absence sentinel throughout this codebase (V30)."""
    return np.abs(boxes).sum(axis=-1) > 0


def motion_model(boxes, percentile=95.0):
    """Learn what a plausible one-step displacement looks like, from data.

    Only steps where the object is present in **both** frames count. A
    present-to-absent step is not a displacement, and letting it in would
    inflate the bound until nothing could ever be called a teleport — the same
    absence bug that has now been found eight times in this codebase.
    """
    b = np.asarray(boxes, dtype=np.float64)
    if b.ndim == 2:
        b = b[:, None, :]
    if len(b) < 2:
        return {"max_step": None, "max_area_ratio": None, "n_steps": 0}

    present = _present(b)
    both = present[:-1] & present[1:]

    centres = np.stack([(b[..., 0] + b[..., 2]) / 2.0,
                        (b[..., 1] + b[..., 3]) / 2.0], axis=-1)
    disp = np.sqrt(((centres[1:] - centres[:-1]) ** 2).sum(axis=-1))
    disp = disp[both]

    areas = (np.clip(b[..., 2] - b[..., 0], 0, None)
             * np.clip(b[..., 3] - b[..., 1], 0, None))
    prev, nxt = areas[:-1], areas[1:]
    ok = both & (prev > 1.0) & (nxt > 1.0)
    ratio = np.maximum(nxt[ok] / prev[ok], prev[ok] / nxt[ok]) if ok.any() \
        else np.array([])

    moving = disp[disp > 1e-6]
    return {
        # A window where nothing moved gives no bound at all. None, not zero:
        # zero would call every subsequent step a teleport (V29).
        "max_step": float(np.percentile(moving, percentile)) if len(moving) else None,
        "max_area_ratio": float(np.percentile(ratio, percentile)) if len(ratio) else None,
        "n_steps": int(both.sum()),
        "median_step": float(np.median(moving)) if len(moving) else None,
    }


def plan_validity(trace_boxes, model, width=None, height=None, slack=1.0):
    """Fraction of object-steps in a decoded plan that betray nothing.

    Takes **no ground truth**, by design and by test.

    `slack` multiplies the learned displacement bound.

    The default was 99th percentile with slack 1.5 until 2026-08-30. That is
    too loose, and the cost was measured rather than argued: on 13
    Something-Else clips it left the measure SILENT on 8 of them, because
    hand-object manipulation has a heavy-tailed step distribution -- median
    step 2.50 px against a 99th-percentile bound of 16.23, a ratio of 6.5,
    where VidVRD sits at 3.6. A scrambled trajectory has that much room to
    hide under the bound.

    95th percentile with slack 1.0 was chosen by sweeping both corpora, and
    the thing that stops it going tighter is that a REAL trajectory must keep
    scoring high validity:

        pct  slack   VidVRD real / sep      Something-Else real / sep
        99   1.5     1.000 / 0.147          1.000 / 0.000
        95   1.0     1.000 / 0.269          1.000 / 0.083   <- default
        90   1.0     0.990 / 0.288          1.000 / 0.095
        75   1.0     0.922 / 0.378 FLAGGED  0.974 / 0.154

    Separation alone would drive the threshold to zero, so it is not the
    criterion. `TestBoundCalibration` pins both sides.
    """
    b = np.asarray(trace_boxes, dtype=np.float64)
    if b.ndim == 2:
        b = b[:, None, :]
    n_frames, n_objs = b.shape[0], b.shape[1]

    present = _present(b)
    malformed = ((b[..., 2] < b[..., 0]) | (b[..., 3] < b[..., 1])) & present

    if width is not None and height is not None:
        off = present & ((b[..., 0] < -1.0) | (b[..., 1] < -1.0)
                         | (b[..., 2] > width + 1.0) | (b[..., 3] > height + 1.0))
    else:
        off = np.zeros_like(present)

    if n_frames < 2:
        teleport = np.zeros((0, n_objs), dtype=bool)
        flicker = np.zeros((0, n_objs), dtype=bool)
    else:
        centres = np.stack([(b[..., 0] + b[..., 2]) / 2.0,
                            (b[..., 1] + b[..., 3]) / 2.0], axis=-1)
        disp = np.sqrt(((centres[1:] - centres[:-1]) ** 2).sum(axis=-1))
        both = present[:-1] & present[1:]

        bound = model.get("max_step")
        if bound is None:
            teleport = np.zeros_like(both)
        else:
            teleport = both & (disp > bound * slack)

        # An object that is in one frame and gone from the next, or the
        # reverse, inside a short plan. Real objects do leave scenes, so this
        # is evidence rather than proof -- which is why it is reported as its
        # own rate and not folded silently into the total.
        flicker = present[:-1] != present[1:]

    n_steps = max(1, (n_frames - 1) * n_objs)
    n_cells = max(1, n_frames * n_objs)

    bad_step = teleport | flicker
    bad_cell = malformed | off
    # A step is admissible when neither its own step checks nor the cell
    # checks at either end of it fire.
    if n_frames >= 2:
        step_ok = ~(bad_step | bad_cell[:-1] | bad_cell[1:])
        validity = float(step_ok.sum()) / float(step_ok.size)
    else:
        validity = float((~bad_cell).sum()) / float(n_cells)

    return {
        "validity": validity,
        "teleport_rate": float(teleport.sum()) / n_steps,
        "flicker_rate": float(flicker.sum()) / n_steps,
        "malformed_rate": float(malformed.sum()) / n_cells,
        "offcanvas_rate": float(off.sum()) / n_cells,
        "n_frames": int(n_frames),
        "n_objects": int(n_objs),
        "bound_used": model.get("max_step"),
    }


def discrimination(real_trace, scrambled_trace, model, **kw):
    """Can the measure tell a real trajectory from a scrambled one?

    Reported beside every validity score. A measure that calls both admissible
    is silent on that data, and silence must not be read as a pass.
    """
    r = plan_validity(real_trace, model, **kw)
    s = plan_validity(scrambled_trace, model, **kw)
    return {
        "real": r["validity"],
        "scrambled": s["validity"],
        "separation": r["validity"] - s["validity"],
    }


def score_export(path, test_frac=0.3, slack=1.0, width=None, height=None,
                 seed=0, percentile=95.0):
    """Fit the motion model on the early frames, score the late ones."""
    d = np.load(path)
    key = "decoded_boxes" if "decoded_boxes" in d else "gt_boxes"
    if key not in d:
        raise SystemExit("%s has neither decoded_boxes nor gt_boxes" % path)
    boxes = np.asarray(d[key], dtype=np.float64)

    cut = int(round(len(boxes) * (1.0 - test_frac)))
    if cut < 2 or len(boxes) - cut < 2:
        raise SystemExit("%s is too short to split" % path)

    model = motion_model(boxes[:cut], percentile=percentile)
    real = boxes[cut:]
    rng = np.random.RandomState(seed)
    scrambled = real[rng.permutation(len(real))]

    out = plan_validity(real, model, width=width, height=height, slack=slack)
    out.update({"source": key, "n_train_frames": cut})
    out["discrimination"] = discrimination(real, scrambled, model,
                                           width=width, height=height,
                                           slack=slack)
    out["motion_model"] = model
    return out


def verdict(result):
    """Decided in advance, and silence is reported as silence."""
    sep = result["discrimination"]["separation"]
    v = result["validity"]
    if result["bound_used"] is None:
        return ("SILENT: nothing moved in the training frames, so no "
                "displacement bound could be learned.")
    if sep < 0.05:
        return ("SILENT: a scrambled trajectory scores %.3f against the real "
                "%.3f, a separation of %.3f. On this data the measure cannot "
                "tell them apart, so its validity score is not evidence."
                % (result["discrimination"]["scrambled"], v, sep))
    if v >= 0.95:
        return ("ADMISSIBLE: %.1f%% of object-steps betray nothing, and the "
                "measure separates real from scrambled by %.3f."
                % (100.0 * v, sep))
    if v >= 0.7:
        return ("PARTLY ADMISSIBLE: %.1f%% of object-steps are clean; the "
                "rest break physics the training frames never broke."
                % (100.0 * v))
    return ("NOT ADMISSIBLE: only %.1f%% of object-steps are clean. This "
            "sequence could not have happened." % (100.0 * v))


def _svg(results, path, width=760, row_h=26):
    """One bar per export: validity, with the scrambled control behind it."""
    if not results:
        return
    rows = results[:20]
    h = 46 + len(rows) * row_h
    left, span = 250, float(width - 250 - 70)
    out = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" '
           'font-family="sans-serif">' % (width, h)]
    out.append('<text x="8" y="18" font-size="13" fill="#444">plan validity '
               '(blue) against a scrambled trajectory (grey)</text>')
    for i, (name, r) in enumerate(rows):
        y = 34 + i * row_h
        label = name if len(name) <= 34 else name[:33] + "…"
        out.append('<text x="8" y="%d" font-size="11" fill="#333">%s</text>'
                   % (y + 14, label.replace("&", "&amp;").replace("<", "&lt;")))
        out.append('<rect x="%d" y="%d" width="%.1f" height="18" fill="#c9ced8"/>'
                   % (left, y, span * r["discrimination"]["scrambled"]))
        out.append('<rect x="%d" y="%d" width="%.1f" height="12" fill="#1f6feb" '
                   'opacity="0.85"/>' % (left, y + 3, span * r["validity"]))
        out.append('<text x="%.1f" y="%d" font-size="10" fill="#666">%.2f</text>'
                   % (left + span + 6, y + 14, r["validity"]))
    out.append('</svg>')
    with open(path, "w") as f:
        f.write("".join(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("exports", nargs="+", help="planner export npz files")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--slack", type=float, default=1.0)
    ap.add_argument("--percentile", type=float, default=95.0,
                    help="displacement percentile for the bound")
    ap.add_argument("--canvas", default="300x200",
                    help="WxH the boxes were rasterised onto, or 'none'")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)

    if a.canvas.lower() == "none":
        w = h = None
    else:
        w, h = (int(v) for v in a.canvas.lower().split("x"))

    results = []
    for p in a.exports:
        name = os.path.basename(p)
        if name.endswith(".npz"):
            name = name[:-4]
        try:
            r = score_export(p, a.test_frac, a.slack, w, h,
                             percentile=a.percentile)
        except SystemExit as e:
            print("skip %s: %s" % (name, e))
            continue
        results.append((name, r))
        print("%-42s validity %.3f  scrambled %.3f  sep %+.3f  "
              "teleport %.3f  flicker %.3f"
              % (name, r["validity"], r["discrimination"]["scrambled"],
                 r["discrimination"]["separation"],
                 r["teleport_rate"], r["flicker_rate"]))

    if not results:
        raise SystemExit("nothing scored")

    print("")
    if len(results) == 1:
        print(verdict(results[0][1]))
    else:
        # A corpus verdict, not the first row's. Printing one clip's verdict
        # over a table of twenty reads as a statement about all of them.
        seps = [r["discrimination"]["separation"] for _, r in results]
        vals = [r["validity"] for _, r in results]
        loud = [s for s in seps if s >= 0.05]
        print("%d of %d clips are informative (separation >= 0.05); the rest "
              "are SILENT and their validity is not evidence."
              % (len(loud), len(results)))
        print("median separation %.3f, median validity %.3f"
              % (float(np.median(seps)), float(np.median(vals))))
        if loud:
            informative = [v for v, s in zip(vals, seps) if s >= 0.05]
            print("across the informative clips only, median validity %.3f"
                  % float(np.median(informative)))
        else:
            print("NO clip is informative: on this data the measure is silent "
                  "everywhere and reports nothing about the plans.")

    if a.out_dir:
        if not os.path.isdir(a.out_dir):
            os.makedirs(a.out_dir)
        with open(os.path.join(a.out_dir, "validity.json"), "w") as f:
            json.dump({n: r for n, r in results}, f, indent=2)
        _svg(results, os.path.join(a.out_dir, "validity.svg"))
        print("\nwrote %s/validity.json and validity.svg" % a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
