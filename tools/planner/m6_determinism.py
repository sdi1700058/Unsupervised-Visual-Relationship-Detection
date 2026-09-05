#!/usr/bin/env python3
"""M6 - does the same action have the same effect wherever it applies?

**The property under test is what makes a transition writable as STRIPS.** A
STRIPS operator says: in any state where the precondition holds, these
propositions become true and these become false. The quantifier is the point.
It is what lets a planner chain operators it never saw chained, and it is the
only reason search works at all. The Cube-Space AutoEncoder (Asai and Muise
2020, entry `A6` in `notes/docs/RELATED_WORK.md`) builds that quantifier into
the architecture: an action becomes a constant vector added in latent space.

FOSAE has no such prior, so `tools/planner/pddl/planner.py` mines operators
after the fact. It reduces every observed transition to the bits that turn on
and the bits that turn off, calls each distinct pair one operator, and writes
the weakest precondition the effect implies - every add bit off, every delete
bit on. Nothing in training makes that operator true anywhere except at the
transition it came from.

This module measures how far from true it is, using the planner's own operator
definition, so the number describes the action model the planner reads::

    .venv-local/bin/python tools/planner/m6_determinism.py \\
        eval/exports/H14-U40_A2_P20-150010.npz --out-dir eval/m6

    .venv-local/bin/python tools/planner/m6_determinism.py \\
        "oracle VidOR=a.npz,b.npz" other.npz --out-dir eval/m6

**The metric has one obvious way to lie, and it is guarded here rather than in
the reader.** If every transition has its own effect, then every operator's
precondition holds only in the state that produced it, no operator can ever be
contradicted, and the pooled rate is exactly 1.0 - for a corpus that memorised
itself and generalises to nothing. So the headline rate counts only operators
whose precondition holds in more than one state, the naive rate that includes
the rest is reported beside it under its own name, and `verdict` checks effect
reuse and the testable share **before** it grades anything.

numpy and the standard library only, so it runs on the cluster's Python 3.6.
"""

import argparse
import json
import math
import os
import sys
from collections import OrderedDict

import numpy as np


# ---------------------------------------------------------------------------
# Constants, pre-registered in experiments/M6_effect_determinism/README.md.
# Moving one after reading the data turns the measurement into a search for a
# story, so each is named here and referred to by name below.
# ---------------------------------------------------------------------------

# Hamming distance in bits between the predicted successor and the recorded
# one. Zero is the STRIPS reading: the two words must be the same.
TOLERANCE = 0

# Transitions per distinct effect. Below this the operator set is a lookup
# table of the corpus and the rate says nothing about lawfulness.
MIN_REUSE = 2.0

# Share of operators whose precondition holds in more than one state. Only
# those could have been contradicted.
MIN_TESTABLE_SHARE = 0.10

# How far the measured rate must beat the shuffled control.
CONTROL_MARGIN = 0.10

# Grades for the headline rate.
DETERMINISTIC = 0.90
NON_DETERMINISTIC = 0.50


# ---------------------------------------------------------------------------
# Reading a corpus
# ---------------------------------------------------------------------------

def clips_from_frame_ids(frame_ids):
    """The clip each state belongs to. A frame id reads `clip/frame`.

    Written as `clip/frame` by `tools/planner/export_latents.py`. A state whose
    id carries no separator becomes its own clip name, which keeps a malformed
    id from silently joining two clips into one trajectory.
    """
    out = []
    for fid in frame_ids:
        text = str(fid)
        out.append(text.rsplit("/", 1)[0] if "/" in text else text)
    return out


def load_pool(paths):
    """Concatenate several exports into one corpus, keeping the clips apart.

    Returns `(latents, clips)`. Every clip name is prefixed with the file it
    came from, so the join between two files can never read as a transition
    even when the two files name their clips the same way.
    """
    latents, clips = [], []
    for path in paths:
        # allow_pickle stays off: these are exports this project baked, and
        # frame_ids are stored as a plain string array.
        data = np.load(path)
        if "latents" not in data.files:
            raise SystemExit(
                "%s holds %s; a planner export must carry `latents`"
                % (path, sorted(data.files)))
        z = np.asarray(data["latents"])
        stem = os.path.basename(path)
        if stem.endswith(".npz"):
            stem = stem[:-4]
        if "frame_ids" in data.files:
            names = clips_from_frame_ids(data["frame_ids"])
        else:
            names = [""] * len(z)
        latents.append(z)
        clips += ["%s|%s" % (stem, n) for n in names]
    if not latents:
        raise SystemExit("no export to read")
    return np.concatenate(latents, axis=0).astype(np.int8), clips


def parse_row(argument):
    """One command-line row: `path` or `LABEL=path[,path...]`.

    The pooled form exists because a corpus-level number has to pool the clips
    of a corpus. An oracle export holds one clip, so VidOR would otherwise
    arrive as a table of single-clip rows and no number for the corpus.
    """
    if "=" in argument:
        label, rest = argument.split("=", 1)
        files = [p for p in rest.split(",") if p]
        return label.strip(), files
    stem = os.path.basename(argument)
    if stem.endswith(".npz"):
        stem = stem[:-4]
    return stem, [argument]


def transitions(latents, clips=None):
    """Consecutive pairs inside one clip.

    Returns `(pre, suc, dropped)`. A pair that steps over a clip boundary
    records a cut and not a motion, so it is dropped and counted. The export
    concatenates 88 clips, so this is not a rare case.
    """
    z = np.asarray(latents, dtype=np.int8)
    if len(z) < 2:
        empty = np.zeros((0, z.shape[1] if z.ndim > 1 else 0), dtype=np.int8)
        return empty, empty, 0
    if clips is None:
        keep = np.ones(len(z) - 1, dtype=bool)
    else:
        names = list(clips)
        keep = np.asarray([names[i] == names[i + 1]
                           for i in range(len(z) - 1)], dtype=bool)
    dropped = int((~keep).sum())
    idx = np.flatnonzero(keep)
    return z[idx], z[idx + 1], dropped


# ---------------------------------------------------------------------------
# Effects and operators
# ---------------------------------------------------------------------------

def group_effects(pre, suc):
    """The distinct (add, delete) effects, with the transitions behind each.

    Two transitions that flip the same bits the same way are one operator, and
    the no-op is dropped, both exactly as `distinct_effects` in
    `tools/planner/pddl/planner.py` does it. The order is first occurrence, so
    the operator numbering matches the domain file the planner writes.

    A no-op is still a transition. It is dropped from the operator set and it
    stays in the determinism scan, where it can contradict an operator whose
    precondition holds in its source state.
    """
    pre = np.asarray(pre, dtype=np.int8)
    suc = np.asarray(suc, dtype=np.int8)
    add = suc > pre
    dele = pre > suc
    if not len(pre):
        return []

    packed = np.packbits(np.concatenate([add, dele], axis=1), axis=1)
    moved = add.any(axis=1) | dele.any(axis=1)

    groups = OrderedDict()
    for t in np.flatnonzero(moved):
        key = packed[t].tobytes()
        group = groups.get(key)
        if group is None:
            groups[key] = {"add": np.flatnonzero(add[t]).tolist(),
                           "del": np.flatnonzero(dele[t]).tolist(),
                           "members": [int(t)]}
        else:
            group["members"].append(int(t))
    out = list(groups.values())
    for group in out:
        group["support"] = len(group["members"])
    return out


def applicable_mask(pre, add, dele):
    """Where the weakest precondition of an effect holds.

    Every add bit off and every delete bit on, which is the precondition
    `tools/planner/pddl/planner.py` writes. An operator with neither add nor
    delete bits would apply everywhere, but it is never built: the no-op is
    dropped from the operator set.
    """
    pre = np.asarray(pre, dtype=np.int8)
    mask = np.ones(len(pre), dtype=bool)
    if dele:
        mask &= (pre[:, np.asarray(dele, dtype=np.int64)] == 1).all(axis=1)
    if add:
        mask &= (pre[:, np.asarray(add, dtype=np.int64)] == 0).all(axis=1)
    return mask


def precondition_agreement(pre, members, add, dele):
    """How much the states one operator fires from agree with each other.

    `agreement` is the share of latent bits that hold the same value in every
    source state of the group. It reads in one direction only when it is put
    beside `extra_bits`, the count of those constant bits that the mined
    precondition does **not** name:

    * agreement near 1 with many extra bits: the operator fires from states
      that are nearly the same word, so it is a memorised pair wearing an
      operator's clothes;
    * agreement near the size of the effect: the operator fires from states
      that differ everywhere except where they must, which is what a STRIPS
      operator does.

    **A group of one always agrees with itself**, so the aggregate in `measure`
    is taken over groups of two or more. Averaging the singletons in made the
    number report the sparsity of the latent code: on a one-hot oracle where
    96% of operators fire once, the mean read 0.999 whatever the groups did.

    The row also carries `corpus_agreement`, the same fraction over every
    source state in the corpus. A sparse code drives both numbers up together,
    and the distance between them is the part that belongs to the grouping.
    """
    pre = np.asarray(pre, dtype=np.int8)
    block = pre[np.asarray(members, dtype=np.int64)]
    constant = (block == block[0]).all(axis=0)
    n_constant = int(constant.sum())
    named = len(set(add) | set(dele))
    return {"agreement": float(n_constant) / pre.shape[1],
            "constant_bits": n_constant,
            "extra_bits": n_constant - named}


def concentration(supports):
    """How heavily the transitions concentrate on a few effects.

    An export where every transition has a unique effect and an export with ten
    operators used everywhere can report the same determinism, so the shape of
    this distribution has to be reported next to it. `effective_effects` is
    `exp(H)` over the support distribution: the number of equally used
    operators that would give the same entropy.
    """
    supports = [int(s) for s in supports]
    n = len(supports)
    total = sum(supports)
    if not n or not total:
        return {"n_effects": n, "n_with_effect": total, "reuse": None,
                "singleton_share": None, "top1_share": None,
                "top10_share": None, "effective_effects": None}
    order = sorted(supports, reverse=True)
    p = np.asarray(order, dtype=np.float64) / total
    entropy = float(-(p * np.log(p)).sum())
    return {
        "n_effects": n,
        "n_with_effect": total,
        "reuse": total / float(n),
        "singleton_share": sum(1 for s in supports if s == 1) / float(n),
        "top1_share": order[0] / float(total),
        "top10_share": sum(order[:10]) / float(total),
        "effective_effects": math.exp(entropy),
    }


def synthetic_cube_corpus(n_ops=6, n_payload=26, n_clips=20, steps=18, seed=0):
    """A corpus that has the Cube-Space property, as a positive control.

    The shuffled control shows what this measurement gives a corpus with no
    transition structure. It does not show that the measurement can return a
    high rate at all, and a metric that is negative on everything is worth
    nothing. This builds the other end.

    A state is a one-hot mode over `n_ops` bits followed by a payload block
    that is random per clip and never changes. Operator `k` deletes mode `k`
    and adds mode `k+1`, so its effect is a constant two-bit vector, its
    precondition holds in exactly the states whose mode is `k`, and every one
    of those states takes it. The payload differs from clip to clip, so the
    operator fires from many states that agree on nothing but the mode - which
    is what a STRIPS operator does and what none of the real rows do.

    Returns `(latents, clips)`.
    """
    rng = np.random.RandomState(seed)
    width = n_ops + n_payload
    rows, clips = [], []
    for c in range(n_clips):
        payload = rng.randint(0, 2, size=n_payload).astype(np.int8)
        for t in range(steps):
            state = np.zeros(width, dtype=np.int8)
            state[t % n_ops] = 1
            state[n_ops:] = payload
            rows.append(state)
            clips.append("synthetic%d" % c)
    return np.stack(rows), clips


def permute_successors(suc, seed=0):
    """The control: keep every successor, break which state it followed.

    The state distribution survives and the transition structure does not, so
    whatever the control scores is what this measurement gives a corpus that
    learned nothing. On the naive rate the control usually scores near 1.0,
    which is the degenerate reading made visible rather than argued about.
    """
    suc = np.asarray(suc, dtype=np.int8)
    rng = np.random.RandomState(seed)
    return suc[rng.permutation(len(suc))]


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def measure(pre, suc, tolerance=TOLERANCE, max_effects=0):
    """Determinism of the mined operators over one corpus.

    For each operator, every transition whose source satisfies the operator's
    precondition is a test of it: the operator predicts `(s or add) and not
    delete`, and the corpus recorded something. `agree` counts the tests the
    corpus passes.

    **The headline `determinism` counts only testable operators**, those whose
    precondition holds in more than one source state. An operator whose
    precondition holds in exactly one state is satisfied only where it fired,
    was never given the chance to disagree, and would contribute a free 1.0.
    `determinism_naive` includes those, and is reported so that the difference
    between the two is visible rather than hidden.
    """
    pre = np.asarray(pre, dtype=np.int8)
    suc = np.asarray(suc, dtype=np.int8)
    n_transitions = len(pre)
    if not n_transitions:
        raise SystemExit("the corpus holds no transition inside a clip")

    flips = pre != suc
    n_noop = int((~flips.any(axis=1)).sum())

    groups = group_effects(pre, suc)
    supports = [g["support"] for g in groups]
    conc = concentration(supports)

    scanned = groups
    if max_effects and len(groups) > max_effects:
        order = sorted(range(len(groups)),
                       key=lambda i: -groups[i]["support"])[:max_effects]
        scanned = [groups[i] for i in sorted(order)]

    # A transition agrees with an operator exactly when the two flip the same
    # bits: both successors are the source with a bit set flipped, so their
    # Hamming distance is the size of the symmetric difference of those sets.
    # At zero tolerance that reduces to identity of the flip pattern, which is
    # a single integer comparison instead of a pass over the latent width.
    flip_packed = np.packbits(flips, axis=1)
    flip_id = np.zeros(n_transitions, dtype=np.int64)
    seen = {}
    for t in range(n_transitions):
        key = flip_packed[t].tobytes()
        if key not in seen:
            seen[key] = len(seen)
        flip_id[t] = seen[key]

    applicable_total = agree_total = 0
    applicable_testable = agree_testable = 0
    n_testable = 0
    rates = []
    weighted_size = 0.0
    weight = 0
    # Precondition agreement is aggregated over groups of two or more alone. A
    # group of one agrees with itself on every bit, so including the
    # singletons would report the sparsity of the latent code instead.
    multi_agreement = multi_extra = 0.0
    multi_weight = n_multi = 0
    per_effect = []

    for group in scanned:
        add, dele = group["add"], group["del"]
        mask = applicable_mask(pre, add, dele)
        n_applicable = int(mask.sum())
        rows = np.flatnonzero(mask)

        if tolerance == 0:
            target = flip_id[group["members"][0]]
            n_agree = int((flip_id[rows] == target).sum())
        else:
            target_flip = flips[group["members"][0]]
            distance = (flips[rows] != target_flip).sum(axis=1)
            n_agree = int((distance <= tolerance).sum())

        applicable_total += n_applicable
        agree_total += n_agree
        testable = n_applicable > 1
        if testable:
            n_testable += 1
            applicable_testable += n_applicable
            agree_testable += n_agree
            rates.append(n_agree / float(n_applicable))

        pa = precondition_agreement(pre, group["members"], add, dele)
        support = group["support"]
        weight += support
        weighted_size += support * len(set(add) | set(dele))
        if support > 1:
            n_multi += 1
            multi_weight += support
            multi_agreement += support * pa["agreement"]
            multi_extra += support * pa["extra_bits"]

        per_effect.append({"support": support, "applicable": n_applicable,
                           "agree": n_agree, "testable": testable,
                           "size": len(set(add) | set(dele)),
                           "agreement": pa["agreement"],
                           "extra_bits": pa["extra_bits"]})

    out = {
        "n_transitions": n_transitions,
        "n_bits": int(pre.shape[1]),
        "n_noop": n_noop,
        "tolerance": int(tolerance),
        "n_effects_scanned": len(scanned),
        "applicable_total": applicable_total,
        "agree_total": agree_total,
        "determinism_naive": (agree_total / float(applicable_total)
                              if applicable_total else None),
        "applicable_testable": applicable_testable,
        "agree_testable": agree_testable,
        "determinism": (agree_testable / float(applicable_testable)
                        if applicable_testable else None),
        "determinism_macro": float(np.mean(rates)) if rates else None,
        "n_testable": n_testable,
        "testable_share": (n_testable / float(len(scanned))
                           if scanned else 0.0),
        "mean_effect_size": (weighted_size / weight) if weight else None,
        "n_multi_effects": n_multi,
        "mean_precondition_agreement": ((multi_agreement / multi_weight)
                                        if multi_weight else None),
        "mean_extra_constant_bits": ((multi_extra / multi_weight)
                                     if multi_weight else None),
        # The same fraction over every source state in the corpus. A sparse
        # code drives this up on its own, so only the distance between the two
        # belongs to the grouping.
        "corpus_agreement": float((pre == pre[0]).all(axis=0).sum())
                            / pre.shape[1],
    }
    out.update(conc)
    out["_per_effect"] = per_effect
    return out


def analyse(latents, clips=None, tolerance=TOLERANCE, seed=0, label=None,
            max_effects=0, files=None):
    """One table row: the measurement, its control, and its verdict."""
    latents = np.asarray(latents, dtype=np.int8)
    pre, suc, dropped = transitions(latents, clips)
    row = measure(pre, suc, tolerance=tolerance, max_effects=max_effects)
    row["label"] = label
    row["files"] = list(files) if files else []
    row["n_states"] = int(len(latents))
    row["n_distinct_states"] = int(len(np.unique(latents, axis=0)))
    row["all_zero"] = bool(not latents.any())
    row["n_cross_clip_dropped"] = dropped
    row["seed"] = seed

    control = measure(pre, permute_successors(suc, seed=seed),
                      tolerance=tolerance, max_effects=max_effects)
    row["control"] = {k: control[k] for k in
                      ("n_effects", "reuse", "determinism",
                       "determinism_naive", "testable_share", "n_testable",
                       "n_noop")}
    row["verdict"] = verdict(row)
    return row


# ---------------------------------------------------------------------------
# The reading, decided in advance
# ---------------------------------------------------------------------------

def verdict(row):
    """The reading. The two vacuity checks come first, on purpose.

    Grading the rate first would report the worst possible corpus - one that
    memorised every transition - as the best possible result, because that
    corpus scores 1.0. The order of these branches is the guard.
    """
    reuse = row.get("reuse")
    share = row.get("testable_share")
    rate = row.get("determinism")
    naive = row.get("determinism_naive")
    control = row.get("control") or {}

    # Checked before anything else. A collapsed export has no transition at
    # all, so every rate below is 0/0, and a measurement that divided by the
    # transition count would report the emptiest possible export as the most
    # lawful one. Four exports on disk are in exactly this state.
    distinct = row.get("n_distinct_states")
    if distinct is not None and distinct <= 1:
        return ("DEAD EXPORT: the latent holds %d distinct state over %s "
                "frames%s. The encoder collapsed, so there is no transition "
                "to measure and this row carries no evidence about "
                "determinism."
                % (distinct, row.get("n_states"),
                   ", every bit zero" if row.get("all_zero") else ""))

    if reuse is None:
        return ("NO TRANSITION carries an effect. Every pair in this corpus "
                "repeats its own state, so there is no action model to test.")

    if reuse < MIN_REUSE:
        return ("DEGENERATE: %.2f transitions per distinct effect, under the "
                "%.1f this measurement needs. Nearly every transition is its "
                "own operator, so the action model is a lookup table of the "
                "corpus. The naive rate of %s is what that lookup table scores "
                "and it is not evidence of anything."
                % (reuse, MIN_REUSE,
                   "n/a" if naive is None else "%.3f" % naive))

    if share is None or share < MIN_TESTABLE_SHARE:
        return ("VACUOUS: only %s of operators have a precondition that holds "
                "in more than one state, under the %.2f this measurement "
                "needs. Almost no operator could ever be contradicted, so the "
                "rate is built from free agreements."
                % ("none" if share is None else "%.1f%%" % (100.0 * share),
                   MIN_TESTABLE_SHARE))

    if rate is None:
        return ("VACUOUS: no operator has a precondition that holds in more "
                "than one state.")

    control_rate = control.get("determinism")
    prefix = ""
    if control_rate is None:
        prefix = ("The shuffled control has no contradictable operator, so it "
                  "cannot be compared. ")
    elif rate - control_rate < CONTROL_MARGIN:
        return ("NO SIGNAL AGAINST THE CONTROL: %.3f measured against %.3f "
                "with the successors shuffled, a margin under %.2f. Whatever "
                "lawfulness this shows belongs to the shape of the state set "
                "and not to anything the model learned."
                % (rate, control_rate, CONTROL_MARGIN))

    if rate >= DETERMINISTIC:
        return (prefix + "LAWFUL: %.3f of the tests agree. The same operator "
                "does the same thing wherever its precondition holds, so this "
                "transition structure is writable as STRIPS." % rate)
    if rate <= NON_DETERMINISTIC:
        return (prefix + "NOT DETERMINISTIC: %.3f of the tests agree. The same "
                "operator produces different successors in different states, "
                "so the symbolic model is a fiction whatever the "
                "reconstruction loss says. This is the A6 argument measured "
                "inside this project's own data." % rate)
    return (prefix + "PARTLY LAWFUL: %.3f of the tests agree, between %.2f and "
            "%.2f. Neither reading is available; report the rate and the "
            "spread." % (rate, NON_DETERMINISTIC, DETERMINISTIC))


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------

def _esc(text):
    """XML-safe text.

    A raw `<` or `>` in SVG text is invalid XML, and this project has shipped
    an unopenable figure that way three times.
    """
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _bar(x, y, width, height, value, colour, cap=1.0):
    span = 0.0 if value is None else max(0.0, min(1.0, value / cap)) * width
    return ('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s"/>'
            % (x, y, span, height, colour))


def write_svg(rows, path, title="M6 - effect determinism"):
    """Three panels per row: the rate, the reuse, and the testable share.

    The rate alone would be read as a score, and on this metric a perfect score
    is what a corpus that memorised itself produces. The other two panels are
    on the same line so that reading cannot happen.
    """
    left, row_h = 236.0, 66.0
    panel_w = 176.0
    gap = 40.0
    width = left + 3 * panel_w + 3 * gap + 90
    top = 118.0
    height = top + row_h * max(len(rows), 1) + 108

    blue, grey, red, ink = "#1f6feb", "#c9ced8", "#b3261e", "#333"
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
           'width="%d" height="%d" font-family="sans-serif">'
           % (int(width), int(height), int(width), int(height)),
           '<rect width="%d" height="%d" fill="#ffffff"/>'
           % (int(width), int(height)),
           '<text x="14" y="30" font-size="17" fill="#111">%s</text>'
           % _esc(title),
           '<text x="14" y="52" font-size="12" fill="#666">%s</text>'
           % _esc("does one mined operator produce one successor wherever "
                  "its precondition holds")]

    heads = [("determinism (testable operators)", "0 to 1, red tick = "
              "shuffled control"),
             ("effect reuse", "transitions per distinct effect, log scale"),
             ("testable share", "operators that could be contradicted")]
    for i, (name, sub) in enumerate(heads):
        x = left + i * (panel_w + gap)
        out.append('<text x="%.1f" y="86" font-size="11" fill="#111">%s</text>'
                   % (x, _esc(name)))
        out.append('<text x="%.1f" y="100" font-size="9" fill="#888">%s</text>'
                   % (x, _esc(sub)))

    for i, row in enumerate(rows):
        y = top + i * row_h
        bar_y = y + 12
        bar_h = 18.0
        label = row.get("label") or "row %d" % i
        out.append('<text x="14" y="%.1f" font-size="12" fill="%s">%s</text>'
                   % (bar_y + 13, ink, _esc(label[:34])))
        out.append('<text x="14" y="%.1f" font-size="9" fill="#888">%s</text>'
                   % (bar_y + 26,
                      _esc("%d bits, %d transitions, %d effects"
                           % (row.get("n_bits") or 0,
                              row.get("n_transitions") or 0,
                              row.get("n_effects") or 0))))

        # Panel 1, the rate.
        x = left
        out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                   'fill="#f1f3f6"/>' % (x, bar_y, panel_w, bar_h))
        rate = row.get("determinism")
        if rate is None:
            out.append('<text x="%.1f" y="%.1f" font-size="10" fill="%s">%s'
                       '</text>' % (x + 6, bar_y + 13, red,
                                    _esc("no testable operator")))
        else:
            out.append(_bar(x, bar_y, panel_w, bar_h, rate, blue))
            out.append('<text x="%.1f" y="%.1f" font-size="11" fill="#444">'
                       '%.3f</text>' % (x + panel_w + 6, bar_y + 13, rate))
        control = (row.get("control") or {}).get("determinism")
        if control is not None:
            cx = x + max(0.0, min(1.0, control)) * panel_w
            out.append('<rect x="%.1f" y="%.1f" width="2.5" height="%.1f" '
                       'fill="%s"/>' % (cx, bar_y - 3, bar_h + 6, red))

        # Panel 2, the reuse, on a log scale because it spans two decades.
        x = left + panel_w + gap
        out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                   'fill="#f1f3f6"/>' % (x, bar_y, panel_w, bar_h))
        reuse = row.get("reuse")
        if reuse:
            frac = min(1.0, math.log10(max(reuse, 1.0)) / 2.0)
            colour = red if reuse < MIN_REUSE else blue
            out.append(_bar(x, bar_y, panel_w, bar_h, frac, colour))
            out.append('<text x="%.1f" y="%.1f" font-size="11" fill="#444">'
                       '%.2f</text>' % (x + panel_w + 6, bar_y + 13, reuse))
        gate = x + math.log10(MIN_REUSE) / 2.0 * panel_w
        out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
                   'stroke="#555" stroke-width="1" stroke-dasharray="3,2"/>'
                   % (gate, bar_y - 3, gate, bar_y + bar_h + 3))

        # Panel 3, the testable share.
        x = left + 2 * (panel_w + gap)
        out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                   'fill="#f1f3f6"/>' % (x, bar_y, panel_w, bar_h))
        share = row.get("testable_share") or 0.0
        out.append(_bar(x, bar_y, panel_w, bar_h, share,
                        red if share < MIN_TESTABLE_SHARE else blue))
        out.append('<text x="%.1f" y="%.1f" font-size="11" fill="#444">'
                   '%.2f</text>' % (x + panel_w + 6, bar_y + 13, share))
        gate = x + MIN_TESTABLE_SHARE * panel_w
        out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
                   'stroke="#555" stroke-width="1" stroke-dasharray="3,2"/>'
                   % (gate, bar_y - 3, gate, bar_y + bar_h + 3))

    foot = top + row_h * max(len(rows), 1) + 22
    notes = [
        "A perfect rate is what a corpus that memorised every transition "
        "produces, so the rate is read only when reuse clears the dashed "
        "line at %.1f and the testable share clears %.2f."
        % (MIN_REUSE, MIN_TESTABLE_SHARE),
        "An operator is testable when its precondition holds in more than one "
        "state. The naive rate over all operators is in the JSON beside this "
        "figure.",
    ]
    for i, note in enumerate(notes):
        out.append('<text x="14" y="%.1f" font-size="10" fill="#666">%s</text>'
                   % (foot + i * 15, _esc(note)))
    if rows:
        # A count of the branches, not one row's verdict. The first row is
        # whichever row was named first, and quoting it would let the order of
        # the command line decide what the figure says.
        tally = {}
        for row in rows:
            head = (row.get("verdict") or "").split(":")[0].split(".")[0]
            tally[head] = tally.get(head, 0) + 1
        summary = ", ".join("%d %s" % (n, k.lower())
                            for k, n in sorted(tally.items(),
                                               key=lambda kv: -kv[1]))
        out.append('<text x="14" y="%.1f" font-size="10" fill="#111">%s</text>'
                   % (foot + len(notes) * 15 + 8,
                      _esc("%d rows: %s" % (len(rows), summary))))

    out.append('</svg>')
    with open(path, "w") as handle:
        handle.write("\n".join(out) + "\n")
    return path


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _print_row(row):
    def num(key, fmt="%.3f"):
        value = row.get(key)
        return "n/a" if value is None else fmt % value

    print("")
    print("%s" % (row.get("label") or "row"))
    print("  states / transitions      %d / %d  (%d dropped at a clip join, "
          "%d no-op)"
          % (row["n_states"], row["n_transitions"],
             row["n_cross_clip_dropped"], row["n_noop"]))
    print("  distinct states           %d" % row["n_distinct_states"])
    print("  latent width              %d bits" % row["n_bits"])
    print("  distinct effects          %d" % row["n_effects"])
    print("  effect reuse              %s transitions per effect"
          % num("reuse", "%.2f"))
    print("  singleton effects         %s" % num("singleton_share"))
    print("  top 1 / top 10 share      %s / %s"
          % (num("top1_share"), num("top10_share")))
    print("  effective effects         %s" % num("effective_effects", "%.1f"))
    print("  mean effect size          %s bits" % num("mean_effect_size",
                                                      "%.1f"))
    print("  testable operators        %d of %d  (%s)"
          % (row["n_testable"], row["n_effects"], num("testable_share")))
    print("  DETERMINISM               %s   over %d tests"
          % (num("determinism"), row["applicable_testable"]))
    print("  naive rate, all operators %s   over %d tests"
          % (num("determinism_naive"), row["applicable_total"]))
    print("  per-operator mean         %s" % num("determinism_macro"))
    print("  shuffled control          %s  (naive %s, reuse %s)"
          % ("n/a" if row["control"]["determinism"] is None
             else "%.3f" % row["control"]["determinism"],
             "n/a" if row["control"]["determinism_naive"] is None
             else "%.3f" % row["control"]["determinism_naive"],
             "n/a" if row["control"]["reuse"] is None
             else "%.2f" % row["control"]["reuse"]))
    print("  precondition agreement    %s over %d reused operators  (%s "
          "constant bits beyond the mined precondition)"
          % (num("mean_precondition_agreement"), row["n_multi_effects"],
             num("mean_extra_constant_bits", "%.1f")))
    print("  the same over the corpus  %s" % num("corpus_agreement"))
    print("")
    print("  %s" % row["verdict"])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Measure whether one mined operator produces one "
                    "successor wherever its precondition holds.")
    ap.add_argument("rows", nargs="+",
                    help="an export npz, or LABEL=path,path to pool several "
                         "exports into one corpus row")
    ap.add_argument("--tolerance", type=int, default=TOLERANCE,
                    help="Hamming bits between the predicted successor and "
                         "the recorded one. 0 is the STRIPS reading")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the shuffled control")
    ap.add_argument("--max-effects", type=int, default=0,
                    help="scan only this many operators, the most used first. "
                         "0 scans all of them. A safety valve for an export "
                         "with a very large action set; the row reports how "
                         "many were scanned")
    ap.add_argument("--out-dir", default=None,
                    help="where to write m6_determinism.json and .svg")
    ap.add_argument("--positive-control", action="store_true",
                    help="prepend a synthetic corpus built to have the "
                         "Cube-Space property. It shows that this measurement "
                         "can return a high rate, which the shuffled control "
                         "cannot show")
    a = ap.parse_args(argv)

    rows = []
    if a.positive_control:
        z, clips = synthetic_cube_corpus(seed=a.seed)
        rows.append(analyse(z, clips, tolerance=a.tolerance, seed=a.seed,
                            label="synthetic constant-effect control"))
        _print_row(rows[-1])

    for argument in a.rows:
        label, files = parse_row(argument)
        missing = [p for p in files if not os.path.isfile(p)]
        if missing:
            print("skip %s: no such export %s" % (label, ", ".join(missing)))
            continue
        latents, clips = load_pool(files)
        rows.append(analyse(latents, clips, tolerance=a.tolerance,
                            seed=a.seed, label=label,
                            max_effects=a.max_effects, files=files))
        _print_row(rows[-1])

    if not rows:
        raise SystemExit("no row could be read")

    if a.out_dir:
        if not os.path.isdir(a.out_dir):
            os.makedirs(a.out_dir)
        payload = []
        for row in rows:
            keep = dict(row)
            keep.pop("_per_effect", None)
            payload.append(keep)
        with open(os.path.join(a.out_dir, "m6_determinism.json"), "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        write_svg(rows, os.path.join(a.out_dir, "m6_determinism.svg"))
        print("\nwrote %s/m6_determinism.json and m6_determinism.svg"
              % a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
