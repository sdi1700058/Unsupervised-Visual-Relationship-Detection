#!/usr/bin/env python3
"""M2 — did FOSAE learn a rule, or did it memorise the objects?

**The claim under test is the one in the model's name.** FOSAE is a
*First-Order* State AutoEncoder, and first order means a predicate abstracts
over its arguments: `chase(x, y)` is one predicate whether *x* is a dog or a
cat. If the learned predicates only work on the object types they were trained
on, the representation is first-order in name and propositional in fact —
which would be a strong finding, not a failure.

The experiment is a **split**, not a new metric. Take M1's probe and change
only which clips it trains on:

===================  ======================================================
compositional        hold out whole object **categories**. The predicate is
                     still seen in training, on different objects.
random (control)     hold out the same *number* of clips at random.
===================  ======================================================

Both are scored the same way, so the difference isolates composition rather
than difficulty. Reading::

    compositional ~= random     the predicate transferred: a rule
    compositional <  random     it did not: object-specific memorisation
    both ~= prior               the probe never worked; say so, do not
                                report the first case

That last row is the trap. A naive difference test on two scores that are both
at the base rate reports "it generalises" when nothing was learned at all, and
that is exactly the reading a tired reader takes from a small gap.

Why the predicate must be seen in training
------------------------------------------

`compositional_split` reports `transferable_predicates`: those appearing in
training on some category **and** in test on a held-out one. A predicate that
occurs only with the held-out category is dropped, because failing it is a
novel-label result rather than a compositional one. Measured on VidVRD: 2,961
distinct (subject, predicate, object) triples over 35 categories, so there is
ample room to hold categories out and keep the predicates.

    python3 tools/planner/compositional.py eval/probe/batch/*.npz \\
        --annotations data/video/vidvrd/annotations/train \\
        --hold-out dog,cat --out-dir eval/compositional

numpy and the standard library only, so it runs on Sherlock's Python 3.6.
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from predicate_probe import (relation_labels, ridge_probe_multi,   # noqa: E402
                             average_precision, _pair_onehot)


def clip_triples(doc):
    """The (subject_category, predicate, object_category) triples in a clip."""
    cat = {o.get("tid"): o.get("category")
           for o in doc.get("subject/objects", [])}
    out = set()
    for r in doc.get("relation_instances", []):
        s, o = cat.get(r.get("subject_tid")), cat.get(r.get("object_tid"))
        if s and o:
            out.add((s, r["predicate"], o))
    return out


def compositional_split(metas, held_out_categories):
    """Split clips so held-out categories appear only in the test set.

    `metas` is a sequence of dicts with a `triples` set. Returns
    `(train_idx, test_idx, info)`, where `info["transferable_predicates"]` is
    the subset of predicates that can actually test composition: seen in
    training on some category, and in test on a held-out one.
    """
    held = set(held_out_categories)
    train, test = [], []
    for i, m in enumerate(metas):
        cats = {t[0] for t in m["triples"]} | {t[2] for t in m["triples"]}
        (test if (cats & held) else train).append(i)

    if not train or not test:
        raise ValueError(
            "holding out %s leaves %d train and %d test clips"
            % (sorted(held), len(train), len(test)))

    train_preds = {t[1] for i in train for t in metas[i]["triples"]}
    test_held_preds = {t[1] for i in test for t in metas[i]["triples"]
                       if t[0] in held or t[2] in held}

    return train, test, {
        "held_out_categories": sorted(held),
        "transferable_predicates": sorted(train_preds & test_held_preds),
        "n_train_clips": len(train),
        "n_test_clips": len(test),
    }


def matched_random_split(n_clips, n_test, seed=0):
    """A random split of the same size, so the comparison is fair."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_clips)
    test = sorted(order[:n_test].tolist())
    train = sorted(order[n_test:].tolist())
    return train, test


def _stack(clips, predicates, max_slots):
    """Feature and label matrices, plus the clip index of every row."""
    Xs, Ys, cid = [], [], []
    for c, (z, lab) in enumerate(clips):
        z = np.asarray(z, dtype=np.float64)
        keep = lab.frames < len(z)
        if not keep.any():
            continue
        frames, pairs = lab.frames[keep], lab.pairs[keep]
        y = np.zeros((int(keep.sum()), len(predicates)))
        for j, name in enumerate(lab.predicates):
            if name in predicates:
                y[:, predicates.index(name)] = lab.Y[keep][:, j]
        Xs.append(np.hstack([z[frames], _pair_onehot(pairs, max_slots)]))
        Ys.append(y)
        cid.append(np.full(int(keep.sum()), c))
    if not Xs:
        raise SystemExit("no clip had latents covering its annotated frames")
    return np.vstack(Xs), np.vstack(Ys), np.concatenate(cid)


def _score(X, Y, cid, train_clips, test_clips, predicates, keep_predicates):
    """Macro AP over the predicates that can test composition."""
    tr = np.array([i for i in range(len(X)) if cid[i] in set(train_clips)])
    te = np.array([i for i in range(len(X)) if cid[i] in set(test_clips)])
    if len(tr) == 0 or len(te) == 0:
        return None, None

    S = ridge_probe_multi(X[tr], Y[tr], X[te])
    aps, priors = [], []
    for c, name in enumerate(predicates):
        if name not in keep_predicates:
            continue
        y = Y[:, c]
        ap = average_precision(y[te], S[:, c])
        pr = average_precision(y[te], np.full(len(te), y[tr].mean()))
        if ap is not None:
            aps.append(ap)
        if pr is not None:
            priors.append(pr)
    return (float(np.mean(aps)) if aps else None,
            float(np.mean(priors)) if priors else None)


def run(clips, metas, held_out_categories, seed=0):
    """Score the compositional split against a size-matched random one."""
    train, test, info = compositional_split(metas, held_out_categories)
    keep = set(info["transferable_predicates"])
    if not keep:
        raise SystemExit(
            "no predicate is both trained on another category and tested on a "
            "held-out one; holding out %s tests novel labels, not composition"
            % sorted(held_out_categories))

    predicates = sorted({p for _, lab in clips for p in lab.predicates})
    max_slots = max(len(lab.tids) for _, lab in clips)
    X, Y, cid = _stack(clips, predicates, max_slots)

    comp_ap, comp_prior = _score(X, Y, cid, train, test, predicates, keep)
    rtr, rte = matched_random_split(len(clips), len(test), seed=seed)
    rand_ap, _ = _score(X, Y, cid, rtr, rte, predicates, keep)

    out = dict(info)
    out.update({
        "compositional_mAP": comp_ap,
        "random_mAP": rand_ap,
        "prior_mAP": comp_prior,
        "n_predicates_tested": len(keep),
        "n_clips": len(clips),
    })
    if comp_ap is not None and rand_ap is not None:
        out["drop"] = rand_ap - comp_ap
    else:
        out["drop"] = None
    return out


def verdict(r, floor=0.02):
    """Decided in advance, and the no-signal case is checked FIRST."""
    comp, rand = r.get("compositional_mAP"), r.get("random_mAP")
    prior = r.get("prior_mAP")
    if comp is None or rand is None:
        return "No predicate could be scored on both splits."

    # Checked first on purpose. If neither split beat the base rate, the two
    # scores are equal because nothing was learned, and calling that
    # "generalisation" is the worst available reading.
    if prior is not None and max(comp, rand) - prior <= floor:
        return ("NO SIGNAL: neither split beats the label prior (%.3f against "
                "a prior of %.3f). The probe learned nothing on either, so "
                "this says nothing about composition."
                % (max(comp, rand), prior))

    drop = rand - comp
    if drop <= 0.02:
        return ("The predicate TRANSFERS to unseen object categories: "
                "compositional %.3f against random %.3f. First-order in "
                "practice, not only in name." % (comp, rand))
    if drop < 0.10:
        return ("Partial transfer: compositional %.3f against random %.3f, a "
                "drop of %.3f." % (comp, rand, drop))
    return ("MEMORISATION rather than a rule: compositional %.3f against "
            "random %.3f, a drop of %.3f. The predicates are tied to the "
            "object types they were trained on." % (comp, rand, drop))


def _svg(r, path, width=620, height=190):
    comp, rand = r.get("compositional_mAP") or 0.0, r.get("random_mAP") or 0.0
    prior = r.get("prior_mAP") or 0.0
    top = max(comp, rand, prior, 0.05) * 1.25
    bars = [("random split", rand, "#c9ced8"),
            ("compositional", comp, "#1f6feb"),
            ("label prior", prior, "#b3261e")]
    out = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" '
           'font-family="sans-serif">' % (width, height)]
    out.append('<text x="8" y="18" font-size="13" fill="#444">'
               'macro average precision, held-out categories: %s</text>'
               % ", ".join(r.get("held_out_categories", []))[:60])
    left, span = 150, float(width - 150 - 70)
    for i, (name, v, col) in enumerate(bars):
        y = 40 + i * 42
        out.append('<text x="8" y="%d" font-size="12" fill="#333">%s</text>'
                   % (y + 18, name))
        out.append('<rect x="%d" y="%d" width="%.1f" height="24" fill="%s"/>'
                   % (left, y, span * (v / top), col))
        out.append('<text x="%.1f" y="%d" font-size="11" fill="#666">%.3f</text>'
                   % (left + span * (v / top) + 6, y + 17, v))
    out.append('</svg>')
    with open(path, "w") as f:
        f.write("".join(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("exports", nargs="+")
    ap.add_argument("--annotations", required=True,
                    help="directory holding <video_id>.json")
    ap.add_argument("--hold-out", required=True,
                    help="comma-separated object categories to hold out")
    ap.add_argument("--max-objects", type=int, default=3)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)

    held = {c.strip() for c in a.hold_out.split(",") if c.strip()}
    clips, metas = [], []
    for p in a.exports:
        stem = os.path.basename(p)
        if stem.endswith(".npz"):
            stem = stem[:-4]
        ann = os.path.join(a.annotations, stem + ".json")
        if not os.path.isfile(ann):
            print("skip %s: no annotation" % stem)
            continue
        d = np.load(p)
        if "latents" not in d:
            continue
        try:
            lab = relation_labels(ann, num_objs=a.max_objects)
        except SystemExit as e:
            print("skip %s: %s" % (stem, e))
            continue
        with open(ann) as f:
            doc = json.load(f)
        clips.append((d["latents"], lab))
        metas.append({"triples": clip_triples(doc), "stem": stem})

    if len(clips) < 3:
        raise SystemExit("need at least three usable clips, have %d" % len(clips))

    r = run(clips, metas, held)
    print("clips              %d  (train %d / test %d)"
          % (r["n_clips"], r["n_train_clips"], r["n_test_clips"]))
    print("held out           %s" % ", ".join(r["held_out_categories"]))
    print("predicates tested  %d  %s"
          % (r["n_predicates_tested"],
             ", ".join(r["transferable_predicates"][:6])))
    print("")
    print("  compositional split  %s"
          % ("n/a" if r["compositional_mAP"] is None
             else "%.3f" % r["compositional_mAP"]))
    print("  random split         %s"
          % ("n/a" if r["random_mAP"] is None else "%.3f" % r["random_mAP"]))
    print("  label prior          %s"
          % ("n/a" if r["prior_mAP"] is None else "%.3f" % r["prior_mAP"]))
    print("")
    print(verdict(r))

    if a.out_dir:
        if not os.path.isdir(a.out_dir):
            os.makedirs(a.out_dir)
        with open(os.path.join(a.out_dir, "compositional.json"), "w") as f:
            json.dump(r, f, indent=2)
        _svg(r, os.path.join(a.out_dir, "compositional.svg"))
        print("\nwrote %s/compositional.json and compositional.svg" % a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
