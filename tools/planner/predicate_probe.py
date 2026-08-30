#!/usr/bin/env python3
"""M1 — does the latent encode the relations, and are they readable?

**The gap this closes.** The thesis is called *Video Visual Relationship
Detection for Planning, from real world data*. Everything measured so far
scores **planning**: `mse_ratio`, the quantisation floor, action-effect
consistency. Nothing measures **relationship detection**. A reader could
finish the evaluation chapter without learning whether FOSAE's predicates
correspond to any relation a person would name.

VidVRD annotates exactly that. `relation_instances` gives, per ordered object
pair and per frame range, the predicates that hold::

    {"subject_tid": 0, "object_tid": 1, "predicate": "chase",
     "begin_fid": 0, "end_fid": 30}

So the question becomes a standard representation-probing protocol: **fit a
probe from the binary latent to the human labels and see what comes out.**

Two probes, and the pair is the point
-------------------------------------

======================  ===========================================
`ridge_probe`           a **linear** map. High means the relation is
                        *expressed* in the code: readable by the kind of
                        simple downstream reader a planner amounts to.
`knn_probe`             Hamming-nearest neighbours. High means the relation
                        is *present* in the code, however tangled.
======================  ===========================================

Low linear with high kNN is the interesting outcome, not a failure: the
relation is there but the code does not surface it. That is a weaker claim
than "FOSAE learned the relation", and stating it precisely is worth more than
a single number.

Two controls, without which the numbers mean nothing
----------------------------------------------------

- **prior** — predict each predicate's base rate. VidVRD predicates are very
  imbalanced, so a probe that learns nothing still beats chance.
- **shuffled** — the identical probe on latents permuted across the corpus.
  This keeps the label prior AND the object-pair identity and removes only the
  representation.

**The bar is `max(prior, shuffled)`, and it is usually the prior.** Measured
on 20 clips: the shuffled control scored 0.076 while the prior scored 0.119 —
scrambled latents are *worse* than no latent, because they are active noise.
Beating them proves nothing. The shuffled control still earns its place, as
what separates "the latent helped" from "the pair identity helped".

Scored by average precision per predicate, macro-averaged, because the labels
are multi-label and heavily imbalanced.

**Hold out whole clips, and use several.** `probe_export` splits one clip
temporally and is kept only for inspecting a single clip; within one clip many
predicates never change, so the probe and its control both saturate near 1.000
and the number is meaningless. `probe_corpus` is the real protocol, and
holding out whole clips also matches how the video-relation literature reports
mAP (`RELATED_WORK.md` section B).

numpy and the standard library only, so it runs on Sherlock's Python 3.6.

    python3 tools/planner/predicate_probe.py eval/probe/batch/*.npz \\
        --annotation data/video/vidvrd/annotations/train/*.json \\
        --out-dir eval/probe/M1

Reading the result
------------------

`lift = ridge_mAP - max(prior_mAP, shuffled_mAP)`.

======================  ===========================================
lift <= 0.02            the latent carries no relational information a linear
                        reader can use. If `knn_mAP` clears the bar while
                        this does not, the relations are PRESENT but not
                        EXPRESSED, which is a weaker and different claim.
0.02 < lift < 0.10      weak but real.
lift >= 0.10            the latent carries relations a linear reader can
                        recover.
======================  ===========================================

First result, 2026-08-30 — **oracle** latents, 20 screened clips, 6 held out,
58 predicates of which 24 scoreable::

    linear probe      0.116
    kNN probe         0.118
    shuffled control  0.076
    label prior       0.119
    lift             -0.003

A code built from **ground-truth boxes** carries no readable information about
VidVRD's relation labels. That is a statement about the dataset rather than
about FOSAE: predicates such as `play`, `chase` and `touch` are not determined
by where the boxes are in a single frame, so no purely positional
representation can recover them, however perfect. It is a ceiling, and every
positional model sits under it.
"""

import argparse
import json
import os
import sys

import numpy as np


class Labels(object):
    """Per-(frame, ordered pair) multi-label predicate matrix."""

    def __init__(self, frames, pairs, Y, predicates, tids):
        self.frames = np.asarray(frames)
        self.pairs = np.asarray(pairs)
        self.Y = np.asarray(Y)
        self.predicates = list(predicates)
        self.tids = list(tids)

    def __len__(self):
        return len(self.frames)


def relation_labels(ann_path, num_objs=3):
    """Read `relation_instances` into one row per (frame, ordered pair).

    Slots are assigned by total annotated area over the whole clip, which is
    exactly what `oracle.boxes_from_vidvrd` does, so slot *i* here is the same
    object as slot *i* in the latent. Both key on `tid`; ordering by per-frame
    area would make the two disagree whenever two objects' areas crossed.

    `end_fid` is treated as **exclusive**, matching the VidVRD toolkit.
    """
    with open(ann_path) as f:
        doc = json.load(f)

    def area(o):
        b = o["bbox"]
        return float((b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"]))

    totals = {}
    for frame in doc.get("trajectories", []):
        for o in (frame or []):
            tid = o.get("tid")
            if tid is not None:
                totals[tid] = totals.get(tid, 0.0) + area(o)
    if not totals:
        raise SystemExit("%s: no trajectory carries a tid" % ann_path)

    tids = sorted(totals, key=lambda t: totals[t], reverse=True)[:num_objs]
    slot_of = {tid: i for i, tid in enumerate(tids)}

    rels = doc.get("relation_instances", [])
    predicates = sorted({r["predicate"] for r in rels
                         if r.get("subject_tid") in slot_of
                         and r.get("object_tid") in slot_of})
    if not predicates:
        raise SystemExit("%s: no relation involves the chosen slots" % ann_path)

    n_frames = len(doc.get("trajectories", []))
    pair_list = [(i, j) for i in range(len(tids)) for j in range(len(tids))
                 if i != j]
    index = {(f, p): k
             for k, (f, p) in enumerate(
                 (f, p) for f in range(n_frames) for p in pair_list)}

    frames = np.array([f for f in range(n_frames) for _ in pair_list])
    pairs = np.array([p for _ in range(n_frames) for p in pair_list])
    Y = np.zeros((len(frames), len(predicates)), dtype=np.float64)

    for r in rels:
        s, o = slot_of.get(r.get("subject_tid")), slot_of.get(r.get("object_tid"))
        if s is None or o is None or s == o:
            continue
        c = predicates.index(r["predicate"])
        for f in range(max(0, r["begin_fid"]), min(n_frames, r["end_fid"])):
            Y[index[(f, (s, o))], c] = 1.0

    return Labels(frames, pairs, Y, predicates, tids)


def temporal_split(frames, test_frac=0.3):
    """Split row indices by frame number, never at random.

    Consecutive frames of a video are near duplicates. A random split puts a
    frame's own neighbour on the other side, so every probe scores near
    perfectly and the number says nothing.
    """
    frames = np.asarray(frames)
    uniq = np.unique(frames)
    cut = uniq[int(round(len(uniq) * (1.0 - test_frac)))]
    train = np.nonzero(frames < cut)[0]
    test = np.nonzero(frames >= cut)[0]
    return train, test


def average_precision(y_true, scores):
    """Area under the precision-recall curve, by the standard step rule.

    None when the label has only one class present, because AP is undefined
    there rather than zero (`SPEC.md` V29).
    """
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos = int((y > 0).sum())
    if n_pos == 0 or n_pos == len(y):
        return None

    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / n_pos)


def ridge_probe(X_train, y_train, X_test, alpha=1.0):
    """Least squares with an L2 penalty, solved in closed form.

    A *linear* probe on purpose. The question is not "can any model recover
    the relation" — that is `knn_probe` — but "is the relation expressed in a
    form a simple reader can use", which is what a planning front end is.

    Closed form rather than gradient descent so the result is deterministic
    and there is no learning rate to tune, and numpy-only so it runs on
    Sherlock's Python 3.6.
    """
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    Xt = np.asarray(X_test, dtype=np.float64)

    X = np.hstack([X, np.ones((len(X), 1))])
    Xt = np.hstack([Xt, np.ones((len(Xt), 1))])

    n_feat = X.shape[1]
    reg = alpha * np.eye(n_feat)
    reg[-1, -1] = 0.0                    # never penalise the intercept
    w = np.linalg.solve(X.T @ X + reg, X.T @ y)
    return Xt @ w


def ridge_probe_multi(X_train, Y_train, X_test, alpha=1.0):
    """`ridge_probe` for many labels at once, one factorisation for all.

    `X.T @ X` does not depend on the target, so fitting 130 predicates
    separately solves the same system 130 times. Only `X.T @ Y` changes.
    Identical results, and it is the difference between the corpus probe
    finishing in seconds and not finishing at all.
    """
    X = np.asarray(X_train, dtype=np.float64)
    Y = np.asarray(Y_train, dtype=np.float64)
    Xt = np.asarray(X_test, dtype=np.float64)

    X = np.hstack([X, np.ones((len(X), 1))])
    Xt = np.hstack([Xt, np.ones((len(Xt), 1))])

    reg = alpha * np.eye(X.shape[1])
    reg[-1, -1] = 0.0
    W = np.linalg.solve(X.T @ X + reg, X.T @ Y)
    return Xt @ W


def knn_neighbours(X_train, X_test, k=5):
    """Indices of the k nearest training rows for each test row.

    Split out from `knn_probe` because the neighbours depend only on the
    features, never on the label. Recomputing them per predicate made the
    corpus probe run for over ten minutes on twenty clips; with 130 predicates
    that is 130 identical distance matrices.

    Distance is Euclidean, which on binary codes is monotone in Hamming
    distance — the same quantity `Export.boxes_for` falls back on, so a high
    score says the information is reachable the way the pipeline already
    reaches for it.
    """
    X = np.asarray(X_train, dtype=np.float64)
    Xt = np.asarray(X_test, dtype=np.float64)
    if len(X) == 0:
        return np.zeros((len(Xt), 0), dtype=np.int64)

    k = int(min(k, len(X)))
    sq_train = (X * X).sum(axis=1)
    out = np.zeros((len(Xt), k), dtype=np.int64)
    # Chunked, and via the |a-b|^2 = |a|^2 - 2ab + |b|^2 identity rather than
    # a 3-D broadcast, so peak memory is n_block x n_train rather than
    # n_block x n_train x n_features. This workstation has been crashed once
    # by an unbounded allocation.
    for lo in range(0, len(Xt), 512):
        block = Xt[lo:lo + 512]
        d = sq_train[None, :] - 2.0 * (block @ X.T) + (block * block).sum(axis=1)[:, None]
        out[lo:lo + len(block)] = np.argpartition(d, k - 1, axis=1)[:, :k]
    return out


def knn_probe(X_train, y_train, X_test, k=5, neighbours=None):
    """Fraction of the k nearest training rows that carry the label.

    Pass `neighbours` from `knn_neighbours` to score many labels against one
    distance computation.
    """
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if neighbours is None:
        neighbours = knn_neighbours(X_train, X_test, k)
    if neighbours.shape[1] == 0:
        return np.zeros(len(neighbours))
    return y[neighbours].mean(axis=1)


def _pair_onehot(pairs, n_slots):
    """One-hot of which ordered pair a row describes.

    Given to every probe INCLUDING the controls. Without it the probe cannot
    tell "0 chases 1" from "1 chases 0" and the ceiling is artificially low;
    with it in the controls too, the pair identity cannot be mistaken for
    something the latent supplied.
    """
    pair_list = [(i, j) for i in range(n_slots) for j in range(n_slots)
                 if i != j]
    index = {p: k for k, p in enumerate(pair_list)}
    out = np.zeros((len(pairs), len(pair_list)))
    for r, p in enumerate(pairs):
        out[r, index[tuple(p)]] = 1.0
    return out


def probe_export(latents, labels, test_frac=0.3, k=5, seed=0):
    """Run both probes and both controls, and report per predicate."""
    z = np.asarray(latents, dtype=np.float64)
    keep = labels.frames < len(z)
    frames = labels.frames[keep]
    Y = labels.Y[keep]
    pairs = labels.pairs[keep]

    n_slots = len(labels.tids)
    pair_bits = _pair_onehot(pairs, n_slots)
    X = np.hstack([z[frames], pair_bits])

    # The control keeps the pair identity and the label prior and removes only
    # the representation, so the gap is what the latent is responsible for.
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(z))
    X_shuf = np.hstack([z[perm][frames], pair_bits])

    train, test = temporal_split(frames, test_frac)
    if len(train) == 0 or len(test) == 0:
        raise SystemExit("clip too short to split temporally")

    rows = []
    for c, name in enumerate(labels.predicates):
        y = Y[:, c]
        ap_prior = average_precision(y[test], np.full(len(test), y[train].mean()))
        rows.append({
            "predicate": name,
            "positives": int(y.sum()),
            "ridge": average_precision(y[test],
                                       ridge_probe(X[train], y[train], X[test])),
            "knn": average_precision(y[test],
                                     knn_probe(X[train], y[train], X[test], k)),
            "shuffled": average_precision(
                y[test], ridge_probe(X_shuf[train], y[train], X_shuf[test])),
            "prior": ap_prior,
        })

    def macro(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "n_rows": int(len(frames)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_predicates": len(labels.predicates),
        "n_scored": sum(1 for r in rows if r["ridge"] is not None),
        "ridge_mAP": macro("ridge"),
        "knn_mAP": macro("knn"),
        "shuffled_mAP": macro("shuffled"),
        "prior_mAP": macro("prior"),
    }
    if summary["ridge_mAP"] is not None and summary["shuffled_mAP"] is not None:
        summary["gap"] = summary["ridge_mAP"] - summary["shuffled_mAP"]
    else:
        summary["gap"] = None
    return summary, rows


def probe_corpus(clips, k=5, test_frac=0.3, seed=0):
    """Probe across many clips, holding whole clips out. **Prefer this.**

    Running the single-clip version first was worth it, because it produced a
    finding about the measurement rather than about the model: within one clip
    the linear probe and its shuffled control both scored 1.000 on five of
    eight clips. 16% of VidVRD relation instances span their whole clip, and
    within the short test half of a short clip many more are effectively
    constant — so there was nothing to predict and the probe was reading a
    label that never changed.

    Holding out whole **clips** removes that failure and is also the protocol
    the video-relation literature uses (mAP over a corpus, `RELATED_WORK.md`
    section B), which makes the number comparable rather than bespoke.

    `clips` is a sequence of `(latents, Labels)` pairs.
    """
    if not clips:
        raise SystemExit("no clips to probe")

    predicates = sorted({p for _, lab in clips for p in lab.predicates})
    if not predicates:
        raise SystemExit("no predicate appears in any clip")
    max_slots = max(len(lab.tids) for _, lab in clips)

    Xs, Ys, clip_id = [], [], []
    for c, (z, lab) in enumerate(clips):
        z = np.asarray(z, dtype=np.float64)
        keep = lab.frames < len(z)
        if not keep.any():
            continue
        frames, pairs = lab.frames[keep], lab.pairs[keep]

        # One column block per predicate in the CORPUS vocabulary, so clips
        # that lack a predicate contribute genuine negatives for it.
        y = np.zeros((int(keep.sum()), len(predicates)))
        for j, name in enumerate(lab.predicates):
            y[:, predicates.index(name)] = lab.Y[keep][:, j]

        Xs.append(np.hstack([z[frames], _pair_onehot(pairs, max_slots)]))
        Ys.append(y)
        clip_id.append(np.full(int(keep.sum()), c))

    if not Xs:
        raise SystemExit("no clip had latents covering its annotated frames")

    widths = {x.shape[1] for x in Xs}
    if len(widths) != 1:
        raise SystemExit("clips disagree on latent width: %s" % sorted(widths))

    X = np.vstack(Xs)
    Y = np.vstack(Ys)
    clip_id = np.concatenate(clip_id)

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(clips))
    n_test = max(1, int(round(len(clips) * test_frac)))
    test_clips = set(order[:n_test].tolist())
    test = np.array([i for i in range(len(X)) if clip_id[i] in test_clips])
    train = np.array([i for i in range(len(X)) if clip_id[i] not in test_clips])
    if len(train) == 0 or len(test) == 0:
        raise SystemExit("need at least two clips to hold one out")

    # The control keeps the label prior and the pair identity, and permutes
    # only the latent columns -- across the whole corpus, so a clip's own
    # latents cannot survive in place.
    n_z = X.shape[1] - (max_slots * (max_slots - 1))
    X_shuf = X.copy()
    X_shuf[:, :n_z] = X[rng.permutation(len(X)), :n_z]

    # One factorisation and one distance matrix for every predicate, not one
    # each. See `ridge_probe_multi` and `knn_neighbours`.
    S_real = ridge_probe_multi(X[train], Y[train], X[test])
    S_shuf = ridge_probe_multi(X_shuf[train], Y[train], X_shuf[test])
    nbr = knn_neighbours(X[train], X[test], k)

    rows = []
    for c, name in enumerate(predicates):
        y = Y[:, c]
        rows.append({
            "predicate": name,
            "positives": int(y.sum()),
            "ridge": average_precision(y[test], S_real[:, c]),
            "knn": average_precision(
                y[test], knn_probe(None, y[train], None, neighbours=nbr)),
            "shuffled": average_precision(y[test], S_shuf[:, c]),
            "prior": average_precision(y[test],
                                       np.full(len(test), y[train].mean())),
        })

    def macro(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "n_clips": len(clips),
        "n_test_clips": len(test_clips),
        "n_rows": int(len(X)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_predicates": len(predicates),
        "n_scored": sum(1 for r in rows if r["ridge"] is not None),
        "ridge_mAP": macro("ridge"),
        "knn_mAP": macro("knn"),
        "shuffled_mAP": macro("shuffled"),
        "prior_mAP": macro("prior"),
    }
    summary["gap"] = (None if summary["ridge_mAP"] is None
                      or summary["shuffled_mAP"] is None
                      else summary["ridge_mAP"] - summary["shuffled_mAP"])
    return summary, rows


def verdict(summary):
    """One sentence, against the HARDER of the two controls.

    The shuffled control alone is too easy, and the first corpus run showed
    why: scrambled latents scored 0.076 while simply predicting each
    predicate's base rate scored 0.119. Latents permuted across the corpus are
    worse than no latent at all, because they are active noise. Beating them
    is not evidence of anything.

    So the baseline is `max(prior, shuffled)`. The shuffled control still
    earns its place — it is what separates "the latent helped" from "the pair
    identity helped" — but it is a floor, not the bar.
    """
    ridge = summary.get("ridge_mAP")
    if ridge is None:
        return "No predicate could be scored: every label is one class."

    controls = [v for v in (summary.get("prior_mAP"),
                            summary.get("shuffled_mAP")) if v is not None]
    if not controls:
        return "No control could be scored, so the probe means nothing."
    base = max(controls)
    lift = ridge - base
    summary["lift_over_best_control"] = lift

    knn = summary.get("knn_mAP")
    tangled = (knn is not None and knn - base > 0.05 and lift <= 0.02)

    if lift <= 0.02:
        head = ("The latent carries NO relational information a linear reader "
                "can use: probe %.3f against a best control of %.3f."
                % (ridge, base))
        if tangled:
            return (head + " The kNN probe reaches %.3f, so the information IS "
                    "present but is not expressed in a readable form."
                    % knn)
        return head + " The relations are not there to be read."
    if lift < 0.10:
        return ("The latent carries WEAK relational information: probe %.3f "
                "against a best control of %.3f, a lift of %.3f."
                % (ridge, base, lift))
    return ("The latent CARRIES relations a linear reader can recover: probe "
            "%.3f against a best control of %.3f, a lift of %.3f."
            % (ridge, base, lift))


def _svg(rows, path, width=760, bar_h=18):
    """A bar per predicate: ridge against its shuffled control."""
    scored = [r for r in rows if r["ridge"] is not None][:24]
    if not scored:
        return
    h = 40 + len(scored) * (bar_h + 8)
    left = 190
    span = float(width - left - 60)
    out = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" '
           'font-family="sans-serif">' % (width, h)]
    out.append('<text x="8" y="18" font-size="13" fill="#444">'
               'linear probe (blue) against its shuffled control (grey), '
               'average precision</text>')
    for i, r in enumerate(scored):
        y = 34 + i * (bar_h + 8)
        sh = r["shuffled"] or 0.0
        name = r["predicate"]
        if len(name) > 26:
            name = name[:25] + "…"
        out.append('<text x="8" y="%d" font-size="11" fill="#333">%s</text>'
                   % (y + 13, name.replace("&", "&amp;").replace("<", "&lt;")))
        out.append('<rect x="%d" y="%d" width="%.1f" height="%d" fill="#c9ced8"/>'
                   % (left, y, span * sh, bar_h))
        out.append('<rect x="%d" y="%d" width="%.1f" height="%d" fill="#1f6feb" '
                   'opacity="0.85"/>' % (left, y + 4, span * r["ridge"], bar_h - 8))
        out.append('<text x="%.1f" y="%d" font-size="10" fill="#666">%.2f</text>'
                   % (left + span + 6, y + 13, r["ridge"]))
    out.append('</svg>')
    with open(path, "w") as f:
        f.write("".join(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("export", nargs="+",
                    help="planner export npz files holding `latents`. With "
                         "more than one, whole clips are held out, which is "
                         "the protocol to prefer -- see `probe_corpus`.")
    ap.add_argument("--annotation", required=True, nargs="+",
                    help="the VidVRD annotation json for each export, in the "
                         "same order")
    ap.add_argument("--max-objects", type=int, default=3)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("-k", type=int, default=5, help="neighbours for the kNN probe")
    ap.add_argument("--out-dir", default=None,
                    help="write probe.json and probe.svg here")
    a = ap.parse_args(argv)

    if len(a.export) != len(a.annotation):
        raise SystemExit("%d exports but %d annotations"
                         % (len(a.export), len(a.annotation)))

    clips = []
    for exp, ann in zip(a.export, a.annotation):
        # No allow_pickle: only `latents` is read and it is a plain numeric
        # array, so nothing here needs to deserialise an object array.
        d = np.load(exp)
        if "latents" not in d:
            raise SystemExit("%s has no `latents`" % exp)
        try:
            clips.append((d["latents"],
                          relation_labels(ann, num_objs=a.max_objects)))
        except SystemExit as e:
            print("skip %s: %s" % (os.path.basename(ann), e))

    if not clips:
        raise SystemExit("no usable clip")

    if len(clips) == 1:
        print("NOTE: one clip only. Within a single clip many predicates never "
              "change, so the probe and its control both saturate and the gap "
              "is meaningless. Pass several exports.\n")
        summary, rows = probe_export(clips[0][0], clips[0][1], a.test_frac, a.k)
        print("clip           %s" % os.path.basename(a.annotation[0]))
    else:
        summary, rows = probe_corpus(clips, k=a.k, test_frac=a.test_frac)
        print("clips          %d, holding out %d whole"
              % (summary["n_clips"], summary["n_test_clips"]))

    latents = clips[0][0]
    print("latent         %d bits" % latents.shape[1])
    print("rows           %d  (train %d / test %d)"
          % (summary["n_rows"], summary["n_train"], summary["n_test"]))
    print("predicates     %d, of which %d scoreable"
          % (summary["n_predicates"], summary["n_scored"]))
    print("")
    for key, label in (("ridge_mAP", "linear probe   "),
                       ("knn_mAP", "kNN probe      "),
                       ("shuffled_mAP", "shuffled control"),
                       ("prior_mAP", "label prior    ")):
        v = summary[key]
        print("  %s %s" % (label, "n/a" if v is None else "%.3f" % v))
    print("")
    v = verdict(summary)
    lift = summary.get("lift_over_best_control")
    print("  lift over the best control  %s"
          % ("n/a" if lift is None else "%+.3f" % lift))
    print("")
    print(v)

    if a.out_dir:
        if not os.path.isdir(a.out_dir):
            os.makedirs(a.out_dir)
        with open(os.path.join(a.out_dir, "probe.json"), "w") as f:
            json.dump({"summary": summary, "per_predicate": rows}, f, indent=2)
        _svg(rows, os.path.join(a.out_dir, "probe.svg"))
        print("\nwrote %s/probe.json and probe.svg" % a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
