#!/usr/bin/env python3
"""The ceiling for the thesis question: perfect relations, fed to a planner.

**What an oracle is here.** A *source of state*, not a metric. Every evaluation
method in this project consumes a planner export; an oracle builds that export
from the annotations instead of from a trained model, so a method can be run on
perfect input and on learned input and the two compared. That is what a "paired
combination" means -- one dataset, one method, both sources.

**Why this one exists beside `oracle.py`.** `oracle.py` builds the state from
ground-truth **boxes**, which answers "can a classical planner do frame
interpolation from a binary state at all". That is a ceiling on the planner and
it says nothing about relations.

The thesis asks whether FOSAE's **relations** are plannable. Until 2026-09-07
ground-truth relations had never been fed to a planner anywhere in this
project: `relation_instances` was read by `compositional.py` and `m7_map.py`,
and both are scorers. So there was no ceiling for the question actually being
asked. This is that ceiling. **If a planner cannot use perfect relations, no
amount of training would have helped.**

**The encoding.** VidVRD and VidOR annotate relations as

    {"subject_tid": 0, "predicate": "stand_behind", "object_tid": 1,
     "begin_fid": 0, "end_fid": 30}

so the true relation set at frame `f` is every instance with
`begin_fid <= f < end_fid`. Verified against the data: `end_fid` is exclusive
and never exceeds the trajectory length; relations are annotated in 30-frame
segments.

The state is one bit per (ordered object pair, predicate). Ordered, because
`dog bites frisbee` is not `frisbee bites dog`. Self-pairs are absent from the
layout, so no bit can encode an object related to itself. The width is
`num_objs * (num_objs - 1) * len(vocabulary)`.

**Slots come from `tid`, fixed once for the whole clip**, for the reason
`puzzle_vidvrd._slot_order_by_tid` gives: a slot that means a different object
in consecutive frames turns a non-transition into a transition.

**The vocabulary is per clip by default** and travels in the export. A global
vocabulary would make two clips' latents comparable bit for bit, but it also
makes the state mostly zeros and the mined action set enormous. `--vocabulary`
takes a file when clips must be compared directly.

**Boxes are still in the export.** The relational state is what the planner
plans over; `gt_boxes` is what a method scores against. A ground-truth state
decodes to the ground-truth boxes of the frame it came from, so
`decoded_boxes` is `gt_boxes` -- exact by construction rather than by lookup.

    python3 tools/planner/relation_oracle.py \\
        data/video/vidvrd/annotations/train/ILSVRC2015_train_00005003.json \\
        --out eval/exports/reloracle-5003.npz --max-objects 3

Standard library and numpy only, Python 3.6 clean.
"""

import argparse
import json
import os
import sys

import numpy as np


def vocabulary(doc):
    """Every predicate the clip uses, sorted so the layout is reproducible."""
    return sorted({r["predicate"] for r in doc.get("relation_instances", [])
                   if r.get("predicate")})


def pair_count(num_objs):
    """Ordered pairs of distinct slots. Self-pairs are not represented."""
    return num_objs * (num_objs - 1)


def _pair_index(num_objs):
    """`{(subject_slot, object_slot): position}`, self-pairs excluded."""
    out, i = {}, 0
    for s in range(num_objs):
        for o in range(num_objs):
            if s == o:
                continue
            out[(s, o)] = i
            i += 1
    return out


def slot_of_tid(doc, num_objs):
    """`{tid: slot}`, fixed for the whole clip.

    Ranked by how many frames the track appears in, then by total box area,
    then by tid so the order is total. Tracks past `num_objs` get no slot and
    their relations are dropped and counted, rather than rotating into
    whichever slot happens to be free.
    """
    count, area = {}, {}
    for frame in doc.get("trajectories", []):
        for obj in frame or []:
            tid = obj.get("tid")
            if tid is None:
                continue
            b = obj.get("bbox") or {}
            count[tid] = count.get(tid, 0) + 1
            area[tid] = area.get(tid, 0) + (
                max(0, b.get("xmax", 0) - b.get("xmin", 0))
                * max(0, b.get("ymax", 0) - b.get("ymin", 0)))
    ranked = sorted(count, key=lambda t: (-count[t], -area.get(t, 0), str(t)))
    return dict((tid, slot) for slot, tid in enumerate(ranked[:num_objs]))


def states_from_annotation(doc, num_objs=3, vocab=None):
    """`(states, vocab, stats)` for one clip.

    `states` is `(n_frames, pair_count * len(vocab))` int8. `vocab` defaults to
    the clip's own predicates; pass one to compare clips bit for bit.
    """
    vocab = list(vocabulary(doc) if vocab is None else vocab)

    n_frames = len(doc.get("trajectories", []))
    pairs = _pair_index(num_objs)
    slots = slot_of_tid(doc, num_objs)
    width = len(pairs) * len(vocab)
    states = np.zeros((n_frames, width), dtype=np.int8)

    index_of = dict((p, i) for i, p in enumerate(vocab))
    stats = {"n_frames": n_frames, "n_predicates": len(vocab),
             "n_pairs": len(pairs), "dropped_unmapped": 0,
             "dropped_unknown_predicate": 0,
             "slot_of_tid": dict((str(k), v) for k, v in slots.items())}

    for r in doc.get("relation_instances", []):
        s_slot = slots.get(r.get("subject_tid"))
        o_slot = slots.get(r.get("object_tid"))
        if s_slot is None or o_slot is None or s_slot == o_slot:
            # A relation naming a track with no slot cannot be placed. Counted
            # rather than guessed: putting it in slot 0 would invent a fact.
            stats["dropped_unmapped"] += 1
            continue
        p = index_of.get(r.get("predicate"))
        if p is None:
            stats["dropped_unknown_predicate"] += 1
            continue
        col = pairs[(s_slot, o_slot)] * len(vocab) + p
        begin = max(0, int(r.get("begin_fid", 0)))
        end = min(n_frames, int(r.get("end_fid", 0)))     # exclusive
        if end > begin:
            states[begin:end, col] = 1

    stats["n_transitions"] = int(sum(
        1 for i in range(n_frames - 1)
        if not np.array_equal(states[i], states[i + 1])))
    return states, vocab, stats


def boxes_from_annotation(doc, num_objs=3):
    """`(n_frames, num_objs, 4)` boxes in the same slot order as the states."""
    slots = slot_of_tid(doc, num_objs)
    frames = doc.get("trajectories", [])
    out = np.zeros((len(frames), num_objs, 4), dtype=np.float64)
    for f, frame in enumerate(frames):
        for obj in frame or []:
            slot = slots.get(obj.get("tid"))
            if slot is None:
                continue
            b = obj.get("bbox") or {}
            out[f, slot] = [b.get("xmin", 0), b.get("ymin", 0),
                            b.get("xmax", 0), b.get("ymax", 0)]
    return out


def build_export(path, out_path, num_objs=3, vocab_path=None):
    """Write a planner export whose latents are ground-truth relations."""
    with open(path) as handle:
        doc = json.load(handle)

    vocab = None
    if vocab_path:
        with open(vocab_path) as handle:
            vocab = [line.strip() for line in handle if line.strip()]

    states, vocab, stats = states_from_annotation(
        doc, num_objs=num_objs, vocab=vocab)

    if not vocab or states.shape[1] == 0:
        raise SystemExit(
            "%s annotates no relations, so there is no relational state to "
            "build. An all-zero export would read downstream as a planner "
            "that solved nothing, which is a different claim." % path)
    if not stats["n_transitions"]:
        sys.stderr.write(
            "warning: no relation changes anywhere in %s, so this clip holds "
            "no transition for a planner to mine.\n" % path)

    boxes = boxes_from_annotation(doc, num_objs=num_objs)
    video = doc.get("video_id") or os.path.splitext(os.path.basename(path))[0]
    frame_ids = np.array(["%s/%06d" % (video, i) for i in range(len(states))],
                         dtype="U256")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(
        out_path,
        latents=states.astype(np.int8),
        gt_boxes=boxes,
        # A ground-truth state decodes to the ground-truth boxes of the frame
        # it came from, exactly, so there is no lookup and no fallback.
        decoded_boxes=boxes,
        n_bits=np.int64(states.shape[1]),
        model_name=np.str_("relation-oracle"),
        frame_ids=frame_ids,
        predicates=np.array(vocab, dtype="U64"),
        num_objs=np.int64(num_objs),
    )
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("annotation", nargs="+", help="VidVRD/VidOR json")
    ap.add_argument("--out", default=None, help="output npz (single input)")
    ap.add_argument("--out-dir", default=None, help="output dir (many inputs)")
    ap.add_argument("--max-objects", type=int, default=3)
    ap.add_argument("--vocabulary", default=None,
                    help="one predicate per line, to compare clips directly")
    a = ap.parse_args(argv)

    if len(a.annotation) > 1 and not a.out_dir:
        sys.stderr.write("several inputs need --out-dir\n")
        return 2

    rc = 0
    for path in a.annotation:
        stem = os.path.splitext(os.path.basename(path))[0]
        out = a.out if a.out else os.path.join(a.out_dir, stem + ".npz")
        try:
            stats = build_export(path, out, num_objs=a.max_objects,
                                 vocab_path=a.vocabulary)
        except SystemExit as exc:
            sys.stderr.write("%s\n" % exc)
            rc = 1
            continue
        print("%-40s %3d frames  %3d predicates  %3d pairs  %5d bits  "
              "%3d transitions%s"
              % (stem, stats["n_frames"], stats["n_predicates"],
                 stats["n_pairs"],
                 stats["n_pairs"] * stats["n_predicates"],
                 stats["n_transitions"],
                 ("  (%d relations dropped, no slot)"
                  % stats["dropped_unmapped"])
                 if stats["dropped_unmapped"] else ""))
    return rc


if __name__ == "__main__":
    sys.exit(main())
