#!/usr/bin/env python3
"""ActionGenome per-class viability + unique-predicate report (SPEC §I11, C3).

Pure stdlib + pickle — does NOT import keras / tensorflow / latplan.
Reads `data/video/actiongenome/annotations/*.pkl` directly.

Usage:
    python3 tools/video/inspect_ag.py                         # full table
    python3 tools/video/inspect_ag.py --category chair        # one class detail
    python3 tools/video/inspect_ag.py --threshold 5000        # viability cutoff
    python3 tools/video/inspect_ag.py --split test            # test split

Output:
    Per-class:
      videos (primary-subject = class)
      frames (any visible instance)
      sequential_transitions
      unique_predicates (attention ∪ spatial ∪ contacting)
      viable (>= threshold)
"""

import os
import pickle
import argparse
import collections


_REL_KEYS = ("attention_relationship", "spatial_relationship", "contacting_relationship")


def _load(ann_dir, split):
    with open(os.path.join(ann_dir, "object_bbox_and_relationship.pkl"), "rb") as f:
        obj = pickle.load(f)
    with open(os.path.join(ann_dir, "person_bbox.pkl"), "rb") as f:
        per = pickle.load(f)
    with open(os.path.join(ann_dir, "frame_list.txt")) as f:
        frames = [l.strip() for l in f if l.strip()]
    if split is not None:
        kept = []
        for k in frames:
            anns = obj.get(k, [])
            if anns and (anns[0].get("metadata") or {}).get("set") == split:
                kept.append(k)
        frames = kept
    return obj, per, frames


def _primary(frames_of_vid, obj_anno):
    counts, areas = {}, {}
    for fk in frames_of_vid:
        for o in obj_anno.get(fk, []):
            if not o.get("visible") or o.get("bbox") is None:
                continue
            c = o.get("class")
            if c is None or c == "person":
                continue
            counts[c] = counts.get(c, 0) + 1
            x1, y1, x2, y2 = o["bbox"]
            areas[c] = areas.get(c, 0) + max(0, x2 - x1) * max(0, y2 - y1)
    if not counts:
        return None
    return max(counts, key=lambda c: (counts[c], areas.get(c, 0)))


def _scan(obj_anno, per_anno, frames, category=None):
    vid_to_frames = collections.defaultdict(list)
    for k in frames:
        vid_to_frames[k.split("/", 1)[0]].append(k)
    for v in vid_to_frames:
        vid_to_frames[v].sort()

    n_videos = n_frames = n_trans = 0
    preds = set()

    for vid, fks in vid_to_frames.items():
        if category is not None and _primary(fks, obj_anno) != category:
            continue
        kept_per_vid = 0
        for fk in fks:
            objs   = obj_anno.get(fk, [])
            person = per_anno.get(fk, {}) or {}
            visible = [o for o in objs if o.get("visible") and o.get("bbox") is not None]
            has_person = person.get("bbox") is not None and len(person.get("bbox", [])) > 0
            if not visible and not has_person:
                continue
            kept_per_vid += 1
            for o in objs:
                if not o.get("visible"):
                    continue
                for rk in _REL_KEYS:
                    r = o.get(rk)
                    if not r:
                        continue
                    for p in r:
                        preds.add(p)
        if kept_per_vid > 0:
            n_videos += 1
            n_frames += kept_per_vid
            n_trans  += max(kept_per_vid - 1, 0)
    return n_videos, n_frames, n_trans, len(preds), sorted(preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann-dir",   default="data/video/actiongenome/annotations")
    p.add_argument("--split",     default="train", choices=["train", "test", None])
    p.add_argument("--category",  default=None)
    p.add_argument("--threshold", type=int, default=5000)
    args = p.parse_args()

    obj, per, frames = _load(args.ann_dir, args.split)

    if args.category:
        v, f, t, n_p, preds = _scan(obj, per, frames, args.category)
        print(f"category={args.category}  videos={v}  frames={f}  transitions={t}  unique_predicates={n_p}")
        print(f"viable (>= {args.threshold}): {'YES' if t >= args.threshold else 'no'}")
        print(f"[predicates]: {', '.join(preds)}")
        print(f"Suggested P floor = {n_p}; target (U*P) ≥ 3200 (see AUDIT §V10 amendment).")
        return

    v_all, f_all, t_all, n_p_all, _ = _scan(obj, per, frames)
    print(f"ALL CATEGORIES  videos={v_all}  frames={f_all}  transitions={t_all}  unique_predicates={n_p_all}")
    print(f"viable threshold = {args.threshold} sequential transitions")
    print()

    # enumerate non-person classes that appear as primary subject for at least 1 video
    vid_to_frames = collections.defaultdict(list)
    for k in frames:
        vid_to_frames[k.split("/", 1)[0]].append(k)
    primary_counter = collections.Counter()
    for vid, fks in vid_to_frames.items():
        c = _primary(fks, obj)
        if c is not None:
            primary_counter[c] += 1

    print(f"{'category':22s} {'videos':>6} {'transitions':>12} {'predicates':>11}  viable")
    print("-" * 70)
    for cat, n_vids in primary_counter.most_common():
        v, f, t, n_p, _ = _scan(obj, per, frames, cat)
        flag = "YES" if t >= args.threshold else "no"
        print(f"{cat:22s} {n_vids:>6d} {t:>12d} {n_p:>11d}  {flag}")


if __name__ == "__main__":
    main()
