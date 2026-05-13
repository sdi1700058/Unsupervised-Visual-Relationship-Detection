#!/usr/bin/env python3
"""
VidVRD per-category inspector. No training required.

Reports for each category:
  - num videos containing that category
  - total annotated frames (with extracted jpg present)
  - sequential transitions (= frames - 1, summed per video)

Usage:
  python inspect_vidvrd.py                      # all-categories summary
  python inspect_vidvrd.py --category dog       # detail for one category
  python inspect_vidvrd.py --fps 5              # different fps frame dir
  python inspect_vidvrd.py --threshold 5000     # mark viable categories

Threshold default 5000 (per video-world data-volume rule).
"""

import os, json, glob, argparse, collections


def scan(ann_dir, frames_dir, category=None):
    n_videos = n_frames = n_trans = 0
    unique_predicates = set()
    for f in sorted(glob.glob(os.path.join(ann_dir, "*.json"))):
        with open(f) as fp:
            ann = json.load(fp)
        cats = {o["category"] for o in ann.get("subject/objects", [])}
        if category and category not in cats:
            continue
        vid = ann["video_id"]
        vid_frames_dir = os.path.join(frames_dir, vid)
        n = 0
        for fid, objs in enumerate(ann.get("trajectories", [])):
            if not objs:
                continue
            if os.path.exists(os.path.join(vid_frames_dir, f"{fid:06d}.jpg")):
                n += 1
        if n:
            n_videos += 1
            n_frames += n
            n_trans  += max(n - 1, 0)
            for rel in ann.get("relation_instances", []):
                unique_predicates.add(rel.get("predicate"))
    return n_videos, n_frames, n_trans, len(unique_predicates), sorted(unique_predicates)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann-dir",    default="data/video/vidvrd/annotations/train")
    p.add_argument("--frames-dir", default=None,
                   help="default: data/video/vidvrd/frames_<fps>fps/train")
    p.add_argument("--fps",        type=int, default=3)
    p.add_argument("--category",   default=None)
    p.add_argument("--threshold",  type=int, default=5000,
                   help="min transitions per category to mark viable")
    args = p.parse_args()

    if args.frames_dir is None:
        args.frames_dir = f"data/video/vidvrd/frames_{args.fps}fps/train"

    if args.category:
        v, f, t, n_preds, preds = scan(args.ann_dir, args.frames_dir, args.category)
        print(f"category={args.category}  videos={v}  frames={f}  transitions={t}  unique_predicates={n_preds}")
        print(f"viable (>= {args.threshold}): {'YES' if t >= args.threshold else 'NO'}")
        print(f"[predicates]: {', '.join(preds)}")
        print(f"Recommended P >= {n_preds} (exact); P={max(8, (n_preds + 3)//4 * 4)} adds headroom")
        return

    cats_in_videos = collections.Counter()
    for f in sorted(glob.glob(os.path.join(args.ann_dir, "*.json"))):
        with open(f) as fp:
            ann = json.load(fp)
        for c in {o["category"] for o in ann.get("subject/objects", [])}:
            cats_in_videos[c] += 1

    v_all, f_all, t_all, n_preds_all, preds_all = scan(args.ann_dir, args.frames_dir)
    print(f"ALL CATEGORIES   videos={v_all}  frames={f_all}  transitions={t_all}  unique_predicates={n_preds_all}")
    print(f"viable threshold = {args.threshold} sequential transitions")
    print()
    print(f"{'category':30s} {'videos':>6} {'transitions':>12} {'predicates':>10} viable")
    print("-" * 75)
    for cat, n_vids in cats_in_videos.most_common():
        v, f, t, n_preds, preds = scan(args.ann_dir, args.frames_dir, cat)
        flag = "YES" if t >= args.threshold else "no"
        print(f"{cat:30s} {n_vids:>6d} {t:>12d}  {n_preds:>10d}  {flag}")


if __name__ == "__main__":
    main()
