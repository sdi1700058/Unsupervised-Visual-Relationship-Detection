#!/usr/bin/env python3
"""Verify VidVRD category filter loads only intended videos. No training.

Run inside the fosae conda env (needs PIL + numpy):
  python3 verify_category.py --category bicycle
  VIDVRD_STRICT_CATEGORY=0 python3 verify_category.py --category dog  # legacy loose
"""
import argparse, collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latplan.puzzles.puzzle_vidvrd import build_dataset, last_load_metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--fps", type=int, default=3)
    p.add_argument("--split", default="train")
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--head", type=int, default=10)
    args = p.parse_args()

    root = os.path.join(os.path.dirname(__file__), "data", "video", "vidvrd")
    build_dataset(
        annotations_dir=os.path.join(root, "annotations", args.split),
        frames_dir=os.path.join(root, f"frames_{args.fps}fps", args.split),
        max_videos=args.max_videos, category_filter=args.category)
    m = last_load_metadata
    print(f"[verify] category={m['category_filter']} strict={m['strict']}")
    print(f"[verify] loaded {m['num_videos']} videos, {m['num_states']} states")
    print(f"[verify] first {args.head}:")
    for vid in m["video_ids"][:args.head]:
        print(f"  {vid}  primary={m['primary_categories'].get(vid)}")
    hist = collections.Counter(m["primary_categories"].values())
    print(f"[verify] primary-category histogram: {dict(hist)}")
    bad = [v for v, c in m["primary_categories"].items() if c != args.category]
    if bad:
        print(f"[verify] FAIL — {len(bad)} videos have primary != {args.category}")
        sys.exit(1)
    print(f"[verify] OK — every loaded video has primary={args.category}")


if __name__ == "__main__":
    main()
