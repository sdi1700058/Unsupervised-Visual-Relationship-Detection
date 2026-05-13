#!/usr/bin/env python3
"""ActionGenome strict primary-category validator (SPEC §I11, C4).

Loads the actiongenome dataset via the production loader with the strict
filter enabled and prints the resulting manifest. Errors if any loaded
video's primary subject does not match the requested category.

Usage:
    python3 tools/video/verify_ag.py --category chair
    AG_STRICT_CATEGORY=0 python3 tools/video/verify_ag.py --category chair  # legacy loose
"""

import os
import sys
import argparse
import collections


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--split",    default="train")
    p.add_argument("--head",     type=int, default=10)
    args = p.parse_args()

    # Force strict default unless caller overrode the env var.
    os.environ.setdefault("AG_STRICT_CATEGORY", "1")
    strict = os.environ.get("AG_STRICT_CATEGORY", "1") == "1"

    from latplan.domains.video.actiongenome import build_dataset, last_load_metadata
    build_dataset(category_filter=args.category, split=args.split)

    primaries = last_load_metadata.get("primary_categories", {}) or {}
    vid_ids   = last_load_metadata.get("video_ids", []) or []
    hist = collections.Counter(primaries.values())

    print(f"category={args.category!r}  strict={strict}  num_videos={len(vid_ids)}")
    print(f"primary-category histogram: {dict(hist)}")
    print(f"first {args.head} loaded video_ids: {vid_ids[:args.head]}")

    if strict and any(c != args.category for c in primaries.values()):
        offenders = [v for v, c in primaries.items() if c != args.category]
        print(f"FAIL — strict mode violated by {len(offenders)} video(s): {offenders[:5]}", file=sys.stderr)
        sys.exit(1)

    print(f"OK — every loaded video has primary={args.category}" if strict
          else f"OK (loose mode) — {len(vid_ids)} videos contain {args.category}")


if __name__ == "__main__":
    main()
