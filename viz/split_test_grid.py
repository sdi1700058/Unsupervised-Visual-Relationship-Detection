#!/usr/bin/env python3
"""viz/split_test_grid.py — decompose model.py-emitted multi-state grid PNGs
(`autoencoding_test.png`, `booleans_test.png`) into per-state single-concept
PNGs + caption.md sidecars (SPEC §D6 / V6).

model.py is read-only per C2; this script post-processes the grids it writes.

Usage:
    python3 viz/split_test_grid.py <out_dir>/                  # auto-detect both files
    python3 viz/split_test_grid.py <out_dir>/ --rows 2 --cols 5
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

from viz.io import save_with_caption


def _slice_and_save(img, out_prefix, rows, cols, what_template, why, how_to_read):
    """Slice `img` (PIL) into rows×cols equal tiles; save each as a captioned PNG."""
    W, H = img.size
    tw, th = W // cols, H // rows
    written = []
    import matplotlib.pyplot as plt
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            tile = img.crop((c*tw, r*th, (c+1)*tw, (r+1)*th))
            fig, ax = plt.subplots(figsize=(tw/100.0, th/100.0))
            ax.imshow(np.asarray(tile))
            ax.axis('off')
            png, _ = save_with_caption(
                fig,
                f"{out_prefix}_state_{idx}",
                what=what_template.format(idx=idx),
                why=why,
                how_to_read=how_to_read)
            written.append(png)
    return written


def split_autoencoding(out_dir, rows, cols):
    src = os.path.join(out_dir, "autoencoding_test.png")
    if not os.path.isfile(src):
        return []
    img = Image.open(src).convert("RGB")
    return _slice_and_save(
        img,
        os.path.join(out_dir, "auto_test"),
        rows, cols,
        what_template="Single-state extraction from the model.py-emitted `autoencoding_test.png` grid — state #{idx}.",
        why="The legacy `autoencoding_test.png` is a dense MxN grid of input/recon pairs that's hard to read at a glance. The per-state extract isolates one example for inspection.",
        how_to_read="Match the tile against your training set: if the reconstruction half is recognisably the same scene as the input half, the model trained.")


def split_booleans(out_dir, rows, cols):
    src = os.path.join(out_dir, "booleans_test.png")
    if not os.path.isfile(src):
        return []
    img = Image.open(src).convert("RGB")
    return _slice_and_save(
        img,
        os.path.join(out_dir, "booleans_test"),
        rows, cols,
        what_template="Single-state extraction from the model.py-emitted `booleans_test.png` per-predicate positive/negative grid — state #{idx}.",
        why="The legacy `booleans_test.png` packs many predicates × many examples into one image. Per-state extract lets you eyeball one predicate's typical positive/negative neighbourhood.",
        how_to_read="Compare positive vs negative columns: the predicate should ON-fire for one visually-coherent cluster and OFF-fire for everything else.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", help="Trained-model output dir containing autoencoding_test.png / booleans_test.png")
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--cols", type=int, default=5)
    args = p.parse_args()

    written = []
    written += split_autoencoding(args.out_dir, args.rows, args.cols)
    written += split_booleans(args.out_dir, args.rows, args.cols)

    if not written:
        print(f"No grid images found in {args.out_dir!r} (expected autoencoding_test.png / booleans_test.png)", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {len(written)} per-state PNG + caption.md pairs to {args.out_dir}")


if __name__ == "__main__":
    main()
