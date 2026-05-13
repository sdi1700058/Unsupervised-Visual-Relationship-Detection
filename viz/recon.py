#!/usr/bin/env python3
"""viz/recon.py — single-glance reconstruction-quality grid for a trained FOSAE.

Reads the per-category npz cache that the model was trained on (resolved via
the model_dir's `loaded_videos.json`), samples N evenly-spaced states,
encodes/decodes them, and writes a single grid PNG + caption showing:

    row 1: input scene
    row 2: decoded reconstruction
    row 3: |input - reconstruction|  (hot)

This is the most direct test of whether FOSAE is learning the world. No
`frames_dir` is touched — the cache npz alone is sufficient.

Usage:
    python3 viz/recon.py <model_dir> [--num 8] [--domain vidvrd|actiongenome]
"""

import os
import sys
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("--num", type=int, default=8)
    p.add_argument("--domain", default=None,
                   help="vidvrd | actiongenome (auto-detected from model_dir path / manifest if omitted)")
    args = p.parse_args()

    # --- read manifest for category + fps ---
    manifest_path = os.path.join(args.model_dir, "loaded_videos.json")
    if not os.path.isfile(manifest_path):
        sys.exit(f"ERROR: no loaded_videos.json in {args.model_dir}; cannot determine training subset.")
    with open(manifest_path) as f:
        manifest = json.load(f)
    category = manifest.get("category_filter")
    fps      = manifest.get("fps", 3)
    print(f"[recon] category={category!r}  fps={fps!r}")

    # --- domain dispatch ---
    if args.domain is None:
        args.domain = "actiongenome" if "actiongenome" in args.model_dir else "vidvrd"

    if args.domain == "vidvrd":
        from latplan.puzzles.puzzle_vidvrd import build_dataset
    elif args.domain == "actiongenome":
        from latplan.domains.video.actiongenome import build_dataset
    else:
        sys.exit(f"ERROR: unknown --domain {args.domain!r}")

    from latplan.puzzles.puzzle_labeled_objects import PICSIZE
    from latplan.puzzles.util import preprocess
    from strips import bboxes_to_onehot
    from viz.io import save_with_caption

    # --- model load ---
    import latplan
    import latplan.model
    import keras.optimizers
    from keras_radam import RAdam
    from keras_adabound import AdaBound
    setattr(keras.optimizers, "radam", RAdam)
    setattr(keras.optimizers, "adabound", AdaBound)

    print(f"[recon] loading model from {args.model_dir}")
    ae = latplan.model.load(args.model_dir)
    ae.load()
    U = ae.parameters['U']
    P = ae.parameters['P']
    print(f"[recon] model loaded — U={U} P={P} A={ae.parameters['A']}")

    # --- load arrays (cache hit if available) ---
    print(f"[recon] loading {args.domain} cache for category={category!r}")
    images, bboxes, per_state_names, frame_ids = build_dataset(category_filter=category, fps=fps)

    picsize_grid = (np.array(PICSIZE) // 5).astype(int)
    Y, X = picsize_grid[0], picsize_grid[1]
    num_states, num_objs = images.shape[0], images.shape[1]

    images_f = (images.astype(np.float32) / 256)
    images_f = preprocess(images_f)
    bboxes_onehot = bboxes_to_onehot(bboxes, X, Y)
    states = np.concatenate(
        (images_f.reshape((num_states, num_objs, -1)),
         bboxes_onehot.reshape((num_states, num_objs, -1))),
        axis=-1).astype(np.float32)

    # --- evenly-spaced sample ---
    N = min(args.num, num_states)
    idx = np.linspace(0, num_states - 1, N).astype(int)
    sample = states[idx]

    render_fn, _ = ae.blocks_renderer()
    recon_feat   = ae.autoencode(sample)
    inputs       = render_fn(sample)
    recons       = render_fn(recon_feat)
    diffs        = np.abs(inputs - recons)
    per_state_mse = ((recon_feat - sample) ** 2).mean(axis=(1, 2))

    # --- grid figure (3 rows × N cols) ---
    cols = max(1, N)
    fig, axes = plt.subplots(3, cols, figsize=(2.0 * cols, 6))
    if cols == 1:
        axes = axes.reshape(3, 1)
    for i in range(N):
        for r, (img, title, cmap) in enumerate([
            (inputs[i], f"in #{idx[i]}", "gray"),
            (recons[i], "recon",          "gray"),
            (diffs[i],  f"diff MSE={per_state_mse[i]:.3f}", "hot"),
        ]):
            ax = axes[r, i]
            if img.ndim == 2:
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(np.clip(img, 0, 1))
            ax.axis("off")
            ax.set_title(title, fontsize=8)
    fig.suptitle(f"Reconstruction grid — {args.domain} category={category!r}  N={N} states  "
                 f"mean MSE={per_state_mse.mean():.4f}", fontsize=10)

    out_dir = os.path.join(args.model_dir, "viz")
    os.makedirs(out_dir, exist_ok=True)
    save_with_caption(fig, os.path.join(out_dir, "recon_grid"),
        what=f"Side-by-side input / reconstruction / absolute-difference for {N} evenly-spaced sample states from the training set (category={category!r}, domain={args.domain}).",
        why="This is the single-glance answer to 'is the model learning?'. If row 2 visually matches row 1 you have a working autoencoder; if row 3 is mostly dark the reconstruction is faithful. Per-state MSE annotated on the diff row.",
        how_to_read="Row 1 = input scene rendered on the FOSAE canvas. Row 2 = decoded scene after encode→decode. Row 3 = |row1 - row2| with the 'hot' colormap (brighter = worse).")

    print(f"[recon] wrote {out_dir}/recon_grid.png + recon_grid.caption.md")
    print(f"[recon] per-state MSE  min={per_state_mse.min():.4f}  max={per_state_mse.max():.4f}  mean={per_state_mse.mean():.4f}")


if __name__ == "__main__":
    main()
