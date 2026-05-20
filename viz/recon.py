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
                   help="vidvrd | actiongenome | puzzle | blocks (auto-detected from model_dir path / manifest if omitted)")
    p.add_argument("--type",  default="mnist", help="(puzzle domain) puzzle type")
    p.add_argument("--width", type=int, default=3)
    p.add_argument("--height", type=int, default=3)
    p.add_argument("--track", default="blocks-5-3", help="(blocks domain) track name")
    args = p.parse_args()

    # --- domain dispatch ---
    if args.domain is None:
        if   "actiongenome" in args.model_dir: args.domain = "actiongenome"
        elif "vidvrd"        in args.model_dir: args.domain = "vidvrd"
        elif "blocks"        in args.model_dir: args.domain = "blocks"
        else:                                   args.domain = "puzzle"

    # video-world domains need a manifest (loaded_videos.json) to know category + fps
    is_video = args.domain in ("vidvrd", "actiongenome")
    category, fps, vid_id, npz_path = None, None, None, None
    if is_video:
        manifest_path = os.path.join(args.model_dir, "loaded_videos.json")
        if not os.path.isfile(manifest_path):
            sys.exit(f"ERROR: no loaded_videos.json in {args.model_dir}; cannot determine training subset.")
        with open(manifest_path) as f:
            manifest = json.load(f)
        category = manifest.get("category_filter")
        fps      = manifest.get("fps", 3)
        vid_id   = manifest.get("video_id_filter") or manifest.get("video_id")
        npz_path = manifest.get("npz_path")
        print(f"[recon] category={category!r}  fps={fps!r}  video_id={vid_id!r}")

    from latplan.puzzles.util import preprocess
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

    # --- load arrays + build the (N, num_objs, feature_dim) sample tensor ---
    if args.domain == "puzzle":
        # puzzle domain — paper's 8-puzzle. Tiles, not bboxed objects.
        from latplan.util.paths import find_dataset
        import importlib
        p_mod = importlib.import_module(f"latplan.puzzles.puzzle_{args.type}")
        p_mod.setup()
        path = find_dataset(f"puzzle-{args.type}-{args.width}-{args.height}.npz")
        with np.load(path) as data:
            configs = data["pres"]
        all_objects = p_mod.to_objects(configs, args.width, args.height, False)  # (N_all, n_tiles, 15)
        N = min(args.num, all_objects.shape[0])
        idx = np.linspace(0, all_objects.shape[0] - 1, N).astype(int)
        sample = all_objects[idx].astype(np.float32)
        render_fn, _ = ae.puzzle_renderer()

    elif args.domain == "blocks":
        # blocksworld — uses the same encoding as video-world domains.
        from latplan.util.paths import find_dataset
        from strips import bboxes_to_onehot
        path = find_dataset(args.track + ".npz")
        with np.load(path) as data:
            images_all = data["images"].astype(np.float32) / 256
            bboxes_all = data["bboxes"]
            picsize    = data["picsize"]
        picsize_grid = (picsize // 5).astype(int)
        Y, X = picsize_grid[0], picsize_grid[1]
        N = min(args.num, images_all.shape[0])
        idx = np.linspace(0, images_all.shape[0] - 1, N).astype(int)
        images_sub = images_all[idx]
        bboxes_sub = bboxes_all[idx]
        images_f      = preprocess(images_sub)
        bboxes_onehot = bboxes_to_onehot(bboxes_sub, X, Y)
        sample = np.concatenate(
            (images_f.reshape((N, images_f.shape[1], -1)),
             bboxes_onehot.reshape((N, images_f.shape[1], -1))),
            axis=-1).astype(np.float32)
        render_fn, _ = ae.blocks_renderer()

    else:
        # video-world (vidvrd / actiongenome) — load via the domain build_dataset.
        if args.domain == "vidvrd":
            from latplan.puzzles.puzzle_vidvrd import build_dataset
        elif args.domain == "actiongenome":
            from latplan.domains.video.actiongenome import build_dataset
        else:
            sys.exit(f"ERROR: unknown --domain {args.domain!r}")
        from latplan.puzzles.puzzle_labeled_objects import PICSIZE
        from strips import bboxes_to_onehot

        # Honour the manifest: load the overfit npz directly if available; else
        # rebuild via build_dataset with video_id_filter so the recon grid shows
        # the exact data the model trained on.
        from latplan.util.cache import load_cached as _load_cached
        if npz_path and os.path.exists(npz_path):
            print(f"[recon] loading overfit npz {npz_path}")
            hit = _load_cached(npz_path)
            if hit is None:
                sys.exit(f"ERROR: cannot read npz_path {npz_path!r}")
            images, bboxes, per_state_names, frame_ids, _ = hit
        else:
            kw = dict(category_filter=category, fps=fps)
            if vid_id:
                kw["video_id_filter"] = vid_id
            print(f"[recon] rebuild via build_dataset kwargs={kw}")
            images, bboxes, per_state_names, frame_ids = build_dataset(**kw)
        picsize_grid = (np.array(PICSIZE) // 5).astype(int)
        Y, X = picsize_grid[0], picsize_grid[1]
        num_states, num_objs = images.shape[0], images.shape[1]
        print(f"[recon] dataset has {num_states} states; sampling {args.num} BEFORE preprocess to keep RAM bounded")
        N = min(args.num, num_states)
        idx = np.linspace(0, num_states - 1, N).astype(int)
        images_sub = images[idx]
        bboxes_sub = bboxes[idx]
        images_f      = preprocess(images_sub.astype(np.float32) / 256)
        bboxes_onehot = bboxes_to_onehot(bboxes_sub, X, Y)
        sample = np.concatenate(
            (images_f.reshape((N, num_objs, -1)),
             bboxes_onehot.reshape((N, num_objs, -1))),
            axis=-1).astype(np.float32)
        render_fn, _ = ae.blocks_renderer()

    recon_feat    = ae.autoencode(sample)
    inputs        = render_fn(sample)
    recons        = render_fn(recon_feat)
    diffs         = np.abs(inputs - recons)
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
