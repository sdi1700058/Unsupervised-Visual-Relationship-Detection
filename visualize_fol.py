#!/usr/bin/env python3
"""
Visualize FOL predicates from a trained FOSAE model.

Generates side-by-side plots showing:
1. Original state (rendered as image)
2. Reconstruction
3. Attention maps (which objects are bound to which predicate units)
4. FOL predicate text overlay

Usage:
  python visualize_fol.py <model_dir> [--domain puzzle|blocks|labeled_objects] [--num N]

Examples:
  python visualize_fol.py out/_smoke_mnist           --domain puzzle --num 4
  python visualize_fol.py out/_smoke_labeled_objects --domain labeled_objects --num 6
"""

import config
import numpy as np
import latplan
import latplan.model
from latplan.util.fol import extract_fol_from_model, format_fol_state
import os, sys, argparse

# --- Canonical directories ---
from latplan.util.paths import DATA_DIR, find_dataset as _find_dataset

# Must be set before matplotlib import in some environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_puzzle_data_and_renderer(ae, type='mnist', width=3, height=3, num=10):
    """Load puzzle data and build renderer."""
    import importlib
    p = importlib.import_module(f'latplan.puzzles.puzzle_{type}')
    p.setup()
    path = _find_dataset(f"puzzle-{type}-{width}-{height}.npz")
    with np.load(path) as data:
        configs = data['pres'][:num]
    objects = p.to_objects(configs, width, height, False)
    object_names = [f"tile_{i}" for i in range(width * height)]

    render, _ = ae.puzzle_renderer()
    return objects, object_names, render


def load_blocks_data_and_renderer(ae, track="blocks-5-3", num=10):
    """Load blocks data and build renderer."""
    from latplan.puzzles.util import preprocess
    from strips import bboxes_to_onehot

    path = _find_dataset(track + ".npz")
    with np.load(path) as data:
        images = data['images'].astype(np.float32) / 256
        bboxes = data['bboxes']
        picsize = data['picsize']

    picsize_grid = (picsize // 5).astype(int)
    Y, X = picsize_grid[0], picsize_grid[1]
    num_states, num_objs = bboxes.shape[0:2]

    images = preprocess(images)
    bboxes_onehot = bboxes_to_onehot(bboxes, X, Y)
    all_states = np.concatenate(
        (images.reshape((num_states, num_objs, -1)),
         bboxes_onehot.reshape((num_states, num_objs, -1))),
        axis=-1
    )

    states = all_states[:num]
    object_names = [f"block_{i}" for i in range(num_objs)]
    render, _ = ae.blocks_renderer()
    return states, object_names, render


def load_labeled_objects_data_and_renderer(ae, model_dir, num=10,
                                           dataset_path=None, images_dir=None):
    """Load labeled-objects data using the same blocks-domain encoding used at
    training time, and return the blocks_renderer for visualization.

    Returns: states, object_names (flat, from first state), render_fn,
             per_state_names (list-of-lists)
    """
    from latplan.puzzles.puzzle_labeled_objects import (
        build_dataset, PICSIZE)
    from latplan.puzzles.util import preprocess
    from strips import bboxes_to_onehot
    import json as _json

    images, bboxes, per_state_names, image_ids = build_dataset(
        dataset_path=dataset_path, images_dir=images_dir)

    # Try to load saved names that match training order
    names_path = os.path.join(model_dir, "object_names.json")
    if os.path.isfile(names_path):
        with open(names_path) as f:
            saved = _json.load(f)
        per_state_names = saved["object_names"]
        image_ids       = saved["image_ids"]

    # Apply the same preprocessing pipeline as strips.py labeled_objects()
    picsize_grid = (np.array(PICSIZE) // 5).astype(int)
    Y, X = picsize_grid[0], picsize_grid[1]
    num_states, num_objs = images.shape[0], images.shape[1]

    images = images.astype(np.float32) / 256
    images = preprocess(images)
    bboxes_onehot = bboxes_to_onehot(bboxes, X, Y)
    states = np.concatenate(
        (images.reshape   ((num_states, num_objs, -1)),
         bboxes_onehot.reshape((num_states, num_objs, -1))),
        axis=-1)

    states          = states[:num]
    per_state_names = per_state_names[:num]
    flat_names      = per_state_names[0] if per_state_names else []

    render_fn, _ = ae.blocks_renderer()

    return states, flat_names, render_fn, per_state_names


def plot_attention_heatmap(ax, attention_matrix, title="", object_names=None):
    """Plot attention weights as a heatmap.

    Args:
        ax: matplotlib axes
        attention_matrix: (A, num_objs)  - attention for one predicate unit
        title: axis title
        object_names: labels for x-axis
    """
    A, num_objs = attention_matrix.shape
    im = ax.imshow(attention_matrix, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.set_yticks(range(A))
    ax.set_yticklabels([f"arg_{a}" for a in range(A)], fontsize=7)
    if object_names:
        ax.set_xticks(range(num_objs))
        ax.set_xticklabels(object_names, fontsize=5, rotation=45, ha='right')
    ax.set_title(title, fontsize=7)
    return im


def visualize_single_state(ae, state, render_fn, object_names, state_idx,
                           output_dir, confidence_threshold=0.5):
    """Per-state figures — SPEC §V6 / D2-D4: ONE concept per PNG, w/ caption.md.

    Emits (per state i, given U predicate units):
      recon_<i>_original.png
      recon_<i>_reconstruction.png
      recon_<i>_diff.png
      latent_<i>.png
      attention_<i>_unit<u>.png   (× U)
      fol_<i>.png
    Each is accompanied by a `<name>.caption.md` sidecar."""
    from viz.io import save_with_caption
    from latplan.util.fol import extract_fol_from_model

    U = ae.parameters['U']
    P = ae.parameters['P']
    A = ae.parameters['A']

    x         = state[np.newaxis]            # (1, num_objs, F)
    attention = ae.encode_attention(x)[0]    # (U, A, num_objs)
    latent    = ae.encode(x)[0]              # (U*P,)
    recon     = ae.autoencode(x)             # (1, num_objs, F)

    x_rendered = render_fn(x)[0]
    y_rendered = render_fn(recon)[0]

    # D2.1 — original
    fig, ax = plt.subplots(figsize=(4, 3))
    if x_rendered.ndim == 2: ax.imshow(x_rendered, cmap='gray')
    else:                    ax.imshow(np.clip(x_rendered, 0, 1))
    ax.axis('off'); ax.set_title(f"State {state_idx}: input scene")
    save_with_caption(fig, os.path.join(output_dir, f"recon_{state_idx}_original"),
        what=f"Object patches placed on the FOSAE canvas at their ground-truth bbox positions for input state #{state_idx}.",
        why="Side-by-side with the reconstruction this is the most direct evidence FOSAE 'sees' the world: did the autoencoder preserve spatial layout and per-object appearance?",
        how_to_read="If your domain is image-based each tracked object's silhouette should be recognisable; the canvas margin is the decoded-bbox geometry, NOT the original photo background.")

    # D2.2 — reconstruction
    fig, ax = plt.subplots(figsize=(4, 3))
    if y_rendered.ndim == 2: ax.imshow(y_rendered, cmap='gray')
    else:                    ax.imshow(np.clip(y_rendered, 0, 1))
    ax.axis('off'); ax.set_title(f"State {state_idx}: decoded reconstruction")
    save_with_caption(fig, os.path.join(output_dir, f"recon_{state_idx}_reconstruction"),
        what=f"Reconstruction of state #{state_idx} after a full encode → decode pass through FOSAE.",
        why="If reconstruction visibly degrades (blur, object loss, ghosting) the bottleneck isn't carrying enough information — usually fixed by raising U×P or the preencoder dim (cf. V10).",
        how_to_read=f"Compare against `recon_{state_idx}_original.png`. Look for: all objects present? colours/shapes recognisable? positions roughly correct?")

    # D2.3 — diff
    fig, ax = plt.subplots(figsize=(4, 3))
    diff = np.abs(y_rendered - x_rendered)
    if diff.ndim == 2: ax.imshow(diff, cmap='hot')
    else:              ax.imshow(np.clip(diff, 0, 1))
    ax.axis('off'); ax.set_title(f"State {state_idx}: |original - reconstruction|")
    save_with_caption(fig, os.path.join(output_dir, f"recon_{state_idx}_diff"),
        what=f"Per-pixel absolute difference between the input scene and the reconstruction for state #{state_idx}.",
        why="Bright regions = pixels the autoencoder is losing. Edge hotspots → bbox decoding errors; uniform background brightness → global appearance drift.",
        how_to_read="Hotter colour = larger reconstruction error. Mostly-dark image = good reconstruction.")

    # D2.4 — latent code (U × P heatmap)
    fig, ax = plt.subplots(figsize=(4, 3))
    latent_2d = latent.round().reshape(U, P)
    ax.imshow(latent_2d, cmap='Greys', vmin=0, vmax=1, aspect='auto')
    ax.set_xlabel("Predicates (P)"); ax.set_ylabel("Units (U)")
    ax.set_title(f"State {state_idx}: latent code (U×P bits)")
    save_with_caption(fig, os.path.join(output_dir, f"latent_{state_idx}"),
        what=f"Binarised (Gumbel-Softmax rounded) U×P latent code for state #{state_idx}. Each cell = one symbolic proposition.",
        why="The latent code IS the symbolic state. Visually-similar states should produce similar codes; structurally-different states should differ in many cells.",
        how_to_read="Black = predicate fires (True), white = doesn't fire (False). Diff adjacent states (e.g. `latent_0` vs `latent_1`) to see which predicates encode the transition.")

    # D3 — per-unit attention
    for u in range(U):
        att      = attention[u]
        bindings = np.argmax(att, axis=-1)
        preds    = latent_2d[u]
        fig, ax  = plt.subplots(figsize=(4, 3))
        ax.imshow(att, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        ax.set_yticks(range(A)); ax.set_yticklabels([f"arg_{a}" for a in range(A)])
        if object_names:
            ax.set_xticks(range(len(object_names)))
            ax.set_xticklabels(object_names, fontsize=7, rotation=45, ha='right')
        ax.set_title(f"State {state_idx}: unit {u} attention")
        save_with_caption(fig, os.path.join(output_dir, f"attention_{state_idx}_unit{u}"),
            what=f"Per-argument attention weights for predicate unit #{u} on state #{state_idx}. Rows = arity slots `arg_0..arg_{A-1}`; columns = candidate objects.",
            why=f"This is the predicate's grounding — which objects unit #{u} reads. Stable bindings across states ⇒ a learnt referent; thrashing bindings ⇒ unstable / collapsing predicate.",
            how_to_read=f"Deeper blue = higher attention. Argmax bindings this state: {list(bindings)}; firing pattern this state: {preds.astype(int).tolist()}. A meaningful predicate should pick out 1-2 objects per argument.")

    # D4 — FOL text card
    results  = extract_fol_from_model(ae, x, object_names=object_names,
                                      confidence_threshold=confidence_threshold)
    fol_text = format_fol_state(results[0], show_negated=False, show_confidence=False)
    fig, ax  = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    ax.text(0.0, 1.0, fol_text, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title(f"State {state_idx}: extracted FOL predicates")
    save_with_caption(fig, os.path.join(output_dir, f"fol_{state_idx}"),
        what=f"Human-readable first-order-logic transcript of the predicates firing on state #{state_idx} above the {confidence_threshold:.2f} confidence threshold.",
        why="This is the symbolic interpretation of the state. Empty cards or near-identical cards across distinct states ⇒ the FOSAE collapsed to a near-constant code (V10 P-budget too tight).",
        how_to_read="Each line `predX(obj_a, obj_b, …)` + confidence score. Compare across adjacent states to see which predicates encode the transition.")


def visualize_attention_overview(ae, data, object_names, output_dir, num_states=10):
    """Per-domain summaries — SPEC §V6 / D4-D5.

    Emits:
      attention_avg_unit<u>.png      (× U)   — replaces legacy multi-panel grid
      summary_recon_mse.png
      summary_latent_entropy.png
    """
    from viz.io import save_with_caption

    U = ae.parameters['U']
    P = ae.parameters['P']
    A = ae.parameters['A']

    attention = ae.encode_attention(data[:num_states])      # (N, U, A, num_objs)
    avg_att   = attention.mean(axis=0)                      # (U, A, num_objs)

    # D5 — one figure per unit (replaces attention_overview.png grid)
    for u in range(U):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(avg_att[u], cmap='Blues', vmin=0, vmax=1, aspect='auto')
        ax.set_yticks(range(A)); ax.set_yticklabels([f"arg_{a}" for a in range(A)])
        if object_names:
            ax.set_xticks(range(len(object_names)))
            ax.set_xticklabels(object_names, fontsize=7, rotation=45, ha='right')
        ax.set_title(f"Unit {u}: mean attention over {num_states} states")
        save_with_caption(fig, os.path.join(output_dir, f"attention_avg_unit{u}"),
            what=f"Mean attention weights for predicate unit #{u} averaged across the first {num_states} test states.",
            why=f"Per-state attention can drift; averaging reveals whether unit #{u} has a stable referent. Concentrated column ⇒ a learnt object binding; uniform ⇒ unit unused / collapsed.",
            how_to_read="Look for one or two columns that dominate per row — that's the predicate's referent. Diffuse ⇒ raise P or refine the data.")

    # D4.1 — per-frame reconstruction MSE
    recon  = ae.autoencode(data[:num_states])
    diff   = (recon - data[:num_states]) ** 2
    while diff.ndim > 1:
        diff = diff.mean(axis=-1)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(diff, marker='o', linewidth=1)
    ax.set_xlabel("test-state index"); ax.set_ylabel("MSE")
    ax.set_title(f"Per-frame reconstruction MSE (first {num_states} test states)")
    ax.grid(True, alpha=0.3)
    save_with_caption(fig, os.path.join(output_dir, "summary_recon_mse"),
        what=f"Per-frame reconstruction MSE across the first {num_states} test states.",
        why="A flat low line is the success signal for the experimental thesis (FOSAE reconstructs the video world). Spikes localise failure cases; an upward trend ⇒ drift.",
        how_to_read="Y-axis = feature-space MSE. < 0.05 is good for our realistic-image domains; > 0.2 ⇒ collapse / under-training.")

    # D4.2 — per-unit latent binarisation entropy
    latent     = ae.encode(data[:num_states]).round().reshape((num_states, U, P))
    p_one      = latent.mean(axis=0)                                                # (U, P)
    eps        = 1e-8
    entropy    = -(p_one * np.log2(p_one + eps) + (1 - p_one) * np.log2(1 - p_one + eps))
    per_unit_H = entropy.mean(axis=1)
    fig, ax    = plt.subplots(figsize=(6, 3))
    ax.bar(range(U), per_unit_H)
    ax.set_xlabel("Predicate unit index (U)"); ax.set_ylabel("Mean binarisation entropy (bits)")
    ax.set_title(f"Per-unit latent entropy across {num_states} test states")
    ax.set_ylim(0, 1)
    save_with_caption(fig, os.path.join(output_dir, "summary_latent_entropy"),
        what=f"Binarisation entropy of each predicate unit across {num_states} test states, averaged over the P propositions in that unit.",
        why="Entropy 0 ⇒ the predicate always returns the same value (dead / collapsed unit). Entropy near 1 ⇒ healthy ~50/50 firing. A bar chart full of near-zeros is the V10 P-budget collapse signal.",
        how_to_read="Tall bars = informative units; short bars = dead. Aim for the majority of bars in [0.4, 0.9]. If most < 0.1, raise P or lower regularisation.")


def load_vidvrd_data_and_renderer(ae, model_dir, num=10,
                                   annotations_dir=None, frames_dir=None,
                                   category=None):
    """VidVRD analogue of load_labeled_objects_data_and_renderer."""
    from latplan.puzzles.puzzle_vidvrd import build_dataset, PICSIZE
    from latplan.puzzles.util import preprocess
    from strips import bboxes_to_onehot
    import json as _json

    manifest_path = os.path.join(model_dir, "loaded_videos.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as _f:
            _m = _json.load(_f)
        _trained_cat = _m.get("category_filter")
        if _trained_cat != category:
            raise SystemExit(
                f"ERROR: model_dir was trained with category={_trained_cat!r} "
                f"but --category={category!r} was passed. Either drop --category "
                f"or point at the matching out/video-{_trained_cat}/... model_dir.")

    images, bboxes, per_state_names, frame_ids = build_dataset(
        annotations_dir=annotations_dir, frames_dir=frames_dir,
        category_filter=category)

    names_path = os.path.join(model_dir, "object_names.json")
    if os.path.isfile(names_path):
        with open(names_path) as f:
            saved = _json.load(f)
        per_state_names = saved["object_names"]
        frame_ids       = saved.get("frame_ids", saved.get("image_ids", frame_ids))

    picsize_grid = (np.array(PICSIZE) // 5).astype(int)
    Y, X = picsize_grid[0], picsize_grid[1]
    num_states, num_objs = images.shape[0], images.shape[1]

    images = images.astype(np.float32) / 256
    images = preprocess(images)
    bboxes_onehot = bboxes_to_onehot(bboxes, X, Y)
    states = np.concatenate(
        (images.reshape   ((num_states, num_objs, -1)),
         bboxes_onehot.reshape((num_states, num_objs, -1))),
        axis=-1)

    states          = states[:num]
    per_state_names = per_state_names[:num]
    flat_names      = per_state_names[0] if per_state_names else []

    render_fn, _ = ae.blocks_renderer()
    return states, flat_names, render_fn, per_state_names


def load_actiongenome_data_and_renderer(ae, model_dir, num=10,
                                         annotations_dir=None, frames_dir=None,
                                         category=None):
    """ActionGenome analogue of load_vidvrd_data_and_renderer (C6)."""
    from latplan.domains.video.actiongenome import build_dataset
    from latplan.puzzles.puzzle_labeled_objects import PICSIZE
    from latplan.puzzles.util import preprocess
    from strips import bboxes_to_onehot
    import json as _json

    manifest_path = os.path.join(model_dir, "loaded_videos.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as _f:
            _m = _json.load(_f)
        _trained_cat = _m.get("category_filter")
        if _trained_cat != category:
            raise SystemExit(
                f"ERROR: model_dir trained with category={_trained_cat!r} "
                f"but --category={category!r} was passed. Drop --category "
                f"or point at out/video/actiongenome/{_trained_cat}/... model_dir.")

    images, bboxes, per_state_names, frame_ids = build_dataset(
        annotations_dir=annotations_dir, frames_dir=frames_dir,
        category_filter=category)

    names_path = os.path.join(model_dir, "object_names.json")
    if os.path.isfile(names_path):
        with open(names_path) as f:
            saved = _json.load(f)
        per_state_names = saved["object_names"]
        frame_ids       = saved.get("frame_ids", saved.get("image_ids", frame_ids))

    picsize_grid = (np.array(PICSIZE) // 5).astype(int)
    Y, X = picsize_grid[0], picsize_grid[1]
    num_states, num_objs = images.shape[0], images.shape[1]

    images = images.astype(np.float32) / 256
    images = preprocess(images)
    bboxes_onehot = bboxes_to_onehot(bboxes, X, Y)
    states = np.concatenate(
        (images.reshape   ((num_states, num_objs, -1)),
         bboxes_onehot.reshape((num_states, num_objs, -1))),
        axis=-1)

    states          = states[:num]
    per_state_names = per_state_names[:num]
    flat_names      = per_state_names[0] if per_state_names else []

    render_fn, _ = ae.blocks_renderer()
    return states, flat_names, render_fn, per_state_names


def main():
    parser = argparse.ArgumentParser(
        description="Visualize FOL predicates from trained FOSAE")
    parser.add_argument("model_dir", help="Path to trained model directory")
    parser.add_argument("--domain", default="puzzle",
                        choices=["puzzle", "blocks", "labeled_objects", "vidvrd", "actiongenome"])
    parser.add_argument("--type", default="mnist", help="Puzzle type (default: mnist)")
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--height", type=int, default=3)
    parser.add_argument("--track", default="blocks-5-3", help="Blocks track name")
    parser.add_argument("--num", type=int, default=4, help="Number of states to visualize")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--dataset-path", default=None,
                        help="Path to JSON dataset (labeled_objects only)")
    parser.add_argument("--images-dir", default=None,
                        help="Path to raw_images/ dir (labeled_objects only)")
    parser.add_argument("--annotations-dir", default=None,
                        help="VidVRD annotations dir (vidvrd domain; default: auto)")
    parser.add_argument("--frames-dir", default=None,
                        help="VidVRD frames dir (vidvrd domain; default: auto)")
    parser.add_argument("--category", default=None,
                        help="VidVRD category filter (vidvrd domain; default: all)")
    args = parser.parse_args()

    # Setup
    from keras.optimizers import Adam
    from keras_adabound import AdaBound
    from keras_radam import RAdam
    import keras.optimizers
    setattr(keras.optimizers, "radam", RAdam)
    setattr(keras.optimizers, "adabound", AdaBound)

    # Load model
    print(f"Loading model from {args.model_dir}...")
    ae = latplan.model.load(args.model_dir)
    ae.load()
    print(f"Model: U={ae.parameters['U']}, A={ae.parameters['A']}, P={ae.parameters['P']}")

    # Load data + renderer
    per_state_names = None
    if args.domain == "puzzle":
        data, obj_names, render_fn = load_puzzle_data_and_renderer(
            ae, args.type, args.width, args.height, args.num)
    elif args.domain == "labeled_objects":
        data, obj_names, render_fn, per_state_names = \
            load_labeled_objects_data_and_renderer(
                ae, args.model_dir, args.num,
                dataset_path=args.dataset_path,
                images_dir=args.images_dir)
    elif args.domain == "vidvrd":
        data, obj_names, render_fn, per_state_names = \
            load_vidvrd_data_and_renderer(
                ae, args.model_dir, args.num,
                annotations_dir=args.annotations_dir,
                frames_dir=args.frames_dir,
                category=args.category)
    elif args.domain == "actiongenome":
        data, obj_names, render_fn, per_state_names = \
            load_actiongenome_data_and_renderer(
                ae, args.model_dir, args.num,
                annotations_dir=args.annotations_dir,
                frames_dir=args.frames_dir,
                category=args.category)
    else:
        data, obj_names, render_fn = load_blocks_data_and_renderer(
            ae, args.track, args.num)

    output_dir = args.output or os.path.join(args.model_dir, "fol_visualizations")
    os.makedirs(output_dir, exist_ok=True)

    # Visualize individual states
    print("Generating per-state visualizations...")
    for i in range(min(args.num, len(data))):
        # For labeled_objects use per-state names; fall back to shared list
        state_names = per_state_names[i] if per_state_names is not None else obj_names
        visualize_single_state(ae, data[i], render_fn, state_names, i,
                               output_dir, confidence_threshold=args.confidence)

    # Attention overview  - use flat obj_names for axis labels
    print("Generating attention overview...")
    visualize_attention_overview(ae, data, obj_names, output_dir,
                                num_states=min(args.num, len(data)))

    print(f"\n Visualizations saved to {output_dir}/")


if __name__ == '__main__':
    main()
