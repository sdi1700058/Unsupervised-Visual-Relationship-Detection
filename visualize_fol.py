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


def load_labeled_objects_data_and_renderer(model_dir, num=10,
                                           dataset_path=None, images_dir=None):
    """Load labeled-objects data and build a patch-grid renderer.

    The renderer arranges object patches (16x16 greyscale) into a small grid
    so the state is humanly interpretable even without a spatial canvas.

    Returns: states, object_names (flat, from first state), render_fn,
             per_state_names (list-of-lists)
    """
    from latplan.puzzles.puzzle_labeled_objects import (
        build_dataset, PATCH_SIZE, MAX_OBJECTS)
    import json as _json

    states, per_state_names, image_ids = build_dataset(
        dataset_path=dataset_path, images_dir=images_dir)

    # Try to load saved names that match training order
    names_path = os.path.join(model_dir, "object_names.json")
    if os.path.isfile(names_path):
        with open(names_path) as f:
            saved = _json.load(f)
        per_state_names = saved["object_names"]
        image_ids       = saved["image_ids"]

    states          = states[:num]
    per_state_names = per_state_names[:num]
    flat_names      = per_state_names[0] if per_state_names else []

    patch_dim = PATCH_SIZE * PATCH_SIZE           # 256 grey pixels per object

    def render_fn(x):
        """x : (batch, num_objs, feature_dim) -> (batch, grid_H, grid_W)"""
        batch, num_objs, _ = x.shape
        cols = min(4, num_objs)
        rows = (num_objs + cols - 1) // cols
        grid_H = rows * PATCH_SIZE
        grid_W = cols * PATCH_SIZE
        out = np.zeros((batch, grid_H, grid_W), dtype=np.float32)
        for b in range(batch):
            for o in range(num_objs):
                patch = x[b, o, :patch_dim].reshape(PATCH_SIZE, PATCH_SIZE)
                r, c  = o // cols, o % cols
                out[b,
                    r * PATCH_SIZE:(r + 1) * PATCH_SIZE,
                    c * PATCH_SIZE:(c + 1) * PATCH_SIZE] = patch
        return out

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
    """Create a single visualization for one state.

    Layout:
    - Left: original rendered image + reconstruction
    - Middle: attention heatmaps per predicate unit
    - Right: FOL predicate text
    """
    U = ae.parameters['U']
    P = ae.parameters['P']
    A = ae.parameters['A']
    num_objs = state.shape[0]

    x = state[np.newaxis]  # (1, num_objs, num_features)

    # Get model outputs
    attention = ae.encode_attention(x)[0]  # (U, A, num_objs)
    latent = ae.encode(x)[0]               # (U*P,)
    recon = ae.autoencode(x)               # (1, num_objs, num_features)

    # Render images
    x_rendered = render_fn(x)[0]
    y_rendered = render_fn(recon)[0]

    # Extract FOL
    from latplan.util.fol import extract_fol_from_model
    results = extract_fol_from_model(ae, x, object_names=object_names,
                                     confidence_threshold=confidence_threshold)
    fol_text = format_fol_state(results[0], show_negated=False, show_confidence=False)

    # --- Figure layout ---
    # --- Image 1: Reconstruction, Original, Difference, Latent Code ---
    fig1 = plt.figure(figsize=(8, 4))
    gs1 = gridspec.GridSpec(2, 2, width_ratios=[1, 1])

    # Original
    ax_orig = fig1.add_subplot(gs1[0, 0])
    if x_rendered.ndim == 2:
        ax_orig.imshow(x_rendered, cmap='gray')
    else:
        ax_orig.imshow(np.clip(x_rendered, 0, 1))
    ax_orig.set_title("Original", fontsize=9)
    ax_orig.axis('off')

    # Reconstruction
    ax_recon = fig1.add_subplot(gs1[1, 0])
    if y_rendered.ndim == 2:
        ax_recon.imshow(y_rendered, cmap='gray')
    else:
        ax_recon.imshow(np.clip(y_rendered, 0, 1))
    ax_recon.set_title("Reconstruction", fontsize=9)
    ax_recon.axis('off')

    # Difference
    ax_diff = fig1.add_subplot(gs1[0, 1])
    diff = np.abs(y_rendered - x_rendered)
    if diff.ndim == 2:
        ax_diff.imshow(diff, cmap='hot')
    else:
        ax_diff.imshow(np.clip(diff, 0, 1))
    ax_diff.set_title("Difference", fontsize=9)
    ax_diff.axis('off')

    # Latent code
    ax_latent = fig1.add_subplot(gs1[1, 1])
    latent_2d = latent.round().reshape(U, P)
    ax_latent.imshow(latent_2d, cmap='Greys', vmin=0, vmax=1, aspect='auto')
    ax_latent.set_xlabel("Predicates (P)", fontsize=7)
    ax_latent.set_ylabel("Units (U)", fontsize=7)
    ax_latent.set_title("Latent Code", fontsize=9)
    ax_latent.tick_params(labelsize=6)

    fig1.suptitle(f"FOSAE Reconstruction  - State {state_idx}", fontsize=11, y=0.98)
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    filepath1 = os.path.join(output_dir, f"fol_state_{state_idx}_recon.png")
    fig1.savefig(filepath1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Saved {filepath1}")

    # --- Image 2: Attention Maps ---
    n_attention_cols = min(U, 5)
    fig2 = plt.figure(figsize=(2.5 * n_attention_cols, 4))
    gs2 = gridspec.GridSpec(1, n_attention_cols)
    for u_idx in range(n_attention_cols):
        ax = fig2.add_subplot(gs2[0, u_idx])
        att = attention[u_idx]
        bindings = np.argmax(att, axis=-1)
        preds = latent.round().reshape(U, P)[u_idx]
        plot_attention_heatmap(
            ax, att,
            title=f"Unit {u_idx}\nbinds: {list(bindings)}\npreds: {preds.astype(int).tolist()}",
            object_names=object_names)
    fig2.suptitle(f"FOSAE Attention Maps  - State {state_idx}", fontsize=11, y=0.98)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    filepath2 = os.path.join(output_dir, f"fol_state_{state_idx}_attention.png")
    fig2.savefig(filepath2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved {filepath2}")

    # --- Image 3: FOL Predicate List ---
    fig3 = plt.figure(figsize=(6, 4))
    ax_fol = fig3.add_subplot(111)
    ax_fol.axis('off')
    ax_fol.text(
        0.0, 1.0, fol_text,
        transform=ax_fol.transAxes,
        fontsize=6, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    fig3.suptitle(f"FOSAE FOL Predicates  - State {state_idx}", fontsize=11, y=0.98)
    fig3.tight_layout(rect=[0, 0, 1, 0.95])
    filepath3 = os.path.join(output_dir, f"fol_state_{state_idx}_predicates.png")
    fig3.savefig(filepath3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"  Saved {filepath3}")


def visualize_attention_overview(ae, data, object_names, output_dir, num_states=10):
    """Create an overview plot of all attention patterns across states."""
    U = ae.parameters['U']
    P = ae.parameters['P']
    A = ae.parameters['A']

    attention = ae.encode_attention(data[:num_states])  # (N, U, A, num_objs)
    latent = ae.encode(data[:num_states])                # (N, U*P)

    # Average attention across states
    avg_attention = attention.mean(axis=0)  # (U, A, num_objs)

    fig, axes = plt.subplots(2, (U+1)//2, figsize=(3*(U+1)//2, 6))
    axes = axes.flatten()

    for u in range(U):
        ax = axes[u]
        im = ax.imshow(avg_attention[u], cmap='Blues', vmin=0, vmax=1, aspect='auto')
        ax.set_title(f"Unit {u}", fontsize=9)
        ax.set_yticks(range(A))
        ax.set_yticklabels([f"a{a}" for a in range(A)], fontsize=7)
        ax.set_xticks(range(len(object_names)))
        ax.set_xticklabels(object_names, fontsize=5, rotation=45, ha='right')

    # Hide extra axes
    for idx in range(U, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f"Average Attention Patterns (over {num_states} states)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    filepath = os.path.join(output_dir, "attention_overview.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize FOL predicates from trained FOSAE")
    parser.add_argument("model_dir", help="Path to trained model directory")
    parser.add_argument("--domain", default="puzzle",
                        choices=["puzzle", "blocks", "labeled_objects"])
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
                args.model_dir, args.num,
                dataset_path=args.dataset_path,
                images_dir=args.images_dir)
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
