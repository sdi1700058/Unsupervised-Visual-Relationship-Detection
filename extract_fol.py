#!/usr/bin/env python3
"""
Extract FOL predicates from a trained FOSAE model.

Usage:
  python extract_fol.py <model_dir> [--domain puzzle|blocks|labeled_objects] [--num N] [--output DIR]

Examples:
  # Extract from smoke test model
  python extract_fol.py out/_smoke_mnist --domain puzzle --num 20

  # Extract from a full training run
  python extract_fol.py out/puzzle_FirstOrderSAE_mnist_3_3_None_None_None_5000 --domain puzzle

  # Extract from blocksworld
  python extract_fol.py out/blocks-5-3/blocksworld_FirstOrderSAE_blocks-5-3_None_None_None_10000_BCE5 --domain blocks --track blocks-5-3

  # Extract from labeled-objects (VLM-annotated COCO images)
  python extract_fol.py out/labeled_objects/_smoke --domain labeled_objects
"""

import config
import numpy as np
import latplan
import latplan.model
from latplan.util.fol import (
    extract_fol_from_model,
    format_fol_state,
    save_fol_to_json,
    save_fol_to_text,
    analyze_predicate_usage,
)
import os, sys, argparse, json

# --- Canonical directories ---
from latplan.util.paths import DATA_DIR, find_dataset as _find_dataset


def load_puzzle_data(type='mnist', width=3, height=3, num_examples=None):
    """Load puzzle data and convert to object representation."""
    import importlib
    p = importlib.import_module(f'latplan.puzzles.puzzle_{type}')
    p.setup()
    path = _find_dataset(f"puzzle-{type}-{width}-{height}.npz")
    with np.load(path) as data:
        configs = data['pres'][:num_examples] if num_examples else data['pres']

    objects = p.to_objects(configs, width, height, False)

    # Build object names for puzzle tiles
    object_names = [f"tile_{i}" for i in range(width * height)]

    return objects, configs, object_names


def load_blocks_data(track="blocks-5-3", num_examples=None):
    """Load blocksworld data and convert to object representation."""
    from latplan.puzzles.util import preprocess

    path = _find_dataset(track + ".npz")
    with np.load(path) as data:
        images = data['images'].astype(np.float32) / 256
        bboxes = data['bboxes']
        all_transitions_idx = data['transitions']
        picsize = data['picsize']

    patchsize = images.shape[2:]
    picsize_grid = (picsize // 5).astype(int)
    Y, X = picsize_grid[0], picsize_grid[1]
    num_states, num_objs = bboxes.shape[0:2]

    images = preprocess(images)

    from strips import bboxes_to_onehot
    bboxes_onehot = bboxes_to_onehot(bboxes, X, Y)
    all_states = np.concatenate(
        (images.reshape((num_states, num_objs, -1)),
         bboxes_onehot.reshape((num_states, num_objs, -1))),
        axis=-1
    )

    if num_examples and num_examples < num_states:
        idx = np.random.choice(num_states, num_examples, replace=False)
        states = all_states[idx]
    else:
        states = all_states

    object_names = [f"block_{i}" for i in range(num_objs)]
    return states, object_names


def load_labeled_objects_data(model_dir, dataset_path=None, images_dir=None,
                               num_examples=None):
    """Load the VLM-annotated COCO labeled-objects dataset.

    Returns
    -------
    states            : (N, num_objs, feature_dim) float32 array
    per_state_names   : list[list[str]]  - semantic object labels per state
                        Pass directly as object_names to extract_fol_from_model;
                        it will be detected as per-state and used accordingly.
    image_ids         : list[str] COCO filenames, one per state

    Object names saved during training (object_names.json in model_dir) are
    loaded when available and used instead of rebuilding them from the dataset.
    """
    from latplan.puzzles.puzzle_labeled_objects import build_dataset

    states, per_state_names, image_ids = build_dataset(
        dataset_path=dataset_path, images_dir=images_dir)

    # Try to load object names that were saved during the training run so that
    # state ordering and names exactly match what the model was trained on.
    names_path = os.path.join(model_dir, "object_names.json")
    if os.path.isfile(names_path):
        with open(names_path) as f:
            saved = json.load(f)
        per_state_names = saved["object_names"]
        image_ids       = saved["image_ids"]
        print(f"Loaded object names from {names_path}")

    if num_examples and num_examples < len(states):
        states          = states[:num_examples]
        per_state_names = per_state_names[:num_examples]
        image_ids       = image_ids[:num_examples]

    return states, per_state_names, image_ids


def main():
    parser = argparse.ArgumentParser(
        description="Extract FOL predicates from a trained FOSAE model")
    parser.add_argument("model_dir", help="Path to the trained model directory")
    parser.add_argument("--domain", default="puzzle",
                        choices=["puzzle", "blocks", "labeled_objects"],
                        help="Domain type (default: puzzle)")
    parser.add_argument("--type", default="mnist",
                        help="Puzzle type for puzzle domain (default: mnist)")
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--height", type=int, default=3)
    parser.add_argument("--track", default="blocks-5-3",
                        help="Track name for blocks domain")
    parser.add_argument("--dataset-path", default=None,
                        help="Path to fosae_labeled_dataset_unsloth.json "
                             "(labeled_objects domain only; default: auto)")
    parser.add_argument("--images-dir", default=None,
                        help="Path to raw_images/ directory "
                             "(labeled_objects domain only; default: auto)")
    parser.add_argument("--num", type=int, default=50,
                        help="Number of states to extract (default: 50)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <model_dir>/fol_output)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Min attention confidence threshold (default: 0.5)")
    parser.add_argument("--show-negated", action="store_true",
                        help="Include negated predicates in text output")
    args = parser.parse_args()

    # Setup Keras optimizers
    from keras.optimizers import Adam
    from keras_adabound import AdaBound
    from keras_radam import RAdam
    import keras.optimizers
    setattr(keras.optimizers, "radam", RAdam)
    setattr(keras.optimizers, "adabound", AdaBound)

    # --- Load model ---
    print(f"Loading model from {args.model_dir}...")
    ae = latplan.model.load(args.model_dir)
    ae.load()
    U = ae.parameters['U']
    P = ae.parameters['P']
    A = ae.parameters['A']
    print(f"Model loaded: U={U}, A={A}, P={P} -> {U*P} total latent bits")

    # --- Load data ---
    print(f"Loading {args.domain} data...")
    image_ids = None
    if args.domain == "puzzle":
        data, configs, object_names = load_puzzle_data(
            args.type, args.width, args.height, args.num)
    elif args.domain == "blocks":
        data, object_names = load_blocks_data(args.track, args.num)
    elif args.domain == "labeled_objects":
        data, object_names, image_ids = load_labeled_objects_data(
            args.model_dir,
            dataset_path=args.dataset_path,
            images_dir=args.images_dir,
            num_examples=args.num,
        )
        print(f"  Per-state object names loaded for {len(object_names)} states")
    else:
        raise ValueError(f"Unknown domain: {args.domain}")

    print(f"Data shape: {data.shape}")

    # --- Extract FOL ---
    print("Extracting FOL predicates...")
    results = extract_fol_from_model(
        ae, data,
        object_names=object_names,
        confidence_threshold=args.confidence,
    )

    # --- Print sample ---
    print("\n" + "="*60)
    print("SAMPLE FOL REPRESENTATIONS")
    print("="*60)
    for result in results[:min(5, len(results))]:
        print()
        print(format_fol_state(result, show_negated=args.show_negated))

    # --- Analyze predicate usage ---
    summary = analyze_predicate_usage(results, U, P)
    print("\n" + "="*60)
    print("PREDICATE USAGE ANALYSIS")
    print("="*60)
    print(f"Total states analyzed: {summary['total_states']}")
    print(f"Always-on predicates:  {summary['num_always_on']}")
    print(f"Always-off predicates: {summary['num_always_off']}")
    print(f"Variable predicates:   {summary['num_variable']} (these carry information)")
    if summary['variable']:
        print("\nVariable predicates (activation rates):")
        for v in sorted(summary['variable'], key=lambda x: -x['activation_rate']):
            print(f"  Unit {v['unit']}, Pred {v['predicate']}: "
                  f"{v['activation_rate']*100:.1f}% active")

    # --- Save outputs ---
    output_dir = args.output or os.path.join(args.model_dir, "fol_output")
    os.makedirs(output_dir, exist_ok=True)

    save_fol_to_json(results,
                     os.path.join(output_dir, "fol_predicates.json"),
                     parameters={**ae.parameters, 'num_objs': data.shape[1]})
    save_fol_to_text(results,
                     os.path.join(output_dir, "fol_predicates.txt"),
                     show_negated=args.show_negated)

    # For labeled_objects, also save a state->image mapping so results can be
    # correlated back to the original COCO image files.
    if image_ids is not None:
        mapping = [{"state_idx": i, "image": image_ids[i],
                    "object_names": results[i].get("object_names", [])}
                   for i in range(len(results))]
        with open(os.path.join(output_dir, "state_image_mapping.json"), 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"State->image mapping saved to {output_dir}/state_image_mapping.json")

    with open(os.path.join(output_dir, "predicate_analysis.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Predicate analysis saved to {output_dir}/predicate_analysis.json")

    print(f"\n FOL extraction complete. Output in {output_dir}/")


if __name__ == '__main__':
    main()
