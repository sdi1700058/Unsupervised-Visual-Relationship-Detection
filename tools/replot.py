#!/usr/bin/env python3
"""tools/replot.py — re-run the ORIGINAL-AUTHOR visualisation suite
(`plot_autoencoding_image` + `show_summary`, both from strips.py) on a
saved model WITHOUT retraining.

Calls the same `ae.plot_render` / `ae.plot_pos_neg` / `ae.plot_pn_decisiontree`
methods that the training run invokes at the end. Lands the following PNGs
in `<model_dir>/`:

    render_test.png             render_train.png
    render_test_shuffled.png    render_train_shuffled.png
    booleans_test.png           test*.pdf / test*.gv   (decision-tree)

Per CLAUDE.md "Use original author code when possible".

Usage:
    python3 tools/replot.py <model_dir>                        # auto-detect domain
    python3 tools/replot.py <model_dir> --num 200              # bigger sample
    python3 tools/replot.py <model_dir> --domain puzzle --type mnist --width 3 --height 3
    python3 tools/replot.py <model_dir> --domain blocks --track blocks-5-3
"""

import argparse
import json
import os
import sys
import numpy as np

import keras.optimizers
from keras_adabound import AdaBound
from keras_radam   import RAdam
setattr(keras.optimizers, "radam",    RAdam)
setattr(keras.optimizers, "adabound", AdaBound)

import latplan
import latplan.model
from latplan.puzzles.util import preprocess
from strips import bboxes_to_onehot, plot_autoencoding_image, show_summary


def _auto_domain(model_dir):
    if "actiongenome" in model_dir: return "actiongenome"
    if "vidvrd" in model_dir or "video-" in model_dir: return "vidvrd"
    if "blocks" in model_dir: return "blocks"
    return "puzzle"


def _load_video(model_dir, domain, num):
    if domain == "vidvrd":
        from latplan.puzzles.puzzle_vidvrd import build_dataset
    else:
        from latplan.domains.video.actiongenome import build_dataset
    from latplan.puzzles.puzzle_labeled_objects import PICSIZE

    manifest = json.load(open(os.path.join(model_dir, "loaded_videos.json")))
    cat, fps = manifest["category_filter"], manifest.get("fps", 3)
    print(f"[replot] {domain} category={cat!r} fps={fps!r}")
    images, bboxes, names, frame_ids = build_dataset(category_filter=cat, fps=fps)

    N = min(num, images.shape[0])
    rng = np.random.RandomState(0)
    idx = rng.choice(images.shape[0], N, replace=False)
    images, bboxes = images[idx], bboxes[idx]

    picsize_grid = (np.array(PICSIZE) // 5).astype(int)
    Y, X = picsize_grid[0], picsize_grid[1]
    images_f      = preprocess(images.astype(np.float32) / 256)
    bboxes_onehot = bboxes_to_onehot(bboxes, X, Y)
    states = np.concatenate(
        (images_f.reshape((N, images_f.shape[1], -1)),
         bboxes_onehot.reshape((N, images_f.shape[1], -1))),
        axis=-1).astype(np.float32)
    return states, "blocks"


def _load_blocks(track, num):
    from latplan.util.paths import find_dataset
    path = find_dataset(track + ".npz")
    with np.load(path) as data:
        images = data["images"].astype(np.float32) / 256
        bboxes = data["bboxes"]
        picsize = data["picsize"]
    picsize_grid = (picsize // 5).astype(int)
    Y, X = picsize_grid[0], picsize_grid[1]
    N = min(num, images.shape[0])
    rng = np.random.RandomState(0)
    idx = rng.choice(images.shape[0], N, replace=False)
    images = images[idx]
    bboxes = bboxes[idx]
    images_f      = preprocess(images)
    bboxes_onehot = bboxes_to_onehot(bboxes, X, Y)
    states = np.concatenate(
        (images_f.reshape((N, images_f.shape[1], -1)),
         bboxes_onehot.reshape((N, images_f.shape[1], -1))),
        axis=-1).astype(np.float32)
    return states, "blocks"


def _load_puzzle(ptype, width, height, num):
    from latplan.util.paths import find_dataset
    import importlib
    p = importlib.import_module(f"latplan.puzzles.puzzle_{ptype}")
    p.setup()
    path = find_dataset(f"puzzle-{ptype}-{width}-{height}.npz")
    with np.load(path) as data:
        configs = data["pres"]
    objects = p.to_objects(configs, width, height, False)
    N = min(num, objects.shape[0])
    rng = np.random.RandomState(0)
    idx = rng.choice(objects.shape[0], N, replace=False)
    return objects[idx].astype(np.float32), "puzzle"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("--num",   type=int, default=200, help="sample size for replot (subset before preprocess)")
    p.add_argument("--domain", default=None, choices=[None, "vidvrd", "actiongenome", "blocks", "puzzle"])
    p.add_argument("--type",  default="mnist")
    p.add_argument("--width", type=int, default=3)
    p.add_argument("--height", type=int, default=3)
    p.add_argument("--track", default="blocks-5-3")
    args = p.parse_args()

    if args.domain is None:
        args.domain = _auto_domain(args.model_dir)
    print(f"[replot] domain={args.domain}  model_dir={args.model_dir}")

    if args.domain in ("vidvrd", "actiongenome"):
        states, plotmode = _load_video(args.model_dir, args.domain, args.num)
    elif args.domain == "blocks":
        states, plotmode = _load_blocks(args.track, args.num)
    else:
        states, plotmode = _load_puzzle(args.type, args.width, args.height, args.num)

    print(f"[replot] sample shape={states.shape}  plotmode={plotmode!r}")

    print(f"[replot] loading model from {args.model_dir}")
    ae = latplan.model.load(args.model_dir)
    ae.load()
    print(f"[replot] model loaded — U={ae.parameters['U']} P={ae.parameters['P']} A={ae.parameters['A']}")

    N = states.shape[0]
    split = max(1, int(N * 0.9))
    train, test = states[:split], states[split:]

    show_summary(ae, train, test)
    plot_autoencoding_image(ae, test, train, plotmode)

    print(f"[replot] DONE. Output PNGs in {args.model_dir}/")


if __name__ == "__main__":
    main()
