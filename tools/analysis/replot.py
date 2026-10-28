#!/usr/bin/env python3
"""tools/analysis/replot.py — re-run the ORIGINAL-AUTHOR visualisation suite
(`plot_autoencoding_image` + `show_summary`, both from model/train/strips.py) on a
saved model WITHOUT retraining.

Calls the same `ae.plot_render` / `ae.plot_pos_neg` / `ae.plot_pn_decisiontree`
methods that the training run invokes at the end. Lands the following PNGs
in `<model_dir>/`:

    render_test.png             render_train.png
    render_test_shuffled.png    render_train_shuffled.png
    booleans_test.png           test*.pdf / test*.gv   (decision-tree)

Per the working rules: use the original author code when possible.

Usage:
    python3 tools/analysis/replot.py <model_dir>                        # auto-detect domain
    python3 tools/analysis/replot.py <model_dir> --num 200              # bigger sample
    python3 tools/analysis/replot.py <model_dir> --domain puzzle --type mnist --width 3 --height 3
    python3 tools/analysis/replot.py <model_dir> --domain blocks --track blocks-5-3
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

import model
import model.model
from model.loaders.image import preprocess
import model.train.strips as _strips
# plot_autoencoding_image short-circuits unless `strips.mode` contains "plot"
# (upstream gating). Force it on so this replot script does its job.
_strips.mode = "plot"
from model.train.strips import bboxes_to_onehot, plot_autoencoding_image, show_summary


def _auto_domain(model_dir):
    if "vidvrd" in model_dir or "video-" in model_dir: return "vidvrd"
    if "blocks" in model_dir: return "blocks"
    return "puzzle"


def _load_video(model_dir, domain, num):
    from model.loaders.vidvrd import build_dataset
    from model.loaders.labeled_objects import PICSIZE
    from model.loaders.cache import load_cached

    manifest = json.load(open(os.path.join(model_dir, "loaded_videos.json")))
    cat       = manifest.get("category_filter")
    fps       = manifest.get("fps", 3)
    vid_id    = manifest.get("video_id_filter") or manifest.get("video_id")
    npz_path  = manifest.get("npz_path")
    print(f"[replot] {domain} category={cat!r} fps={fps!r} video_id={vid_id!r}")

    # 1) If an overfit npz was used at training time, read it directly so the
    #    plot reflects EXACTLY the data the model saw (no other-video drift).
    if npz_path and os.path.exists(npz_path):
        print(f"[replot] loading overfit npz {npz_path}")
        hit = load_cached(npz_path)
        if hit is not None:
            images, bboxes, names, frame_ids, _ = hit
            print(f"[replot] npz states={len(images)}")
        else:
            images = bboxes = None
    else:
        images = bboxes = None

    # 2) Otherwise rebuild via build_dataset, honouring video_id_filter so we
    #    don't fall back to the full category cache (which would render frames
    #    the model never saw).
    if images is None:
        kwargs = dict(category_filter=cat, fps=fps)
        if vid_id:
            kwargs["video_id_filter"] = vid_id
        images, bboxes, names, frame_ids = build_dataset(**kwargs)
        print(f"[replot] rebuilt {len(images)} states from build_dataset")

    # A run trained with SPLIT=clip names its val and test clips in the
    # manifest. Its test pictures then come from the test clips only, so the
    # render_test.png the author looks at shows clips the model never saw.
    # Without that record, any state of the npz may land in either set.
    split_info = manifest.get("split_info") or {}
    rng = np.random.RandomState(0)
    n_train = None
    if split_info.get("test_clips"):
        from model.loaders.split import frames_of
        test_pool = frames_of(frame_ids, split_info["test_clips"])
        held = set(frames_of(frame_ids, list(split_info.get("val_clips") or [])
                             + list(split_info["test_clips"])))
        train_pool = [k for k in range(images.shape[0]) if k not in held]
        n_test = min(len(test_pool), max(1, num // 10))
        n_train = min(len(train_pool), num - n_test)
        idx = np.concatenate([rng.choice(train_pool, n_train, replace=False),
                              rng.choice(test_pool, n_test, replace=False)])
        print(f"[replot] SPLIT=clip: {n_train} training states, {n_test} "
              f"states from the {len(split_info['test_clips'])} test clips")
    else:
        idx = rng.choice(images.shape[0], min(num, images.shape[0]),
                         replace=False)
    N = len(idx)
    images, bboxes = images[idx], bboxes[idx]

    picsize_grid = (np.array(PICSIZE) // 5).astype(int)
    Y, X = picsize_grid[0], picsize_grid[1]
    images_f      = preprocess(images.astype(np.float32) / 256)
    bboxes_onehot = bboxes_to_onehot(bboxes, X, Y)
    states = np.concatenate(
        (images_f.reshape((N, images_f.shape[1], -1)),
         bboxes_onehot.reshape((N, images_f.shape[1], -1))),
        axis=-1).astype(np.float32)
    return states, "blocks", n_train


def _load_blocks(track, num):
    from model.paths import find_dataset
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
    from model.paths import find_dataset
    import importlib
    p = importlib.import_module(f"model.loaders.{ptype}")
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
    p.add_argument("--domain", default=None, choices=[None, "vidvrd", "blocks", "puzzle"])
    p.add_argument("--type",  default="mnist")
    p.add_argument("--width", type=int, default=3)
    p.add_argument("--height", type=int, default=3)
    p.add_argument("--track", default="blocks-5-3")
    args = p.parse_args()

    if args.domain is None:
        args.domain = _auto_domain(args.model_dir)
    print(f"[replot] domain={args.domain}  model_dir={args.model_dir}")

    n_train = None
    if args.domain == "vidvrd":
        states, plotmode, n_train = _load_video(args.model_dir, args.domain,
                                                args.num)
    elif args.domain == "blocks":
        states, plotmode = _load_blocks(args.track, args.num)
    else:
        states, plotmode = _load_puzzle(args.type, args.width, args.height, args.num)

    print(f"[replot] sample shape={states.shape}  plotmode={plotmode!r}")

    print(f"[replot] loading model from {args.model_dir}")
    ae = model.model.load(args.model_dir)
    ae.load()
    print(f"[replot] model loaded — U={ae.parameters['U']} P={ae.parameters['P']} A={ae.parameters['A']}")

    N = states.shape[0]
    split = n_train if n_train else max(1, int(N * 0.9))
    train, test = states[:split], states[split:]

    show_summary(ae, train, test)
    plot_autoencoding_image(ae, test, train, plotmode)

    print(f"[replot] DONE. Output PNGs in {args.model_dir}/")


if __name__ == "__main__":
    main()
