#!/usr/bin/env python3

import config
import numpy as np
import random
import latplan
import latplan.model
from latplan.util        import curry
from latplan.util.tuning import grid_search, nn_task
from latplan.util.noise  import gaussian

import keras.backend as K
import tensorflow as tf

import os
import os.path

float_formatter = lambda x: "%.5f" % x
import sys
np.set_printoptions(threshold=sys.maxsize,formatter={'float_kind':float_formatter})

# ── Canonical data directory (npz cache root per SPEC §I6) ─────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "npz")
os.makedirs(DATA_DIR, exist_ok=True)

# Video overfit npz keyspace — separate from default per-category cache
# (data/npz/video/<ds>/<cat>-<fps>fps.npz) so single-video / sub-sample
# experiments don't clobber the canonical cache.
_VIDEO_OVERFIT_SUBDIR = "overfit"

def _save_and_symlink(data_path, filename):
    """Create a backward-compat relative symlink in latplan/puzzles/ pointing to data/."""
    legacy_path = os.path.join(latplan.__path__[0], "puzzles", filename)
    if os.path.abspath(data_path) != os.path.abspath(legacy_path):
        os.makedirs(os.path.dirname(legacy_path), exist_ok=True)
        if os.path.exists(legacy_path) or os.path.islink(legacy_path):
            os.remove(legacy_path)
        # Use relative symlink so the project stays portable
        rel_target = os.path.relpath(os.path.abspath(data_path),
                                     os.path.dirname(os.path.abspath(legacy_path)))
        os.symlink(rel_target, legacy_path)
        print(f"  Symlinked {legacy_path} -> {rel_target}")

################################################################

def puzzle(type='mnist',width=3,height=3,limit=None):
    # limit = number that "this much is enough"
    filename = "-".join(map(str,["puzzle",type,width,height]))+".npz"
    path = os.path.join(DATA_DIR, filename)
    import importlib
    p = importlib.import_module('latplan.puzzles.puzzle_{}'.format(type))
    p.setup()
    pres = p.generate_random_configs(width*height, limit)
    np.random.shuffle(pres)
    sucs = [ random.choice(p.successors(c1,width,height)) for c1 in pres ]
    np.savez_compressed(path,pres=pres,sucs=sucs)
    print(f"Dataset saved to {path}")
    _save_and_symlink(path, filename)

def hanoi(disks=7,towers=4,limit=None):
    filename = "-".join(map(str,["hanoi",disks,towers]))+".npz"
    path = os.path.join(DATA_DIR, filename)
    import latplan.puzzles.hanoi as p
    p.setup()
    pres = p.generate_random_configs(disks,towers, limit)
    np.random.shuffle(pres)
    sucs = [ random.choice(p.successors(c1,disks,towers)) for c1 in pres ]
    np.savez_compressed(path,pres=pres,sucs=sucs)
    print(f"Dataset saved to {path}")
    _save_and_symlink(path, filename)

def lightsout(type='digital',size=4,limit=None):
    filename = "-".join(map(str,["lightsout",type,size]))+".npz"
    path = os.path.join(DATA_DIR, filename)
    import importlib
    p = importlib.import_module('latplan.puzzles.lightsout_{}'.format(type))
    p.setup()
    pres = p.generate_random_configs(size, limit)
    np.random.shuffle(pres)
    sucs = [ random.choice(p.successors(c1)) for c1 in pres ]
    np.savez_compressed(path,pres=pres,sucs=sucs)
    print(f"Dataset saved to {path}")
    _save_and_symlink(path, filename)

def blocksworld(track="blocks-5-3"):
    """Download blocksworld dataset from the latplan-fosae GitHub releases.

    The original FOSAE project used pre-generated .npz files containing:
      - images: (N, num_objs, H, W, 3) object patch images
      - bboxes: (N, num_objs, 4) bounding boxes [x1, y1, x2, y2]
      - transitions: (2*T,) transition indices into the states array
      - picsize: (2,) the full scene image size [H, W]

    If the file already exists in data/, this is a no-op.
    """
    filename = track + ".npz"
    path = os.path.join(DATA_DIR, filename)

    if os.path.exists(path):
        print(f"Blocksworld dataset already exists: {path}")
        _print_blocksworld_info(path)
        return

    # Check if there's a copy somewhere in the project already
    project_root = os.path.dirname(os.path.abspath(__file__))
    legacy_paths = [
        os.path.join(project_root, track, filename),
        os.path.join(project_root, track, f"blocksworld_FirstOrderSAE_{track}_None_None_None_10000_BCE5", filename),
        os.path.join(latplan.__path__[0], "puzzles", filename),
    ]
    for lp in legacy_paths:
        if os.path.exists(lp) and not os.path.islink(lp):
            import shutil
            print(f"Found existing blocksworld data at {lp}, copying to {path}")
            shutil.copy2(lp, path)
            _save_and_symlink(path, filename)
            _print_blocksworld_info(path)
            return

    # Try multiple possible download URLs
    urls = [
        f"https://github.com/IBM/photorealistic-blocksworld/releases/download/{track}/{track}.npz",
        f"https://github.com/guicho271828/latplan-fosae/raw/fosae/latplan/puzzles/{filename}",
    ]
    for url in urls:
        print(f"Trying: {url}")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, path)
            # Verify it's a valid npz
            with np.load(path) as data:
                _ = list(data.keys())
            print("  Success!")
            _save_and_symlink(path, filename)
            _print_blocksworld_info(path)
            return
        except Exception as e:
            print(f"  Failed: {e}")
            if os.path.exists(path):
                os.remove(path)

    print()
    print(f"ERROR: Could not find or download {filename}")
    print()
    print("To set up blocksworld data manually:")
    print("  1. Clone https://github.com/IBM/photorealistic-blocksworld")
    print("  2. Generate states and render images")
    print("  3. Package into .npz with keys: images, bboxes, transitions, picsize")
    print(f"  4. Place {filename} in {DATA_DIR}/")
    print()
    print("Or copy from an existing latplan-fosae checkout that has the data.")
    sys.exit(1)

def _print_blocksworld_info(path):
    """Print summary info about a blocksworld .npz file."""
    try:
        with np.load(path) as data:
            keys = list(data.keys())
            print(f"  Keys: {keys}")
            for k in keys:
                arr = data[k]
                print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
    except Exception as e:
        print(f"  (could not read: {e})")


################################################################
# ── Video overfit npz baking ───────────────────────────────────────────
#
# Writes a self-describing raw-arrays npz at
#   data/npz/video/<dataset>/overfit/<out_name>.npz
# Same key schema as the default per-category cache (latplan/util/cache.py)
# so strips.py can consume both via the new `npz_path=` kwarg (Phase V3).
################################################################

def _safe_name(s):
    """Make a string safe for use as a file segment (no '/', no spaces)."""
    return s.replace("/", "_").replace(" ", "_")


def _video_out_path(dataset, out_name):
    sub = os.path.join(DATA_DIR, "video", dataset, _VIDEO_OVERFIT_SUBDIR)
    os.makedirs(sub, exist_ok=True)
    if not out_name.endswith(".npz"):
        out_name = out_name + ".npz"
    return os.path.join(sub, out_name)


def _default_video_out_name(category, video_id, fps, max_videos):
    if category is None:
        head = "allcat"
    elif isinstance(category, (list, tuple, set)):
        head = "+".join(sorted(category))
    else:
        head = str(category)
    parts = [_safe_name(head)]
    if isinstance(video_id, (list, tuple, set)):
        ids = sorted(video_id)
        if len(ids) == 1:
            parts.append(_safe_name(str(ids[0])))
        else:
            import hashlib
            h = hashlib.sha1("|".join(ids).encode("utf-8")).hexdigest()[:6]
            parts.append(f"{len(ids)}vids-{h}")
    elif video_id is not None:
        parts.append(_safe_name(str(video_id)))
    elif max_videos is not None:
        parts.append(f"top{max_videos}")
    parts.append(f"{fps}fps")
    return "-".join(parts)


def _bake_video_npz(loader_module, dataset_name, category, video_id, fps,
                    max_videos, out_name, max_objects, fill_annotations=False,
                    patch_size=None, annotations_dir=None, frames_dir=None):
    """Common path for video_ag / video_vidvrd bakers.

    Calls the loader's build_dataset(...) with video_id_filter set, then
    writes the raw-array npz at the overfit/ keyspace.

    video_id : str | list[str] | None — when a list, the loader receives all
    ids and the default out_name is `<cat>-<N>vids-<sha1[:6]>-<fps>fps`.
    patch_size : int | None — overrides the module default (PATCH_SIZE=32).
    Increases pixel resolution per object slot; model.py auto-detects the new
    patch dim from the data tensor (no model.py change).
    annotations_dir / frames_dir : Path overrides forwarded to the loader.
    Used when loading from a non-default directory layout (e.g. VideoNet
    layered under `data/video/videonet/`) without a new loader module.
    """
    from latplan.util.cache import save_cache

    if out_name is None:
        out_name = _default_video_out_name(category, video_id, fps, max_videos)
    out_path = _video_out_path(dataset_name, out_name)

    print(f"[bake] dataset={dataset_name}  category={category}  "
          f"video_id={video_id}  fps={fps}  max_videos={max_videos}  "
          f"max_objects={max_objects}  patch_size={patch_size}")
    print(f"[bake] writing  {out_path}")

    kwargs = dict(category_filter=category, fps=fps, num_objs=max_objects)
    if video_id is not None:
        kwargs["video_id_filter"] = video_id
    if max_videos is not None:
        kwargs["max_videos"] = max_videos
    if fill_annotations and dataset_name == "vidvrd":
        kwargs["fill_annotations"] = True
    if patch_size is not None:
        kwargs["patch_size"] = patch_size
    if annotations_dir is not None:
        kwargs["annotations_dir"] = annotations_dir
    if frames_dir is not None:
        kwargs["frames_dir"] = frames_dir

    images, bboxes, names, frame_ids = loader_module.build_dataset(**kwargs)
    meta = dict(loader_module.last_load_metadata)
    meta.update({
        "dataset":        dataset_name,
        "category":       category,
        "video_id":       video_id,
        "fps":            fps,
        "max_videos":     max_videos,
        "max_objects":    max_objects,
        "patch_size":     patch_size if patch_size is not None
                          else meta.get("patch_size"),
        "fill_annotations": bool(fill_annotations),
        "out_name":       out_name,
        "schema":         "raw_v1",   # versioning hook for future migrations
    })
    save_cache(out_path, images, bboxes, names, frame_ids, meta)
    print(f"[bake] OK — {len(images)} states, {len(meta.get('video_ids', []))} videos")
    print(f"[bake] use with: NPZ_PATH={out_path} DOMAIN={dataset_name} bash sh/submit.sh")
    return out_path


def video_ag(category=None, video_id=None, fps="native",
             max_videos=None, out_name=None, max_objects=10,
             fill_annotations=False, patch_size=None,
             annotations_dir=None, frames_dir=None):
    """Bake an ActionGenome overfit npz for one video / category subset."""
    if category is None:
        raise SystemExit("video_ag: --category is required (e.g. chair, table, food)")
    from latplan.domains.video import actiongenome as _ag
    if fill_annotations:
        print("[bake] note: --fill-annotations has no effect on ActionGenome (AG frame_list is dense already)")
    return _bake_video_npz(_ag, "actiongenome", category, video_id, fps,
                           max_videos, out_name, max_objects,
                           fill_annotations=False, patch_size=patch_size,
                           annotations_dir=annotations_dir, frames_dir=frames_dir)


def _parse_category(category):
    """Normalise the CLI category into what build_dataset expects.

    'all' (or '*') means no filter at all — every video in the split.
    A comma-separated list becomes a list of names. Anything else is
    returned unchanged.
    """
    if category is None:
        return None
    if isinstance(category, str):
        if category.strip().lower() in ("all", "*"):
            return None
        if "," in category:
            names = [c.strip() for c in category.split(",") if c.strip()]
            return names if len(names) > 1 else names[0]
    return category


def video_vidvrd(category=None, video_id=None, fps=3,
                 max_videos=None, out_name=None, max_objects=10,
                 fill_annotations=False, patch_size=None,
                 annotations_dir=None, frames_dir=None):
    """Bake a VidVRD overfit npz for one video / category subset.

    fill_annotations=True : carry forward last non-empty trajectory entry into
        subsequent empty entries (annotations are sparse in VidVRD; ~13 fps for
        most videos). Image stays from the real source-PTS jpeg; only the bbox
        list is reused. Valid when tracked objects are continuously visible.

    patch_size : int | None — overrides per-object patch resolution (default
        32 = puzzle_labeled_objects.PATCH_SIZE).

    video_id : str | list[str] | None — comma-separated CLI list lands here as
        a Python list and is forwarded as video_id_filter.

    category : one name, a comma-separated group ('dog,cat,bird'), or 'all'
        for every video regardless of category. A group or 'all' skips the
        per-category cache, since neither is a single cache key.
    """
    if category is None:
        raise SystemExit("video_vidvrd: --category is required "
                         "(e.g. bicycle, dog, person, 'dog,cat,bird', or 'all')")
    category = _parse_category(category)
    from latplan.puzzles import puzzle_vidvrd as _vv
    return _bake_video_npz(_vv, "vidvrd", category, video_id, fps,
                           max_videos, out_name, max_objects,
                           fill_annotations=fill_annotations,
                           patch_size=patch_size,
                           annotations_dir=annotations_dir,
                           frames_dir=frames_dir)


def _parse_video_args(argv):
    """Lightweight argparse for the two video subcommands. Returns kwargs dict."""
    import argparse
    p = argparse.ArgumentParser(prog="setup-dataset.py video_ag|video_vidvrd")
    p.add_argument("category",
                   help="Primary category (e.g. chair, bicycle). Accepts a "
                        "comma-separated group ('dog,cat,bird') or 'all' for "
                        "every video regardless of category.")
    p.add_argument("--video-id", default=None,
                   help="Restrict to one or more video-ids. Accepts a single id or a comma-separated list "
                        "(AG: '<vid>.mp4'; VidVRD: 'ILSVRC2015_…'). Multi-id triggers a <N>vids-<hash> default out-name.")
    p.add_argument("--fps",        default=None, help="fps key (AG default: native, VidVRD default: 3)")
    p.add_argument("--max-videos", default=None, type=int)
    p.add_argument("--max-objects", default=10,  type=int)
    p.add_argument("--patch-size", default=None, type=int,
                   help="Per-object patch resolution (default = puzzle_labeled_objects.PATCH_SIZE = 32). "
                        "model.py auto-detects the new patch dim from the data tensor.")
    p.add_argument("--out-name",   default=None,
                   help="Output stem (no .npz). Default: <cat>-<video_id?>-<fps>fps")
    p.add_argument("--fill-annotations", action="store_true",
                   help="VidVRD only: carry last non-empty trajectory forward into empty entries (dense per-frame supervision)")
    p.add_argument("--annotations-dir", default=None,
                   help="Override default annotations dir (passed to loader.build_dataset). "
                        "Use when sourcing annotations from a non-default layout (e.g. VideoNet under data/video/videonet/).")
    p.add_argument("--frames-dir", default=None,
                   help="Override default frames dir (passed to loader.build_dataset). "
                        "Use for non-default frame trees (e.g. data/video/videonet/frames_5fps).")
    ns = p.parse_args(argv)
    # Parse comma-separated --video-id into a list (single value stays a string).
    if ns.video_id is not None and "," in ns.video_id:
        ns.video_id = [v.strip() for v in ns.video_id.split(",") if v.strip()]
    return ns


################################################################

def list_datasets():
    """List all available datasets in data/."""
    print(f"Datasets in {DATA_DIR}/:")
    if not os.path.exists(DATA_DIR):
        print("  (directory does not exist)")
        return
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".npz"))
    if not files:
        print("  (none)")
    for f in files:
        fpath = os.path.join(DATA_DIR, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {f}  ({size_mb:.1f} MB)")


def main():
    import sys
    if len(sys.argv) == 1:
        print("Usage: python setup-dataset.py <task> [args...]")
        print()
        print("Tasks:")
        print("  puzzle <type> <width> <height> [limit]")
        print("    type: mnist, digital, mandrill, spider, lenna")
        print("    Example: python setup-dataset.py puzzle mnist 3 3 5000")
        print()
        print("  hanoi <disks> <towers> [limit]")
        print("    Example: python setup-dataset.py hanoi 7 4 5000")
        print()
        print("  lightsout <type> <size> [limit]")
        print("    type: digital, twisted")
        print("    Example: python setup-dataset.py lightsout digital 4 5000")
        print()
        print("  blocksworld [track]")
        print("    track: blocks-5-3 (default), blocks-4-4")
        print("    Downloads from GitHub releases if not present locally.")
        print("    Example: python setup-dataset.py blocksworld blocks-5-3")
        print()
        print("  video_ag <category> [--video-id ID] [--fps native] [--max-videos N] [--max-objects N] [--out-name STR]")
        print("    Bake an ActionGenome overfit npz at data/npz/video/actiongenome/overfit/.")
        print("    Example: python setup-dataset.py video_ag chair --video-id 001YG.mp4")
        print()
        print("  video_vidvrd <category> [--video-id ID] [--fps 3] [--max-videos N] [--max-objects N] [--out-name STR]")
        print("    Bake a VidVRD overfit npz at data/npz/video/vidvrd/overfit/.")
        print("    Example: python setup-dataset.py video_vidvrd bicycle --video-id ILSVRC2015_train_00010001")
        print()
        print("  list")
        print("    List all datasets in data/")
        return
    else:
        print('args:',sys.argv)
        sys.argv.pop(0)
        task = sys.argv.pop(0)

        if task == "list":
            list_datasets()
            return

        # Video subcommands use argparse (named flags); puzzle/blocks/hanoi/
        # lightsout keep the legacy positional `eval` dispatch.
        if task in ("video_ag", "video_vidvrd"):
            ns = _parse_video_args(sys.argv)
            kw = dict(category=ns.category, video_id=ns.video_id,
                      max_videos=ns.max_videos, max_objects=ns.max_objects,
                      out_name=ns.out_name,
                      fill_annotations=ns.fill_annotations,
                      patch_size=ns.patch_size,
                      annotations_dir=ns.annotations_dir,
                      frames_dir=ns.frames_dir)
            if ns.fps is not None:
                # myeval-style coercion so '3' becomes int but 'native' stays str
                try:    kw["fps"] = int(ns.fps)
                except: kw["fps"] = ns.fps
            globals()[task](**kw)
            return

        def myeval(str):
            try:
                return eval(str)
            except:
                return str

        globals()[task](*map(myeval,sys.argv))
    
if __name__ == '__main__':
    main()
