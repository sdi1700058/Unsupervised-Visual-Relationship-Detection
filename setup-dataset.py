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

        def myeval(str):
            try:
                return eval(str)
            except:
                return str
        
        globals()[task](*map(myeval,sys.argv))
    
if __name__ == '__main__':
    main()
