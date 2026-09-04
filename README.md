# labeled-fosae — User Guide

A Master's thesis fork of [guicho271828/latplan-fosae](https://github.com/guicho271828/latplan-fosae). The fork extends FOSAE (First-Order State AutoEncoder) to real-world video: `vidvrd`, `actiongenome`, and `videonet`.

The document is a runbook. It tells you how to install the code, how to bake data, how to train the model, and how to look at the results. The document does not explain the theory. For the theory, read `notes/docs/THEORY.md`.

## 1. What FOSAE Does (Short Version)

- FOSAE reads a set of object patches per frame.
- FOSAE learns a binary latent code of shape `U × P`. The code names first-order predicates such as `pred_3(dog, table)`.
- FOSAE trains without labels. The only loss is reconstruction (BCE on patches, MSE on bboxes).
- The output is (a) a reconstruction of the input frame and (b) a discrete symbolic state that a classical planner can use.

The FOSAE paper is at [arXiv:1902.08093](https://arxiv.org/abs/1902.08093). Read `notes/docs/THEORY.md §1-2` for the architecture.

## 2. What This Fork Adds

- **New video domains.** `vidvrd`, `actiongenome`, and `videonet` loaders under `latplan/puzzles/puzzle_vidvrd.py`, `latplan/domains/video/actiongenome.py`, and `sh/download_videonet.sh`.
- **Per-video overfit pipeline.** `setup-dataset.py video_vidvrd <cat> --video-id <VID>` bakes a single-video training set. The training pipeline reads it through the `NPZ_PATH` env var.
- **Env-var knobs for hyperparameters.** `LR`, `PREENC_LAYERS`, `PREENC_DIM`, `MAX_TEMPERATURE`, `ZEROSUPPRESS`, `DROPOUT`, `NOISE`, `NO_EARLYSTOP`, `EPOCH`, `TRANSITION_MODE`, `BATCH`, `FPS`.
- **Sherlock HPC workflow.** `sh/submit.sh` composes a hierarchical output directory with a sha1 hash. `run_training.sh` runs post-train hooks: `tools/replot.py`, `viz/recon.py`, and `tools/plot_training_curve.py`.
- **Deterministic output paths.** `latplan/util/paths.py::resolved_out_dir` composes `out/<domain>/<category>/<run_tag>/`. Two runs with the same config land in the same directory. Two concurrent runs with the same config get `_2`, `_3` suffixes.
- **Baseline verification.** `sh/baseline_verify.sh` runs a pinned MNIST puzzle config and checks against `val_BCE = 1.038e-07`.

## 3. Install

### 3.1 Local (workstation)

```bash
bash install.sh                                    # first-time setup: conda env `fosae` per environment.yml
conda activate fosae
python setup-dataset.py puzzle mnist 3 3 5000      # bake the puzzle dataset
python strips.py learn puzzle mnist 3 3 5000       # train
bash smoke_test.sh                                 # confirm the install
```

`install.sh` creates the conda env `fosae` from `environment.yml`. `environment.yml` pins `python=3.7`, `tensorflow-gpu==1.15.2`, `h5py==2.10.0`, `numpy>=1.16.0,<1.24.0`, `keras-adabound==0.9.0`, `keras-rectified-adam==0.9.0`, `protobuf==3.20.3`.

### 3.2 Sherlock HPC (canonical for training)

```bash
# On a login node, once per shell session
source activate.sh                                 # loads module collection `fosae` + venv
```

The module collection `fosae` holds `gcc/8.1.0`, `python/3.6.1`, `cudnn/7.6.5`, `cuda/10.0.130`, and `ffmpeg`. `activate.sh` sources `sh/sherlock_env.sh`. That file loads the collection and activates the project venv.

**Warning**: The `cudnn/7.6.5` module has a dependency on `cuda/10.2.89`. Load `cuda/10.0.130` after `cudnn/7.6.5` so that the second `cuda` load swaps to `cuda/10.0.130`. TensorFlow 1.15.2 requires `libcudart.so.10.0` and `libcudnn.so.7`. `sh/sherlock_env.sh` handles the order.

Verify the GPU stack:

```bash
python3 -c "from tensorflow.python.client import device_lib; print([d.name for d in device_lib.list_local_devices()])"
# expect: output contains `/device:GPU:0` AND no `libcudart.so.10.0` dlopen warning
```

## 4. Bake a Dataset

The bake step is one Python command per data setup. The bake writes an `.npz` file into `data/npz/`.

### 4.1 Puzzle datasets

```bash
python setup-dataset.py puzzle mnist 3 3 5000
python setup-dataset.py puzzle mandrill 3 3 5000
python setup-dataset.py blocksworld blocks-5-3 6500
```

### 4.2 Video overfit (single-video, canonical since 2026-05-19)

```bash
# 1. Look at a candidate video
python3 - <<'PY'
import json
a = json.load(open('data/video/vidvrd/annotations/train/<VID>.json'))
print('subjects:', [(o['tid'], o['category']) for o in a.get('subject/objects', [])])
print('non-empty frames:', sum(1 for fr in a.get('trajectories', []) if fr))
PY

# 2. Bake the overfit npz to data/npz/video/vidvrd/overfit/
python3 setup-dataset.py video_vidvrd <CAT> \
    --video-id <VID> --fps 30 --max-objects 3 \
    --fill-annotations \
    --out-name <CAT>-<VID>-30fps-mo3-fill
```

Flags:

- `--video-id ID1[,ID2,...]` — one or more videos. A comma-separated list makes a multi-video npz.
- `--fps N` — frame extraction rate.
- `--max-objects K` — object slot count per frame. Cuts padding-slot noise. Use `K = real_object_count + 1`.
- `--fill-annotations` — forward-fill and backward-fill the trajectory when the annotation density is lower than the frame rate. VidVRD sample: dog-frisbee video goes from 60 to 135 states.
- `--patch-size N` — patch resolution (default 32).
- `--out-name STR` — output file stem.

### 4.3 Video per-category (all-video-in-category, deprecated for main experiment)

```bash
python setup-dataset.py video_vidvrd person --fps 30 --out-name person-30fps
```

This bakes every video in the category into one npz. Per-category data volume on VidVRD is under the viability threshold (largest = `person` ≈ 1967 transitions). Prefer the per-video overfit workflow (`§4.2`) for the main experiment.

### 4.4 ActionGenome

```bash
python setup-dataset.py video_ag chair --max-videos 30 --out-name chair-30vids
```

ActionGenome has 17 viable classes (≥ 5000 transitions). See `tools/video/inspect_ag.py --threshold 5000`.

## 5. Train

The canonical entry point is `sh/submit.sh`. It composes the output directory and submits an sbatch job on Sherlock. For a local dry run, prepend `DRY_RUN=1`.

### 5.1 Video overfit (recommended)

```bash
NPZ_PATH=$SCRATCH/panos/sgg-thesis/data/npz/video/vidvrd/overfit/<CAT>-<VID>-30fps-mo3-fill.npz \
DOMAIN=vidvrd TRANSITION_MODE=sequential EPOCH=2000 NO_EARLYSTOP=1 \
LR=0.001 PREENC_LAYERS=0 MAX_TEMPERATURE=1.0 \
MEM=16G TIME=0:45:00 AUTO_RESOURCES=0 \
bash sh/submit.sh
```

Critical override knobs for video overfit:

| Knob | Overfit value | Default (video) | Reason |
|------|---------------|-----------------|--------|
| `LR` | `0.001` | `0.0001` | The video default is too slow for a small overfit set. |
| `PREENC_LAYERS` | `0` | `2` | The pre-encoder adds capacity that a small overfit set does not need. |
| `NO_EARLYSTOP` | `1` | not set | `EarlyStopMixin` kills training before the Gumbel anneal completes. |
| `MAX_TEMPERATURE` | `1.0` | `5.0` | The paper default is too hot for the Gumbel discretization. MNIST baseline uses `1.0`. |
| `--fill-annotations` (bake) | on | off | VidVRD annotation density (~13 fps) is lower than the frame rate (30 fps). |
| `--max-objects K` (bake) | `real + 1` | 10 | 8 pad slots swamp the reconstruction loss on a 2-object scene. |

### 5.2 MNIST puzzle baseline

```bash
DOMAIN=puzzle PUZZLE_TYPE=mnist WIDTH=3 HEIGHT=3 NUM_EXAMPLES=20000 \
EPOCH=1000 NO_EARLYSTOP=1 \
bash sh/submit.sh
```

Verified result: `val_BCE = 1.038e-07` at epoch 1000. `sh/baseline_verify.sh` runs this as a regression sentinel.

### 5.3 Env-var reference

| Group | Vars |
|-------|------|
| Training | `DOMAIN`, `AECLASS`, `U`, `A`, `P`, `EPOCH`, `MAX_VIDEOS`, `BATCH`, `FPS`, `CATEGORY`, `TRANSITION_MODE`, `ARGS_TAIL`, `TRAIN_CMD`, `NPZ_PATH`, `NO_EARLYSTOP` |
| Hyperparameters | `LR`, `PREENC_LAYERS`, `PREENC_DIM`, `MAX_TEMPERATURE`, `ZEROSUPPRESS`, `ZEROSUPPRESS_DELAY`, `DROPOUT`, `NOISE` |
| Post-train | `EXTRACT_FOL` (0/1), `VISUALIZE` (0/1), `VIS_NUM`, `FOL_VIZ` (0/1) |
| SLURM | `JOB_NAME`, `JOB_SUFFIX`, `PARTITION`, `GPUS`, `CPUS`, `MEM`, `TIME`, `QOS`, `CONSTRAINT`, `MAIL_TYPE`, `MAIL_USER` |
| Estimator | `AUTO_RESOURCES` (0/1), `SACCT_DAYS` |
| Filter | `VIDVRD_STRICT_CATEGORY` (0/1) |
| Modules | `PYTHON_MODULE`, `CUDA_MODULE`, `CUDNN_MODULE`, `GCC_MODULE` |

## 6. Inspect the Outputs

Each run lands in a unique directory. The pattern is:

```
out/<domain>/<category>/<run_tag>/
```

`<run_tag>` = `<aeclass>_U<U>_A<A>_P<P>[_cat<X>][_fps<F>]_<sha1[:6]>[_<idx>]`. The sha1 covers every hyperparameter. A configuration collision gets `_2`, `_3`.

### 6.1 Files that a training run writes

| File | Purpose |
|------|---------|
| `net0.h5` | Trained model weights. |
| `training_history.csv` | Per-epoch `BCE`, `MSE`, `activation`, `loss`, `preencoder_l1` (and `val_*` siblings). CSVLogger writes it every epoch. |
| `training_curve.png` | 4-panel loss plot. `tools/plot_training_curve.py` writes it. |
| `render_test.png`, `render_train.png` | Decoded scene canvases per split. |
| `render_test_shuffled.png`, `render_train_shuffled.png` | Slot-shuffled reconstruction. Tests permutation invariance. |
| `autoencoding_test.png`, `autoencoding_train.png` | Latent grid plus decoded scene per state. |
| `booleans_test.png`, `booleans_train.png` | U × P bit pattern per state. Shows collapsed predicates. |
| `viz/recon_grid.png` | 3 × N grid: input row, reconstruction row, per-pixel diff row. |
| `loaded_videos.json` | Data provenance: `video_ids`, `video_id_filter`, `npz_path`, `fps`, `category_filter`, `transition_mode`. |
| `trial_t1/net0.h5` | Per-trial save from the genetic search wrapper. |
| `test*.pdf` | Per-predicate CART decision trees. |

### 6.2 Post-run inspection commands

```bash
OUT=<OUT_DIR from submit.sh output>
tail -5 $OUT/training_history.csv                  # last 5 epochs
ls $OUT/*.png $OUT/viz/*.png                        # every rendered PNG
jq '.category_filter, .video_id_filter, .npz_path' $OUT/loaded_videos.json
```

### 6.3 Re-run visualization on a saved model

```bash
python3 tools/replot.py $OUT --num 200              # original-author plot suite
python3 viz/recon.py $OUT --num 8                   # single-glance reconstruction grid
python3 tools/plot_training_curve.py $OUT           # loss curve
```

### 6.4 Find your finished runs

```bash
python3 tools/list_runs.py                          # sacct sweep, sorted by val_BCE_min
```

`tools/list_runs.py` reads `sacct` for the user, picks completed jobs, extracts `val_BCE_min` from `training_history.csv`, and lists them best-first.

## 7. Success Criteria

For every trained model, check three things.

1. **Reconstruction.** Open `viz/recon_grid.png`. The reconstruction row should look like the input row. The diff row should be mostly dark. Target: mean MSE < 0.05 for realistic-image domains.
2. **Training curve.** Open `training_curve.png`. All four panels should trend downwards and plateau. A flat line means no learning.
3. **Boolean patterns.** Open `booleans_test.png`. Predicates should vary across states. A predicate that is always on or always off is collapsed.

For a planner-eval run (Phase H), see `notes/docs/STATUS.md §Phase H` and `tools/planner/plan_video.py`.

## 8. Where to Find More

- `notes/docs/THEORY.md` — FOSAE theory and architecture.
- `notes/docs/AUDIT.md` — code alignment with the paper and the upstream repository. Read this before any change to `strips.py` or the loaders.
- `notes/docs/SPEC.md` — task grid, invariants, and gate list.
- `notes/docs/STATUS.md` — weekly progress and the phase timeline.
- `notes/docs/VIZ.md` — figure catalogue.
- `notes/docs/CHANGES.md` — every change from the upstream fork.
- `notes/docs/WORKING_RULES.md` — working rules and conventions.
- `notes/docs/STE.md` — the Simplified Technical English style guide.
- `notes/docs/SOURCES.md` — paper links and dataset links.

## 9. Citation

If you cite this work, cite the source paper and the LatPlan predecessor.

- Asai, M. (2019). *Unsupervised Grounding of Plannable First-Order Logic Representation from Images*. ICAPS 2019. [arXiv:1902.08093](https://arxiv.org/abs/1902.08093).
- Asai, M. and Fukunaga, A. (2018). *Classical Planning in Deep Latent Space: Bridging the Subsymbolic-Symbolic Boundary*. IJCAI 2018. [arXiv:1705.05787](https://arxiv.org/abs/1705.05787).

## 10. Change log for this file

- 2025-08-02: STE verification pass against the real `STE.md` rules. All prose sentences pass the ≤ 25-word descriptive limit (Rule 6.3). Every command block preserved verbatim (Rule 11 STE.md — code stays unchanged). Fixed L46 `which loads` → `That file loads` per GR-1 (prefer `that` conjunction; split the relative clause into a new sentence for clarity). Removed §9 duplicate `Related Files` (Section 8 `Where to Find More` already lists every file). Result: 250 → 240 lines.
