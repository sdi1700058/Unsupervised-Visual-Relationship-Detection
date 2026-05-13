# labeled-fosae

**First-Order State AutoEncoder (FOSAE)** — fork of [guicho271828/latplan-fosae](https://github.com/guicho271828/latplan-fosae) extended with a VLM-annotation pipeline for real-world COCO images.

FOSAE learns to decompose visual scenes into object-centric, interpretable **First-Order Logic (FOL)** predicates entirely without supervision. It bridges raw pixel input with symbolic planning (PDDL/STRIPS). Published at ICAPS 2019; see [arXiv:1902.08093](https://arxiv.org/abs/1902.08093).

This fork extends FOSAE with two new real-world domains:
- **Labeled Objects** — VLM-annotated (Unsloth/Qwen3-VL) COCO images
- **VidVRD** — ImageNet Video Visual Relationship Detection (1000 videos, per-frame object tracking)

---

## Quick Start

```bash
bash install.sh                                    # first-time setup (conda env + package install)
conda activate latplan
python setup-dataset.py puzzle mnist 3 3 5000      # generate puzzle dataset
python strips.py learn puzzle mnist 3 3 5000       # train
bash smoke_test.sh                                 # verify install
```

---

## Table of Contents

1. [Theoretical Background](#1-theoretical-background)
2. [Architecture Deep-Dive](#2-architecture-deep-dive)
3. [Repository Layout](#3-repository-layout)
4. [Domains](#4-domains) — puzzle · blocksworld · labeled\_objects · vidvrd
5. [Data Pipeline](#5-data-pipeline)
6. [Installation](#6-installation)
7. [Training](#7-training)
8. [FOL Extraction](#8-fol-extraction)
9. [Visualization](#9-visualization)
10. [Adding a New Dataset](#10-adding-a-new-dataset)
11. [Output Reference](#11-output-reference)
12. [Citation](#12-citation)

---

## 1. Theoretical Background

### 1.1 The Problem: Images → Symbolic Planning

Classical AI planners (FastForward, LAMA, etc.) operate on **symbolic state descriptions** — sets of logical facts such as `on(blockA, blockB)`, `clear(table)`. Writing these descriptions by hand (PDDL domain files) is expensive and domain-specific.

FOSAE's goal: given only raw images of a planning domain and pairs of (pre-state, successor-state), learn the symbolic state representation automatically.

### 1.2 LatPlan (Prior Work)

The predecessor system, **LatPlan** ([Asai & Fukunaga, IJCAI 2018](https://arxiv.org/abs/1705.05787)), encodes images into **propositional** binary latent vectors — each bit is a boolean proposition. LatPlan works but has limitations:

- Representations are **not relational**: there is no concept of "objects" or "arguments".
- The fixed-size latent space does not generalize to problem instances with a different number of objects.
- Learned bits have no interpretable semantics.

### 1.3 FOSAE: First-Order Logic from Images

FOSAE ([Asai, ICAPS 2019](https://arxiv.org/abs/1902.08093)) extends LatPlan to **first-order logic** by introducing two key ideas:

**1. Object-centric input representation.**
Rather than feeding a single flat image vector to the encoder, the input is decomposed into `N` object feature vectors: shape `(N, feature_dim)`. Each slot represents one detected object with its appearance and location.

**2. Attention-based argument binding.**
The encoder maintains `U` **predicate units**. Each unit selects `A` objects from the scene via a learned **attention mechanism** — these become the *arguments* of the predicate. The attention output has shape `(U, A, N)`, where `attention[u, a, o]` is the weight that unit `u` places on object `o` for argument slot `a`.

After binding, each unit produces `P` binary predicate truth-values. The full latent space has `U × P` bits. A decoded state is the union of all true atoms:

```
pred_p(obj_binding[u,0], obj_binding[u,1], ...) = TRUE   iff   latent[u, p] = 1
```

This yields interpretable atoms such as `pred_3(dog, table)` whose meaning is discovered purely by gradient descent.

**3. Gumbel-Softmax quantization.**
The attention weights and predicate values are made discrete at test time but remain differentiable during training via the **Concrete distribution** (Gumbel-Softmax). Temperature anneals from `max_temperature=5.0` to `min_temperature=0.7` during training.

**4. Zero-suppression loss.**
An optional L1 penalty on the latent bits (`zerosuppress` hyperparameter) encourages most predicates to stay off, producing sparse, interpretable codes.

### 1.4 Why First-Order Logic Matters

- **Generalization**: a predicate `on(X, Y)` learned from 5-block scenes can be applied to 10-block scenes without retraining.
- **Interpretability**: attention maps directly show which objects a predicate is about.
- **Planning compatibility**: the binary latent vector is a valid propositional state; the FOL reading is a bonus semantic layer.

---

## 2. Architecture Deep-Dive

All neural architecture code lives in [latplan/model.py](latplan/model.py).

### 2.1 Class Hierarchy

```
Network
└── AE (autoencoder base)
    └── StateAE  (Gumbel-Softmax latent space, reconstruction loss)
        └── FirstOrderSAE   ← main model  (ICAPS 2019)
              = BaseFirstOrderMixin       (attention encoder, domain renderers)
              + ZeroSuppressMixin         (L1 zero-suppression loss)
              + ConcreteLatentMixin       (Gumbel-Softmax discretization)
              + StateAE
```

`FirstOrderSAE` is also exported as `FirstOrderAE` (alias).

### 2.2 Encoder Forward Pass

Given input `x` of shape `(batch, N_obj, feat_dim)`:

**Step 1 — Pre-encoder** (`_build_preencoder`, optional)

```
x : (batch, N_obj, feat_dim)
  → Dense(layer, relu) → BN → Dropout
  → Dense(N_obj × preencoder_dimention)
  → Reshape(N_obj, preencoder_dimention)
  = o_enc : (batch, N_obj, D)
```

If `preencoder_layers = 0`, this step is skipped and `D = feat_dim`.

**Step 2 — Attention network** (`_build_to_attention`)

```
o_enc : (batch, N_obj, D)
  → flatten
  → Dense(layer, relu) → Dropout
  → Dense(U × A × N_obj)
  → Gumbel-Softmax(N=U×A, M=N_obj)   # soft categorical over N_obj objects
  → Reshape(U, A, N_obj)
  = attention : (batch, U, A, N_obj)
```

Each of the `U × A` slots receives a soft probability distribution over the `N_obj` objects.

**Step 3 — Argument assembly** (einsum)

```python
args_enc = tf.einsum("buao,bof->buaf", attention, o_enc)
# (batch, U, A, D) = weighted sum of object embeddings per unit/argument
```

**Step 4 — Predicate projection** (`_build_to_predicates`)

```
args_enc : (batch, U, A, D)
  → Reshape(U, A×D)
  → Conv1D(P×2, kernel=1)             # independent per-unit, no cross-unit interaction
  → activation (Gumbel-Softmax or sigmoid)
  = latent_logits : (batch, U, P×2)
  → binary rounding at test time
  = latent : (batch, U×P)
```

The `× 2` in `P×2` provides pairs of logits for Gumbel-Softmax binary sampling (on/off).

### 2.3 Decoder

The decoder mirrors the encoder:

```
latent : (batch, U×P)
  → Dense(layer, relu) → Dropout
  → Dense(N_obj × feat_dim)
  → Reshape(N_obj, feat_dim)
  → domain-specific activation (sigmoid / rounded_softmax)
  = reconstruction : (batch, N_obj, feat_dim)
```

Loss: `binary_crossentropy(x, reconstruction)` (default), summed over all feature dimensions.

### 2.4 Domain-Specific Activations

Each domain requires its own output activation because the feature format differs:

| Domain | Activation | Logic |
|--------|-----------|-------|
| `puzzle` | `puzzle_activation` | softmax over tile labels (9 bins) + softmax over position (2×3 bins) |
| `blocks` | `blocks_activation` | sigmoid on RGB patches + rounded_softmax on bbox onehot (x1, y1, x2, y2 bins) |
| `labeled_objects` | `blocks_activation` | same as blocks (identical feature format) |

`labeled_objects_activation` is also defined (plain sigmoid) as an alternative but `blocks_activation` is used by default since the feature layout is identical to blocksworld.

### 2.5 Auxiliary Networks (Visualization)

Calling `ae.build_aux(input_shape)` constructs two additional Keras models on top of the already-trained weights:

- `ae.attention_encoder` → `Model(input, attention)` — returns `(batch, U, A, N_obj)`
- `ae.args_encoder` → `Model(input, decoded_args)` — returns reconstructed argument patches `(batch, U, A, feat_dim)`

These are used by `extract_fol.py` and `visualize_fol.py`.

### 2.6 Key Hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `U` | 40 | Number of predicate units |
| `A` | 2 | Arguments per predicate unit (arity) |
| `P` | 20 | Predicates per unit; total latent bits = U×P |
| `layer` | 200 | Dense layer width |
| `dropout` | 0.3 | Dropout rate |
| `noise` | 0.2 | Gaussian noise σ applied to input during training |
| `lr` | 0.001 | Learning rate |
| `optimizer` | radam | Rectified Adam |
| `epoch` | 1000 | Training epochs (5000 for labeled_objects) |
| `batch_size` | 1000 | Minibatch size |
| `max_temperature` | 5.0 | Gumbel-Softmax start temperature |
| `min_temperature` | 0.7 | Gumbel-Softmax end temperature |
| `zerosuppress` | 0.0 | L1 weight on latent bits |
| `zerosuppress_delay` | 0.1 | Fraction of epochs before zero-suppression activates |
| `preencoder_layers` | 0 | Pre-encoder depth (0 = disabled) |
| `preencoder_dimention` | 50 | Pre-encoder output dim per object |
| `preencoder_l1` | 0.0 | L1 on pre-encoder activations |
| `preencoder_delay` | 0.1 | Fraction of epochs before pre-encoder loss activates |
| `preencoder_output_activation` | `("relu","MSE")` | Pre-encoder output nonlinearity + loss |
| `loss` | `BCE` | Reconstruction loss |
| `eval` | `MSE` | Evaluation metric |

For **labeled_objects** the defaults are overridden to: `preencoder_layers=2`, `preencoder_dimention=256`, `lr=0.0001`, `preencoder_output_activation=("linear","MSE")`, `epoch=5000`.

---

## 3. Repository Layout

```
labeled-fosae/
├── latplan/                            # Core library
│   ├── model.py                        # All neural architectures (2800+ lines)
│   ├── __init__.py
│   ├── puzzles/                        # Domain-specific data generators
│   │   ├── puzzle_mnist.py             # 3×3 MNIST sliding puzzle
│   │   ├── puzzle_digital.py           # 3×3 digital-display puzzle
│   │   ├── puzzle_mandrill.py          # Mandrill image puzzle
│   │   ├── puzzle_spider.py            # Spider image puzzle
│   │   ├── puzzle_lenna.py             # Lenna image puzzle
│   │   ├── puzzle_labeled_objects.py   # NEW: VLM-annotated COCO loader
│   │   ├── hanoi.py                    # Tower of Hanoi
│   │   ├── lightsout_digital.py        # Lights Out (digital)
│   │   ├── lightsout_twisted.py        # Lights Out (twisted)
│   │   ├── model/                      # Puzzle simulation logic
│   │   └── util.py                     # preprocess(), shuffle_objects()
│   └── util/
│       ├── fol.py                      # FOL extraction & formatting
│       ├── layers.py                   # Custom Keras layers (GumbelSoftmax, etc.)
│       ├── tuning.py                   # Hyperparameter search (genetic algorithm)
│       ├── paths.py                    # Canonical PROJECT_ROOT / DATA_DIR / OUT_DIR
│       ├── distances.py                # Distance metrics
│       ├── noise.py                    # Gaussian / salt / pepper noise
│       └── plot.py                     # plot_grid(), squarify()
│
├── strips.py                           # Main training entry point
├── extract_fol.py                      # FOL extraction script
├── visualize_fol.py                    # Visualization script
├── setup-dataset.py                    # Dataset generation / download
├── config.py                           # Keras / TF configuration (GPU)
├── config_cpu.py                       # Keras / TF configuration (CPU)
├── setup.py                            # Python package setup
│
├── notebooks/
│   ├── dataset_gen.ipynb               # VLM annotation pipeline (Unsloth)
│   └── dataset_get.ipynb               # COCO download helpers
│
├── data/                               # Datasets (populated by setup-dataset.py)
│   ├── puzzle-mnist-3-3.npz            # (generated)
│   ├── blocks-5-3.npz                  # (downloaded)
│   └── gen/                            # VLM-annotated outputs
│       ├── fosae_labeled_dataset_unsloth.json
│       └── raw_images/
│
└── out/                                # Training outputs
    ├── puzzle_FirstOrderSAE_mnist_3_3_*/
    ├── blocks-5-3/
    └── labeled_objects/
        └── FirstOrderSAE_U{U}_A{A}_P{P}[_n{N}]/
```

---

## 4. Domains

### 4.1 8-Puzzle (MNIST)

**Domain**: 3×3 sliding tile puzzle where tiles are handwritten MNIST digits.

**Data generation** ([setup-dataset.py](setup-dataset.py)):
- `puzzle_mnist.py.generate_random_configs()` samples random permutations of the 9 tiles.
- `successors()` returns valid next states (one tile slide).
- Saved as `data/puzzle-mnist-3-3.npz` with arrays `pres` and `sucs`.

**Feature representation** ([strips.py](strips.py) `puzzle()`):
- `p.to_objects(configs, width, height)` converts a configuration to `(N_tiles, tile_feature_dim)`.
- For a 3×3 puzzle: 9 object slots, each containing the flattened pixel values of one tile.
- **Pre-encoder disabled** (`preencoder_layers=0`).

**Activation**: `puzzle_activation` — applies `rounded_softmax` over tile labels (9 categories) and over grid positions (2 groups of 3 bins each).

**Training output path**: `out/puzzle_FirstOrderSAE_mnist_3_3_None_None_None_{num_examples}/`

---

### 4.2 Blocksworld

**Domain**: Photorealistic stacking blocks (5 blocks, 3 towers). Pre-generated dataset from the original FOSAE paper.

**Data download** ([setup-dataset.py](setup-dataset.py) `blocksworld()`):
- Downloads `blocks-5-3.npz` from GitHub releases.
- NPZ fields: `images (N, num_objs, H, W, 3)`, `bboxes (N, num_objs, 4)`, `transitions (2T,)`, `picsize (2,)`.

**Feature representation** ([strips.py](strips.py) `blocksworld()`):
```
preprocess(images) / 256          → (N, num_objs, 32, 32, 3) float32 in [0,1]
images.reshape(N, num_objs, 3072) → pixel features

bboxes_to_onehot(bboxes, X=60, Y=40):
  bboxes_grid = bboxes // 5       → grid coordinates (5px resolution)
  x1_onehot (60 bins) | y1_onehot (40 bins)
  x2_onehot (60 bins) | y2_onehot (40 bins)  → 200-dim onehot vector

feature_dim = 3072 + 200 = 3272
```

Each state: shape `(num_objs, 3272)`.

**Activation**: `blocks_activation` — sigmoid on patch pixels + `rounded_softmax` on each of the 4 bbox onehot groups.

**Training output path**: `out/blocks-5-3/{sae_path}/`

---

### 4.3 Labeled Objects (NEW — this fork)

**Domain**: Arbitrary natural images from COCO, with objects detected and labeled by a VLM.

**Motivation**: Extends FOSAE to real-world visual domains without manual feature engineering. A VLM (Unsloth + Qwen3-VL) generates `class` labels and `bbox` coordinates for each image, which are then fed to FOSAE using the same blocksworld-style encoding.

**Pipeline overview**:
```
COCO images
  → dataset_gen.ipynb  (VLM annotation via Unsloth)
  → data/gen/fosae_labeled_dataset_unsloth.json
  → data/gen/raw_images/*.jpg
```

**JSON dataset format** (`fosae_labeled_dataset_unsloth.json`):
```json
[
  {
    "image": "000000001108.jpg",
    "objects": [
      {"class": "dog",  "bbox": [120, 45, 310, 280]},
      {"class": "table","bbox": [10,  200, 400, 350]}
    ]
  },
  ...
]
```
Each `bbox` is `[x1, y1, x2, y2]` in original image pixel coordinates.

**Feature representation** ([latplan/puzzles/puzzle_labeled_objects.py](latplan/puzzles/puzzle_labeled_objects.py)):

Objects are sorted by descending area (largest first) for stable slot ordering, then padded to `MAX_OBJECTS=10` slots with zeros.

```
entry_to_state():
  _crop_object(image, bbox) → 32×32×3 RGB patch (uint8)
  _scale_bbox_to_canvas(bbox, img_w, img_h) → canvas coords (200×300 canvas, 5px grid)

→ patches: (10, 32, 32, 3) uint8
→ bboxes:  (10, 4) uint16  (canvas pixel coords)
```

Then in [strips.py](strips.py) `labeled_objects()`, identical to blocksworld:
```
images.astype(float32)/256 → preprocess → reshape(N, 10, 3072)
bboxes_to_onehot(bboxes, X=60, Y=40) → reshape(N, 10, 200)
states = concat([images, bboxes_onehot], axis=-1)  → (N, 10, 3272)
```

**Transitions**: `build_transitions(states, mode)` supports:
- `sequential` → `state[i] → state[i+1]` for all `i` (N−1 pairs)
- `all_pairs` → every ordered pair `(i, j)` with `i ≠ j` (N×(N−1) pairs) — **default**

`all_pairs` is preferred to maximize training signal when the dataset is small.

**Saved artifact**: `out/labeled_objects/{run_tag}/object_names.json` — maps each state index to its per-state object name list (used by `extract_fol.py` for interpretable predicate labels).

**Training output path**: `out/labeled_objects/FirstOrderSAE_U{U}_A{A}_P{P}[_n{max_images}]/`

---

### 4.4 VidVRD (NEW — this fork)

**Domain**: ImageNet Video Visual Relationship Detection (1000 videos, 35 object categories, bounding boxes per frame).

**Data download**:
```bash
bash sh/download_vidvrd.sh                # downloads ~7 GB, extracts frames at 1fps
bash sh/download_vidvrd.sh --annotations-only   # annotations only (3 MB, no video)
```
Requires: `wget`, `ffmpeg`, `unzip`. Output:
```
data/vidvrd/annotations/<video_id>.json   # one JSON per video
data/vidvrd/frames/<video_id>/<000001>.jpg
```

**Annotation format** (one JSON per video):
```json
{
  "video_id": "ILSVRC2015_train_00005003",
  "width": 1920, "height": 1080, "frame_count": 219,
  "subject/objects": [{"tid": 0, "category": "dog"}, ...],
  "trajectories": [
    [{"tid": 0, "bbox": {"xmin": 672, "ymin": 560, "xmax": 781, "ymax": 693}}],
    ...
  ]
}
```
`bbox` values are pixel coordinates in the original video resolution.

**Feature representation** ([latplan/puzzles/puzzle_vidvrd.py](latplan/puzzles/puzzle_vidvrd.py)):

Identical pipeline to labeled_objects (re-uses `_crop_object`, `_scale_bbox_to_canvas`):
```
per frame: sort tracked objects by area → take top MAX_OBJECTS=10
→ patches: (10, 32, 32, 3) uint8
→ bboxes:  (10, 4) uint16  (CANVAS 200×300, 5px grid → X=60, Y=40)
→ feature dim = 3272  (identical to blocksworld and labeled_objects)
```

**Transitions**: `build_transitions(states, frame_ids, mode='sequential')` only pairs consecutive frames within the same video (cross-video pairs excluded). This gives semantically valid transitions (object motion between adjacent frames).

**Training**:
```bash
python strips.py learn vidvrd FirstOrderSAE 40 2 20
```

**Training output path**: `out/vidvrd/FirstOrderSAE_U{U}_A{A}_P{P}/`

---

## 5. Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ Data Generation / Download                                          │
│                                                                     │
│  puzzle:          setup-dataset.py puzzle mnist 3 3 5000           │
│                   → data/puzzle-mnist-3-3.npz                       │
│                                                                     │
│  blocksworld:     setup-dataset.py blocksworld blocks-5-3           │
│                   → data/blocks-5-3.npz                             │
│                                                                     │
│  labeled_objects: notebooks/dataset_gen.ipynb                       │
│                   → data/gen/fosae_labeled_dataset_unsloth.json     │
│                   → data/gen/raw_images/*.jpg                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Feature Extraction  (strips.py domain fn)                           │
│                                                                     │
│  raw data → object patches + bboxes                                 │
│           → preprocess (float32 / 256)                              │
│           → bboxes_to_onehot(bboxes, X, Y)                          │
│           → concat → states (N, num_objs, feature_dim)             │
│           → 90% train / 5% val / 5% test                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Model Training  (latplan/model.py FirstOrderSAE)                    │
│                                                                     │
│  Input (batch, N_obj, feat_dim)                                     │
│    → PreEncoder (optional)                                          │
│    → Attention  (batch, U, A, N_obj)                                │
│    → einsum → args_enc (batch, U, A, D)                             │
│    → Conv1D → Gumbel-Softmax → latent (batch, U×P)                 │
│    → Decoder → reconstruction                                       │
│                                                                     │
│  Loss: BCE(input, recon) + zerosuppress * L1(latent)                │
│  → simple_genetic_search → best model saved to out/.../             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Artifacts                                                           │
│                                                                     │
│  net0.h5            trained weights                                 │
│  aux.json           hyperparameters + metadata                      │
│  states.csv         binary latent codes for all training states     │
│  actions.csv        (pre_latent, suc_latent) transition pairs       │
│  *.png              reconstruction visualizations                   │
│  object_names.json  (labeled_objects only) per-state object labels  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FOL Extraction  (extract_fol.py)                                    │
│                                                                     │
│  attention = ae.encode_attention(states)  # (N, U, A, N_obj)        │
│  latent    = ae.encode(states)            # (N, U×P)                │
│  bindings  = argmax(attention, axis=-1)   # (N, U, A)               │
│                                                                     │
│  For each state i, unit u, predicate p:                             │
│    if latent[i, u, p] == 1:                                         │
│      pred_p(obj_binding[i,u,0], ...) = TRUE                         │
│                                                                     │
│  → fol_predicates.json / .txt                                       │
│  → predicate_analysis.json                                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Installation

**Requirements**: Python 3.6+, TensorFlow 1.15.x, Keras 2.2.x, CUDA 10.0 (GPU).

**Setup**:
```bash
bash install.sh                          # create conda env + install package + Keras config
conda activate latplan
bash smoke_test.sh                       # verify imports work
```

Done. All dependencies (including Unsloth for VLM annotation) install in the **same** `latplan` env.

**Manual install** (if needed):
```bash
conda env create -f environment.yml
conda run -n latplan pip install -e .
mkdir -p ~/.keras && cp keras-tf.json ~/.keras/keras.json
```

---

## 7. Training

All training via [strips.py](strips.py):
```bash
python strips.py learn <domain> [args...]
```

### 7.1 Dataset Setup

```bash
python setup-dataset.py puzzle mnist 3 3 5000         # puzzle
python setup-dataset.py blocksworld blocks-5-3        # blocksworld
bash sh/download_vidvrd.sh [--annotations-only]       # VidVRD
# labeled_objects: run notebooks/dataset_gen.ipynb (generates JSON + images automatically)
```

### 7.2 Training Commands

**Puzzle**:
```bash
python strips.py learn puzzle mnist 3 3 5000
```

**Blocksworld**:
```bash
python strips.py learn blocksworld blocks-5-3
```

**Labeled Objects** (COCO, VLM-annotated):
```bash
python strips.py learn labeled_objects FirstOrderSAE None None None None None \
    ./data/gen/fosae_labeled_dataset_unsloth.json ./data/gen/raw_images \
    all_pairs 2000 100
# Smaller experiment: 100 images, 2000 epochs
```

**VidVRD** (video frames):
```bash
python strips.py learn vidvrd FirstOrderSAE 40 2 20 None None None sequential 5000
# Sequential transitions (consecutive frames = real state change)
```

**Mode flags** (first arg):
- `learn` — train + dump states/actions
- `learn+plot` — + reconstruction plots
- `learn+dump` — + export CSV
- `learn+summary` — + print metrics
- Combine with `+`: `learn+plot+dump`

### 7.3 Hyperparameter Search

Default: `parameters` dict contains single-element lists (no search, `limit=1`).

To enable grid search: replace with multi-element lists in [strips.py](strips.py), then set `limit=N` in `run()` call.

```python
parameters = {
    'U': [20, 40, 80],
    'A': [2, 3],
    'P': [10, 20],
    ...
}
```

---

## 8. FOL Extraction

[extract_fol.py](extract_fol.py) loads a trained model and extracts human-readable FOL predicates.

### 8.1 How Extraction Works

```python
ae = latplan.model.load(model_dir)   # reads aux.json → instantiates class
ae.load()                             # loads net0.h5 weights
ae.build_aux(input_shape)             # builds attention_encoder model

attention = ae.encode_attention(data)  # (N, U, A, N_obj)
latent    = ae.encode(data)            # (N, U*P)

# Hard binding: which object does each argument slot attend to?
bindings = argmax(attention, axis=-1)  # (N, U, A)  integer indices

# For each state, unit, predicate:
# if latent[i, u, p] == 1: emit atom  pred_p(obj_names[bindings[i,u,0]], ...)
```

Atoms are filtered by a **confidence threshold** on the minimum attention weight across all argument slots of a unit (default `0.5`). Low-confidence bindings are skipped.

### 8.2 Commands

```bash
# Puzzle domain
python extract_fol.py out/puzzle_FirstOrderSAE_mnist_3_3_None_None_None_5000 \
    --domain puzzle --num 50 --output ./fol_output

# Blocksworld
python extract_fol.py out/blocks-5-3/FirstOrderSAE_... \
    --domain blocks --track blocks-5-3 --num 100

# Labeled objects
python extract_fol.py out/labeled_objects/FirstOrderSAE_U40_A2_P20 \
    --domain labeled_objects \
    --dataset-path ./data/gen/fosae_labeled_dataset_unsloth.json \
    --images-dir   ./data/gen/raw_images \
    --num 50 --confidence 0.5
```

### 8.3 All Flags

| Flag | Default | Description |
|------|---------|-------------|
| `model_dir` | (required) | Path to trained model directory |
| `--domain` | `puzzle` | `puzzle`, `blocks`, or `labeled_objects` |
| `--type` | `mnist` | Puzzle type (puzzle domain only) |
| `--width` / `--height` | `3` | Puzzle grid size |
| `--track` | `blocks-5-3` | Blocksworld track name |
| `--dataset-path` | auto | JSON dataset path (labeled_objects) |
| `--images-dir` | auto | Raw images directory (labeled_objects) |
| `--num` | `50` | Number of states to extract |
| `--output` | `<model_dir>/fol_output` | Output directory |
| `--confidence` | `0.5` | Min attention confidence threshold |
| `--show-negated` | off | Include `pred_p(...) = FALSE` atoms |

### 8.4 Output Files

| File | Contents |
|------|----------|
| `fol_predicates.json` | Full extraction: per-state list of atoms with unit, predicate, argument names, truth value, confidence |
| `fol_predicates.txt` | Human-readable text: one atom per line per state |
| `predicate_analysis.json` | Activation rates: which predicates are always-on (>99%), always-off (<1%), variable |
| `state_image_mapping.json` | (labeled_objects only) Maps state index → original COCO filename + object names |

**Sample `fol_predicates.txt` output**:
```
State 0:
  pred_3(dog, table) = TRUE  [conf=0.92]
  pred_7(table, dog) = TRUE  [conf=0.88]
  pred_12(dog, dog) = FALSE  [conf=0.91]

State 1:
  pred_3(cat, chair) = TRUE  [conf=0.95]
  ...
```

---

## 9. Visualization

[visualize_fol.py](visualize_fol.py) generates per-state diagnostic figures.

### 9.1 Output Figures

For each visualized state `i`:

**`fol_state_{i}_recon.png`** — 2×2 grid:
- Top-left: original rendered scene
- Bottom-left: reconstructed scene
- Top-right: absolute difference (`|recon − original|`)
- Bottom-right: binary latent code heatmap (U rows × P columns)

**`fol_state_{i}_attention.png`** — attention heatmaps for the first 5 predicate units:
- Y-axis: argument slots (arg_0, arg_1, ...)
- X-axis: object names
- Color: attention weight (blue scale)
- Title: bound object indices + predicate truth vector for that unit

**`fol_state_{i}_predicates.png`** — text box with extracted FOL atoms (same as `fol_predicates.txt` for that state)

**`attention_overview.png`** — average attention patterns across all visualized states, one heatmap per unit.

### 9.2 Commands

```bash
# Puzzle
python visualize_fol.py out/puzzle_FirstOrderSAE_mnist_3_3_... \
    --domain puzzle --num 6 --output ./viz

# Blocksworld
python visualize_fol.py out/blocks-5-3/FirstOrderSAE_... \
    --domain blocks --num 6

# Labeled objects
python visualize_fol.py out/labeled_objects/FirstOrderSAE_U40_A2_P20 \
    --domain labeled_objects \
    --dataset-path ./data/gen/fosae_labeled_dataset_unsloth.json \
    --images-dir   ./data/gen/raw_images \
    --num 10 --output ./viz
```

---

## 10. Adding a New Dataset

This section is a step-by-step guide for developers who want to run FOSAE on a new visual domain.

### Step 1 — Create a data loader module

Create `latplan/puzzles/puzzle_<name>.py`. It must provide:

```python
# latplan/puzzles/puzzle_myname.py

import numpy as np
import os

PATCH_SIZE  = 32
MAX_OBJECTS = 10
CANVAS_H    = 200
CANVAS_W    = 300
PICSIZE     = [CANVAS_H, CANVAS_W, 3]


def build_dataset(dataset_path, images_dir,
                  num_objs=MAX_OBJECTS, patch_size=PATCH_SIZE,
                  skip_empty=True, max_images=None):
    """
    Load your data and return raw features.

    Returns
    -------
    images       : (N, num_objs, patch_size, patch_size, 3)  uint8
    bboxes       : (N, num_objs, 4)  uint16  pixel coords in CANVAS space
                   (x1, y1, x2, y2) already scaled to CANVAS_H x CANVAS_W
    object_names : list[list[str]]  shape (N, num_objs)
    image_ids    : list[str]  one identifier per state

    The caller (strips.py) will apply:
        images = images.astype(float32) / 256
        images = preprocess(images)
        picsize_grid = (np.array(PICSIZE) // 5).astype(int)
        Y, X = picsize_grid   # 40, 60
        bboxes_onehot = bboxes_to_onehot(bboxes, X, Y)
        states = concat([images.reshape(N, num_objs, -1),
                         bboxes_onehot.reshape(N, num_objs, -1)], axis=-1)
    """
    # ... your loading code ...
    return images, bboxes, object_names, image_ids


def build_transitions(states, mode="sequential"):
    """Build (pre, suc) pairs from state array. See puzzle_labeled_objects.py."""
    n = len(states)
    if mode == "sequential":
        return np.array([states[:-1], states[1:]])
    elif mode == "all_pairs":
        idx_pre, idx_suc = zip(*[(i, j) for i in range(n)
                                         for j in range(n) if i != j])
        return np.array([states[np.array(idx_pre)], states[np.array(idx_suc)]])
    else:
        raise ValueError(f"Unknown mode '{mode}'")
```

If your objects do **not** have bounding boxes, you can set all bboxes to zeros — the model will still learn from pixel patches alone (the bbox onehot dimensions will be constant and effectively ignored).

### Step 2 — Add an activation function to model.py

In [latplan/model.py](latplan/model.py), inside `class BaseFirstOrderMixin`, add a method after `labeled_objects_activation`:

```python
def myname_activation(self, input_shape):
    """Activation for myname domain."""
    # For the standard blocksworld-style feature layout (patches + bbox onehot):
    # just reuse blocks_activation.  If your feature layout is identical, do:
    return self.blocks_activation(input_shape)

    # Or, if your output is pure float (no onehot), use plain sigmoid:
    # def obj_activation(x):
    #     return wrap(x, K.sigmoid(x), name="obj_activation")
    # return obj_activation
```

Also add a renderer for visualization (optional but recommended):

```python
def myname_renderer(self):
    """Reconstruct a scene from object features for visualization.
    Reuse blocks_renderer if feature layout is identical."""
    return self.blocks_renderer()
```

If needed, set `default_parameters["picsize"]` and `default_parameters["picsize_grid"]` in the training function (Step 3) so the renderer knows the canvas size.

### Step 3 — Add a training function to strips.py

In [strips.py](strips.py), add a function following the `labeled_objects()` pattern:

```python
def myname(aeclass="FirstOrderAE", U=None, A=None, P=None,
           num_objects=None, dataset_path=None, images_dir=None,
           transition_mode="all_pairs", epoch=5000,
           max_images=None, batch_size=None):
    from latplan.puzzles.puzzle_myname import (
        build_dataset, build_transitions, MAX_OBJECTS, PICSIZE)
    from latplan.puzzles.util import preprocess
    import json as _json

    # Override hyperparameters
    for name, value in dict(U=U, A=A, P=P).items():
        if value is not None:
            parameters[name] = [value]
    default_parameters["aeclass"]    = aeclass
    default_parameters["activation"] = "self.blocks_activation"  # or "self.myname_activation"
    default_parameters["epoch"]      = epoch
    # Enable pre-encoder for complex features
    parameters['preencoder_layers']            = [2]
    parameters['preencoder_dimention']         = [128]
    parameters['preencoder_output_activation'] = [("linear", "MSE")]
    if batch_size is not None:
        default_parameters["batch_size"] = batch_size

    num_objs = num_objects or MAX_OBJECTS

    # Load data
    images, bboxes, all_object_names, image_ids = build_dataset(
        dataset_path=dataset_path, images_dir=images_dir,
        num_objs=num_objs, max_images=max_images)
    num_states = len(images)

    # Standard preprocessing (identical to blocksworld / labeled_objects)
    picsize      = np.array(PICSIZE)
    picsize_grid = (picsize // 5).astype(int)
    Y, X         = picsize_grid[0], picsize_grid[1]
    default_parameters["picsize_grid"] = list(map(int, picsize_grid))
    default_parameters["picsize"]      = list(map(int, picsize))

    images = images.astype(np.float32) / 256
    images = preprocess(images)
    bboxes_onehot = bboxes_to_onehot(bboxes, X, Y)
    all_states = np.concatenate(
        (images.reshape((num_states, num_objs, -1)),
         bboxes_onehot.reshape((num_states, num_objs, -1))), axis=-1)
    del images, bboxes_onehot

    # Transitions + splits
    transitions = build_transitions(all_states, mode=transition_mode)
    states      = transitions.reshape((transitions.shape[1] * 2, num_objs, -1))
    if num_states < 100:
        train = val = test = all_states
    else:
        train = states[:int(len(states)*0.9)]
        val   = states[int(len(states)*0.9):int(len(states)*0.95)]
        test  = states[int(len(states)*0.95):]

    # Train
    out_path = os.path.join(OUT_DIR, "myname", aeclass)
    os.makedirs(out_path, exist_ok=True)
    ae = run(out_path, train, val, parameters)
    show_summary(ae, train, test)
    plot_autoencoding_image(ae, test, train, "blocks")  # or "myname" if you added renderer
    dump_states(ae, all_states)
    dump_actions(ae, transitions)

    # Save object names
    names_path = os.path.join(out_path, "object_names.json")
    with open(names_path, "w") as f:
        _json.dump({"image_ids": image_ids, "object_names": all_object_names}, f, indent=2)
```

### Step 4 — Add data loaders to extract_fol.py and visualize_fol.py

In [extract_fol.py](extract_fol.py):

```python
def load_myname_data(model_dir, dataset_path=None, images_dir=None, num_examples=None):
    from latplan.puzzles.puzzle_myname import build_dataset, PICSIZE
    from latplan.puzzles.util import preprocess
    from strips import bboxes_to_onehot
    # ... same pattern as load_labeled_objects_data() ...
    return states, per_state_names, image_ids
```

Add `"myname"` to the `--domain` choices and add the corresponding `elif` branch in `main()`.

In [visualize_fol.py](visualize_fol.py):

```python
def load_myname_data_and_renderer(ae, model_dir, num=10, dataset_path=None, images_dir=None):
    # ... same pattern as load_labeled_objects_data_and_renderer() ...
    render_fn, _ = ae.blocks_renderer()  # or ae.myname_renderer()
    return states, flat_names, render_fn, per_state_names
```

Add `"myname"` to `--domain` choices and the corresponding branch in `main()`.

### Step 5 — Generate your dataset and train

```bash
# Generate (or provide your own JSON + images)
python setup-dataset.py myname  # if you added a function there

# Train
python strips.py learn myname FirstOrderAE \
    None None None None None \
    ./data/my_dataset.json ./data/my_images \
    all_pairs 3000 200

# Extract FOL
python extract_fol.py out/myname/FirstOrderAE \
    --domain myname --num 50

# Visualize
python visualize_fol.py out/myname/FirstOrderAE \
    --domain myname --num 10
```

---

## 11. Output Reference

Each training run creates a directory under `out/`. Contents:

| File | Written by | Contents |
|------|-----------|---------|
| `net0.h5` | model.py | Trained Keras weights |
| `aux.json` | model.py | All hyperparameters + class name + metadata |
| `performance.json` | model.py | MSE/BCE on train/val under noise variations |
| `parameter_count.json` | model.py | Trainable parameter counts per layer |
| `logs/` | model.py | TensorBoard event files |
| `states.csv` | strips.py | Binary latent codes (U×P bits) for all training states |
| `actions.csv` | strips.py | Concatenated (pre_latent ++ suc_latent) for all transitions |
| `all_states.csv` | strips.py | (blocksworld) Binary codes for entire dataset |
| `all_actions.csv` | strips.py | (blocksworld) All transitions |
| `autoencoding_train.png` | strips.py | Input / attention / latent / recon grid for train set |
| `autoencoding_test.png` | strips.py | Same for test set |
| `render_train.png` | strips.py | Domain-rendered reconstruction (not raw features) |
| `render_test.png` | strips.py | Same for test set |
| `booleans_test.png` | strips.py | Per-predicate positive/negative example patches |
| `object_names.json` | strips.py | (labeled_objects) Per-state semantic object labels |

After running `extract_fol.py`, a `fol_output/` subdirectory is created:

| File | Contents |
|------|---------|
| `fol_predicates.json` | Structured extraction results |
| `fol_predicates.txt` | Human-readable atoms |
| `predicate_analysis.json` | Predicate activation statistics |
| `state_image_mapping.json` | (labeled_objects) State → COCO filename mapping |

After running `visualize_fol.py`, a `fol_visualizations/` subdirectory is created:

| File | Contents |
|------|---------|
| `fol_state_{i}_recon.png` | Reconstruction diagnostic for state `i` |
| `fol_state_{i}_attention.png` | Attention heatmaps for state `i` |
| `fol_state_{i}_predicates.png` | FOL text for state `i` |
| `attention_overview.png` | Average attention across all states |

---

## 12. Citation

If you use this code, please cite the original FOSAE paper:

```bibtex
@inproceedings{asai2019fosae,
  title     = {Unsupervised Grounding of Plannable First-Order Logic Representation from Images},
  author    = {Asai, Masataro},
  booktitle = {Proceedings of the 29th International Conference on Automated Planning and Scheduling (ICAPS)},
  year      = {2019},
  url       = {https://arxiv.org/abs/1902.08093}
}
```

Original repository: [guicho271828/latplan-fosae](https://github.com/guicho271828/latplan-fosae)

This fork (labeled-fosae) extends FOSAE with VLM-annotated COCO support. If you build on the labeled-objects domain, please also acknowledge this fork.
