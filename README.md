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

### 5.1. Video → Model: how a frame becomes a sample

The video pipeline (`vidvrd` / `actiongenome`) confused users in earlier iterations.
This section explains exactly what the model sees per training step and how the
temporal structure does (and does NOT) flow into training.

**Step 1 — annotation-driven frame loading**

`build_dataset(video_id_filter=..., fps=30)` reads the annotation JSON, then for
*each annotated frame* (one entry per source-PTS index in `trajectories`):

```
images:     (N, num_objs=10, PATCH_SIZE=32, 32, 3)  uint8
bboxes:     (N, num_objs=10, 4)                     uint16 (in 200×300 canvas)
frame_ids:  list[str]  '<video_id>/<NNNNNN>'        — preserves video order
```

N = the count of annotated frames present on disk. For a 30 fps VidVRD video
with 100 annotated frames, N = 100.

**Step 2 — per-object feature build** (`strips.py::vidvrd()` / `actiongenome()`)

```
images   → float32/256 + preprocess()              → (N, 10, 32, 32, 3)
bboxes   → bboxes_to_onehot(bboxes, X=60, Y=40)    → (N, 10, 200)
        concat                                       → (N, 10, 3272)
```

3072 pixel features per slot + 200 one-hot bbox features = **3272-dim per-object
feature vector**. One *state* = `(10, 3272)` = the whole annotated frame
represented as ten object slots (some may be zero-padded if the frame has < 10
visible objects).

**Step 3 — transition pairing** (`build_transitions(mode='sequential')`)

For every consecutive pair of states from the **same video** (i.e. `frame_ids[i]`
and `frame_ids[i+1]` share the `<video_id>` prefix):

```
pres.append(all_states[i])
sucs.append(all_states[i+1])

transitions = np.array([pres, sucs])    # shape (2, M, 10, 3272)
```

M = number of sequential transitions. For N=100 annotated frames in **one**
video, M = 99 (every frame except the first contributes a pre→suc pair).

**Step 4 — flatten back to a state pool** (here's the surprise)

```python
states = transitions.reshape((M * 2, num_objs, -1))   # (198, 10, 3272)
```

The 99 transition pairs are flattened into 198 individual states. The 90/5/5
split then runs over this **flat pool**.

> **⚠ Critical clarification.** From the model's point of view, a *sample* is a
> single state of shape `(num_objs, feature_dim)` — **NOT** a sequence of frames.
> `transition_mode` controls which states end up in the training pool, not how
> many frames the model sees at once. There is no temporal recurrence.

**Step 5 — per-state autoencoder training**

The encoder maps `(10, 3272) → (U, P)` latent predicate codes via per-unit
object attention. The decoder reconstructs the same state. The loss is purely
reconstruction (BCE on patch pixels + per-component MSE on bbox one-hot) plus
the `zerosuppress` L1 regularizer on the latent. **No action label, no temporal
loss, no validation of action accuracy.**

**Step 6 — action emerges post-training**

Only after the autoencoder is trained do transition pairs come back into play:

```python
z_pre = encoder(pres)
z_suc = encoder(sucs)
action_latent = z_suc - z_pre          # used by dump_actions / extract_fol
```

This is where temporal coherence matters. Sequential pairs encode real
frame-to-frame change. Random `all_pairs` pairs encode fictitious transitions
and the extracted "action" is meaningless.

**What this means in practice**

- For the smallest "can FOSAE learn from video frames?" overfit test, only
  `transition_mode='sequential'` is needed. `all_pairs` is **not** a comparison
  experiment at the model-training level — both modes produce the same kind of
  per-state autoencoder loss; they differ only in which states populate the
  pool and in the post-hoc action extraction.
- The `< 100 states` branch in `strips.py` short-circuits to
  `train = val = test = all_states` — for tiny single-video subsets this means
  `transition_mode` has **zero** effect on training (the flattened pool is
  ignored; the model trains directly on the N raw states).
- To make 2a-vs-2b a meaningful TRAINING comparison would require ≥100 source
  states (or removing the short-circuit). To meaningfully compare the
  *action-extraction* output (post-train), tiny N is fine.

**Reproducing the smallest test**

```bash
# Bake one sheep video at 30 fps (100 annotated frames → 100 states)
python3 setup-dataset.py video_vidvrd sheep \
    --video-id ILSVRC2015_train_00256010 \
    --fps 30 \
    --out-name sheep-ILSVRC2015_train_00256010-30fps

# Train sequential overfit (NO_EARLYSTOP=1 so full 1000 epochs run)
NPZ_PATH=data/npz/video/vidvrd/overfit/sheep-ILSVRC2015_train_00256010-30fps.npz \
DOMAIN=vidvrd \
TRANSITION_MODE=sequential \
EPOCH=1000 NO_EARLYSTOP=1 \
bash sh/submit.sh
```

The training history CSV (`<out_dir>/training_history.csv`) should show
`val_loss` decreasing monotonically (with `NO_EARLYSTOP=1` set) and the
reconstruction PNGs in `<out_dir>` should look visually close to the input
frames if the model truly learned.

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

---

## Hyperparameter reference (2026-05-14)

Distilled from the original FOSAE codebase ([guicho271828/latplan-fosae](https://github.com/guicho271828/latplan-fosae); local clone at `/home/panoslat/Dev/Thesis/FOSAE/latplan-fosae`) and the paper [Unsupervised Grounding of Plannable First-Order Logic Representation from Images (Asai 2019, arXiv:1902.08093)](https://arxiv.org/abs/1902.08093). Use this section as the source of truth when configuring a run.

### Original defaults — `default_parameters` (`strips.py:39-52`, byte-identical to upstream)

These are the values that apply to **every** parameter not over-ridden by the tuning grid or CLI args.

| Key | Value | Role |
|-----|-------|------|
| `epoch` | `int(os.environ.get("EPOCH", 1000))` | training epochs (env-overridable, this fork) |
| `batch_size` | `1000` | minibatch size |
| `optimizer` | `"radam"` | Rectified Adam |
| `max_temperature` | `5.0` | Gumbel-softmax initial temperature |
| `min_temperature` | `0.7` | Gumbel-softmax final temperature |
| `N` | `None` | per-PU output length (resolved later as `U*A`) |
| `M` | `2` | binary latent vocabulary (=2 → boolean propositions) |
| `train_gumbel` | `True` | inject Gumbel noise during training |
| `train_softmax` | `True` | continuous latent during training |
| `test_gumbel` | `False` | deterministic latent at inference |
| `test_softmax` | `False` | discrete (rounded) latent at inference |
| `dropout_z` | `False` | dropout on latent layer (paper kept off) |

### Original tuning grid — `parameters` (`strips.py:75-95`, byte-identical to upstream after 2026-05-14)

`simple_genetic_search` samples up to **`LIMIT`** configs from this Cartesian space. Upstream paper used **`LIMIT=300`** (`run()` in `latplan-fosae/strips.py:174`). This fork makes `LIMIT` env-overridable (default `LIMIT=1` for fast smokes).

| Key | Search values | Note |
|-----|---------------|------|
| `beta` | `-0.3, -0.1, 0.0, 0.1, 0.3` | KL-style scalar coefficient |
| `lr` | `0.1, 0.01, 0.001, 0.0001` | learning rate |
| `U` | `20, 40, 80` | number of Predicate Units |
| `A` | `2, 3, 4` | arity of every predicate |
| `P` | `10, 20, 40, 80, 160, 320` | predicates per unit; total propositions = `U × P` |
| `layer` | `50, 100, 400, 1000` | hidden dim of the FC encoder/decoder |
| `dropout` | `0.3, 0.4, 0.5` | encoder/decoder dropout rate |
| `noise` | `0.1, 0.2, 0.4` | additive input noise (denoising AE) |
| `zerosuppress` | `0.0, 0.05, 0.1, 0.2, 0.5` | latent-sparsity penalty weight |
| `zerosuppress_delay` | `0.05, 0.1, 0.2, 0.3, 0.5` | warm-up fraction before `zerosuppress` kicks in |
| `preencoder_dimention` | `10, 25, 50, 100, 200, 400` | preencoder bottleneck width (note: paper's spelling is "dimention", typo preserved by upstream) |
| `preencoder_layers` | `0, 1, 2` | Conv1D layers in the preencoder (0 = preencoder disabled) |
| `preencoder_l1` | `0.0, 1e-5, 1e-4, 1e-3, 1e-2` | L1 regulariser on preencoder output |
| `preencoder_delay` | `0.05, 0.1, 0.2, 0.3, 0.5` | warm-up fraction before preencoder loss is added |
| `preencoder_output_activation` | `("relu","MSE"), ("linear","MSE"), ("sigmoid","MSE"), ("sigmoid","BCE")` | preencoder output activation + loss head |
| `loss` | `"BCE"` | reconstruction loss head |
| `eval` | `"MSE"` | validation metric |

### Per-domain overrides (applied by the task function in `strips.py`)

| Domain | Function `strips.py` | Override |
|--------|----------------------|----------|
| `puzzle` (mnist / mandrill / lenna / spider / digital) | `puzzle()` | `preencoder_dimension=0`, `preencoder_layers=0`, `preencoder_l1=0` — **preencoder disabled** (15-dim feature input is too small to benefit) |
| `blocksworld` | `blocksworld()` | `picsize_grid`, `picsize` injected from npz; activation `self.blocks_activation` |
| `labeled_objects` / `vidvrd` / `actiongenome` | (realistic-image domains) | `preencoder_layers=2`, `preencoder_dimention=256`, `preencoder_output_activation=("linear","MSE")`, `lr=0.0001` — paper-grade preencoder for 3272-dim feature input |

### Paper-published "final picks" (Table 1, Fig 12 of 1902.08093)

These are the **best** configs the paper reports after running the full grid above with `LIMIT=300`. They are **NOT defaults** — they are the *answers* to the grid search, useful as starting points for follow-up tuning.

| Domain | `U` | `A` | `P` | propositions `U·P` | source |
|--------|-----|-----|-----|---------------------|--------|
| 8-puzzle (3×3, mnist) | **25** | **2** | **50** | 1 250 | paper §6, Table 1 |
| 8-puzzle (3×3) Pareto minimum | 9 | 2 | 6 | 54 | paper §6, Fig 12 |
| blocksworld (5 blocks × 3 stacks) | **10** | **2** | **100** | 1 000 | paper §6, Table 1; `train_all_blocks.sh` |
| 8-puzzle extreme arity | 1 | 9 | up to 400 | up to 400 | `train_all_contour.sh` (Fig 8 contour study) |

### Reasonable starting points for **new experiments** in this fork

Pick a column by your data size, not by aesthetics. `propositions` is the discrete-capacity knob — too few starves the model, too many lets it memorise.

| Use case | `U` | `A` | `P` | epoch | `lr` | preencoder | rationale |
|----------|-----|-----|-----|-------|------|------------|-----------|
| Paper-faithful 8-puzzle baseline | 25 | 2 | 50 | 1 000 | 0.001 | off | matches Table 1 |
| Paper-faithful blocksworld baseline | 10 | 2 | 100 | 1 000 | 0.001 | off | matches Table 1 |
| Cheap smoke (any domain) | 10 | 2 | 20 | 100 | 0.001 | off | converges in minutes; lets you confirm pipeline before paying for a full run |
| Realistic-image domain (vidvrd / actiongenome / labeled_objects) | 40 | 2 | 20 | 2 000 | 0.0001 | layers=2 dim=256 act=("linear","MSE") | 3272-dim input → deeper preencoder + smaller lr |
| Aggressive video search | 80 | 2 | 40 | 5 000 | 0.0001 | layers=2 dim=256 | 3 200 propositions; gives FOSAE room for many object/relation predicates |
| Bigger-capacity safety net | 80 | 2 | 80 | 5 000 | 0.0001 | layers=2 dim=256 | 6 400 propositions; use when reconstruction stalls and you suspect bottleneck |

For any of these: **add `LIMIT=20+` to actually search**, otherwise `simple_genetic_search` runs exactly one trial whose seed determines success.

### Environment-variable knobs (this fork)

| Env var | Default | Effect |
|---------|---------|--------|
| `EPOCH` | `1000` | overrides `default_parameters['epoch']` (`strips.py:40`) |
| `LIMIT` | `1` | bounds `simple_genetic_search` trial count (`strips.py:221`) |
| `OUT_DIR` | `<project>/out` | overrides output base; auto-joined with `<domain>/<type>/<run_tag>/` (`latplan/util/paths.py`) |
| `VIDVRD_STRICT_CATEGORY` | `1` | strict primary-subject filter for VidVRD category training |
| `AG_STRICT_CATEGORY` | `1` | same for ActionGenome |

### Sherlock command templates

Prereqs (fresh shell): `module restore fosae && source venv/bin/activate && cd $SCRATCH/panos/sgg-thesis && git pull`. Job submission via `sh/submit.sh`; each job's output lands in `OUT_DIR`.

```bash
# --- mnist puzzle paper baseline (LIMIT=300 trials, ~paper) ---
LIMIT=300 EPOCH=1000 \
  OUT_DIR=$SCRATCH/panos/sgg-thesis/out/baseline-paper-puzzle-mnist \
  TRAIN_CMD="python3 strips.py learn_plot puzzle FirstOrderAE mnist 3 3 None None None 20000" \
  TIME=72:00:00 \
  bash sh/submit.sh

# --- mnist puzzle fixed paper picks U=25 A=2 P=50, single trial ---
EPOCH=1000 \
  OUT_DIR=$SCRATCH/panos/sgg-thesis/out/baseline-fixed-puzzle-mnist \
  TRAIN_CMD="python3 strips.py learn_plot puzzle FirstOrderAE mnist 3 3 25 2 50 20000" \
  bash sh/submit.sh

# --- mandrill puzzle (same hyperparams) ---
EPOCH=1000 \
  OUT_DIR=$SCRATCH/panos/sgg-thesis/out/baseline-fixed-puzzle-mandrill \
  TRAIN_CMD="python3 strips.py learn_plot puzzle FirstOrderAE mandrill 3 3 25 2 50 20000" \
  bash sh/submit.sh

# --- blocksworld paper baseline (reproduce_plot replays best from grid_search.log) ---
MEM=64G EPOCH=1000 \
  OUT_DIR=$SCRATCH/panos/sgg-thesis/out/baseline-paper-blocks-5-3 \
  TRAIN_CMD="python3 strips.py reproduce_plot blocksworld FirstOrderSAE blocks-5-3 None None None 10000 BCE5" \
  TIME=72:00:00 \
  bash sh/submit.sh

# --- blocksworld fixed paper picks U=10 A=2 P=100, single trial ---
MEM=64G EPOCH=1000 \
  OUT_DIR=$SCRATCH/panos/sgg-thesis/out/baseline-fixed-blocks-5-3 \
  TRAIN_CMD="python3 strips.py learn_plot blocksworld FirstOrderAE blocks-5-3 10 2 100 6500" \
  bash sh/submit.sh

# --- modest search (LIMIT=20 trials, ~2-4 h on v100; populates grid_search.log) ---
LIMIT=20 EPOCH=1000 \
  OUT_DIR=$SCRATCH/panos/sgg-thesis/out/search-puzzle-mnist \
  TRAIN_CMD="python3 strips.py learn_plot puzzle FirstOrderAE mnist 3 3 None None None 20000" \
  TIME=8:00:00 \
  bash sh/submit.sh
```

Mode `learn_plot` writes `autoencoding_{test,train}{,_shuffled}.png` + `render_{test,train}{,_shuffled}.png` + `booleans_test.png` + `test*.pdf/.gv` decision-tree at the end. Mode `reproduce_plot` does the same after picking the best entry from a pre-existing `grid_search.log` (i.e. requires a prior `learn` / `learn_plot` job in the same `OUT_DIR`). Mode `learn` skips the plotting.
