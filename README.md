# sgg-thesis

A Master's thesis fork of [guicho271828/latplan-fosae](https://github.com/guicho271828/latplan-fosae).
The question: can FOSAE (First-Order State AutoEncoder) learn a plannable,
symbolic relation representation from real-world video, not just puzzle
images? Four datasets carry a bake-to-evaluation path: Something-Else,
VidVRD, KIT Bimanual, and HA-ViD.

FOSAE reads a set of object patches per frame and learns a binary latent code
that names first-order predicates such as `pred_3(dog, table)`, with no
labels beyond reconstruction loss. The FOSAE paper is at
[arXiv:1902.08093](https://arxiv.org/abs/1902.08093).

## Layout

- `latplan/` — the model (encoder, decoder, training loop), forked from
  upstream.
- `tools/video/` — the dataset conversions: each dataset's raw annotations
  into the common VidVRD-style relation format.
- `tools/planner/` — the classical planner and its evaluation, including
  `triplet_eval.py` (triplet mAP against the official `ivtmetrics`-based
  evaluator).
- `sh/` — Slurm jobs for dataset download, unpack, and conversion.
- `experiments/` — one directory per experiment, each with its own `README.md`
  and run script.

This is the public half of a two-repository split. The private half
(`workbench/`, a nested git repository, git-ignored here) holds the thesis
notes, trained-model exports, and Sherlock job scripts. A clone of this
repository alone has the code but not the data, the notes, or the trained
weights.

## Install

```bash
conda env create -f environment.yml      # conda env `fosae`
conda run -n fosae pip install -e . --no-deps
```

On Sherlock, the canonical interpreter is `.venv-local`, a Python 3.6 virtual
environment pinned to match the cluster's `python/3.6.1` module (TensorFlow
1.15 needs it). `.venv-local/bin/python` is what every command below uses.

Fast Downward and Roswell, needed by the planner, install separately: see
`tools/planner/install_fd.sh` and `tools/planner/install_roswell.sh`.

## Tests

```bash
export TMPDIR=/scratch/users/konkotsa/panos/sgg-thesis/.claude/tmp
.venv-local/bin/python -m unittest discover -s tools/planner/tests
```

## Triplet mAP

```bash
.venv-local/bin/python tools/planner/triplet_eval.py --dataset se --out workbench/eval/triplet/<date>/
```

`--dataset` selects which converted dataset to score; `--out` is a directory
under the private `workbench/eval/`, since results are not part of this
repository.

## Datasets

Each dataset moves through download, prepare, screen, oracle, and verify
stages. `sh/` holds the Slurm job for each stage that needs a compute node
rather than a login node.
