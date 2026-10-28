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

- `model/` — FOSAE, the model (encoder, decoder, layers), forked from
  upstream Latplan. `model/train/strips.py` trains a model
  (`model/train/strips.py <mode> <task> ...`), `model/train/extract_fol.py`
  reads predicates out of a trained model, and `model/config.py` sets up the
  Keras session.
- `tools/` — the tooling: `tools/video/` converts each dataset's raw
  annotations into the common VidVRD-style relation format,
  `tools/setup-dataset.py` bakes a converted dataset into `.npz`, and
  `tools/analysis/` holds the training-run analysis scripts.
- `planner/` — the planners (bfs, pddl, ama3), the latent export
  (`export_latents.py`) and the evaluation, including `triplet_eval.py`
  (triplet mAP with the official evaluator of VidVRD-helper).
- `runs/` — what runs the pipeline on Sherlock: `runs/data/` downloads,
  unpacks and converts the datasets, `runs/eval/` exports and scores, and
  `runs/experiments/` holds one folder per evaluation experiment.
- `tests/` — the test suite.
- `data/`, `out/`, `logs/`, `venv/` — datasets, model outputs, Slurm logs and
  the Python 3.6 environment (`.venv-local` links to it). Not in git.
- `_attic/` — files set aside during the 2026-10 cleanup, kept recoverable.
  `workbench/ARCHIVE.md` lists every one and how to bring it back.

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
`planner/install_fd.sh` and `planner/install_roswell.sh`.

## Tests

```bash
export TMPDIR=/scratch/users/konkotsa/panos/sgg-thesis/.claude/tmp
.venv-local/bin/python -m unittest discover -s tests
```

## Triplet mAP

```bash
.venv-local/bin/python planner/triplet_eval.py --dataset se --out workbench/eval/triplet/<date>/
```

`--dataset` selects which converted dataset to score; `--out` is a directory
under the private `workbench/eval/`, since results are not part of this
repository.

## Datasets

Each dataset moves through download, prepare, screen, oracle, and verify
stages. `runs/data/` and `runs/eval/` hold the Slurm job for each stage that
needs a compute node rather than a login node.
