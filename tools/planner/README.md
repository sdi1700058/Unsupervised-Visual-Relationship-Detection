# Planner evaluation

Does FOSAE learn relations a classical planner can use?

The test is frame interpolation. Give the planner frame *i* and frame
*i+k-1* from a real video. It must reconstruct the *k-2* frames between them.
Score = how far the reconstructed bounding boxes sit from the real ones.

Reconstruction loss says the model can encode a frame. This says the model
learned how frames *follow* each other.

## Why a window and not the whole clip

A 135-frame clip needs a 134-step plan. Fast Downward does not solve that
over an 800-bit state space. A window of 5 frames needs a 4-step plan, which
takes seconds.

The window also pays off in sample count. One clip gives `N - k + 1` windows,
not one data point. A 135-frame clip at `k=5` gives 131 scored windows.

## The baseline matters more than the raw error

An error of 340 square pixels means nothing alone. Objects in short clips
often move close to a straight line, so drawing a line between the two given
frames already lands near the truth. Without a comparison you cannot tell a
model that learned dynamics from a task that was easy.

So every window also scores the straight line, and the headline number is

    mse_ratio = planner error / straight-line error

Below 1 means the planner used something the endpoints did not already give
away. Wider windows weaken the straight line and give the planner more room
to show a difference.

## Three methods

| Directory | What it does | Needs |
|---|---|---|
| `ama3/` | Drives the upstream latplan AMA3 pipeline. The PDDL comes from the author's lisp, so a result here carries the most weight. | Roswell, SBCL, the lisp binaries, Fast Downward |
| `pddl/` | Emits propositional STRIPS from the learned latents, then calls Fast Downward. | Fast Downward |
| `bfs/` | Breadth-first search over the latent space. No PDDL at all. | nothing beyond numpy |

`bfs` ignores preconditions, so it is a lower bound rather than a reading of
the learned schema. It earns its place by exercising the whole encode, decode
and scoring path with no external toolchain, which makes it the right first
smoke test.

Each directory holds one `planner.py` with a `run()` entry. Everything else
lives in `common/`.

## Layout

    plan_video.py           one window, one method
    eval_plannability.sh    slides windows across videos, writes summary.csv
    viz_plannability.py     turns summary.csv into figures
    install_fd.sh           builds Fast Downward
    install_roswell.sh      installs Roswell, SBCL, arrival, lisp binaries

    export_latents.py       run where the model is; writes an export
    common/encode.py        load the model, encode frames  (export side)
    common/decode.py        latents back to boxes          (export side)
    common/export.py        read an export                 (planner side)
    common/windows.py       window layout, straight-line baseline
    common/metrics.py       error, baseline comparison, temporal order
    common/harness.py       the shared run path for all three methods

A method supplies only `_solve(z_init, z_goal, ...)`. The harness does the
loading, decoding, scoring and writing, so the three stay comparable.

## Two stages, two machines

Encoding a frame needs keras, TensorFlow 1.15 and the model. Searching and
scoring need numpy. Fusing them would tie the planner to the machine that
holds the model, so they are split.

Stage one runs where the model lives, usually Sherlock:

    python3 tools/planner/export_latents.py <model_dir> -o dog.npz

That writes the encoded frames, the annotated boxes and the decoder output
for each frame. It is a few hundred kilobytes, so `scp` it and forget it.

Stage two runs anywhere, on any modern Python:

    python3 tools/planner/plan_video.py dog.npz --method bfs --init 0 --goal 4

An export also decouples the two Python versions. Training is stuck on
Python 3.6 or 3.7 because of TensorFlow 1.15; the planner is not.

Because a plan built from the training deltas can only reach states the
model reached, every latent on a plan is normally already in the export.
When one is not, the nearest row by Hamming distance stands in, and
`decode_fallbacks` in the metrics counts how often that happened. A run
leaning on that number is not trustworthy.

## Setup

    bash tools/planner/install_fd.sh        # Fast Downward, for pddl and ama3
    bash tools/planner/install_roswell.sh   # only for ama3

`install_roswell.sh` needs libpng headers for one of its lisp dependencies:

    sudo apt-get install -y libpng-dev

Scoring needs numpy. Figures also need matplotlib, and scipy makes the slot
matching faster:

    pip install -r tools/planner/requirements.txt

## Running

One window:

    python3 tools/planner/plan_video.py dog.npz --method bfs --init 0 --goal 4

Every window in one or more exports:

    bash tools/planner/eval_plannability.sh exports/ --window 5 --methods bfs,pddl
    python3 tools/planner/viz_plannability.py eval/planner/exports

## Output

    eval/planner/<name>/summary.csv                       one row per window
    eval/planner/<name>/<export>/<method>/win_<i>_<j>/    metrics.json, PDDL, plan
    eval/planner/<name>/viz/                              figures and captions

## Metrics

| Column | Meaning |
|---|---|
| `reachability` | the planner returned a plan |
| `plan_length` vs `expected_length` | a plan should take `k-1` steps to cross `k` frames |
| `bbox_mse` | error on the reconstructed frames, squared pixels |
| `baseline_mse` | the same error for the straight line |
| `mse_ratio` | **the headline number, below 1 is a win** |
| `temporal_order` | did the plan walk the video forwards. Diagnostic, not a headline |
| `decode_fallbacks` | latents the plan reached that the export did not hold. Should be 0 |

`temporal_order` does not catch a failure that `bbox_mse` misses, since each
plan step is already compared against its own frame. It tells you *why* a run
failed: high error with high order points at the decoder, high error with low
order means the plan wandered.

## Tests

    python3 -m unittest tools/planner/tests/test_pure_functions.py

38 tests on synthetic exports, including full runs through the harness.
No model, no GPU, no lisp, no scipy.
