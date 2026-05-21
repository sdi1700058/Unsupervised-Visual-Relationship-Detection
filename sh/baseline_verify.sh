#!/usr/bin/env bash
# baseline_verify.sh — Regression sentinel for MNIST 8-puzzle FOSAE training.
#
# Reproduces the verified baseline params from the May 19 2026 archive at
# out/_archive/2026-05-20/mnist-noearlystop-20260519-055940/puzzle/mnist/
# FirstOrderSAE_U40_A2_P20/aux.json. That run reached
# `val_BCE = 1.038188e-07` at epoch 999 on a fresh build.
#
# Run after EVERY `git pull` on Sherlock to confirm the codebase still
# produces a converged MNIST baseline before trusting downstream
# video / hyperparameter results.
#
# Pinned hyperparameters (all set explicitly via env so future drift in the
# active `parameters` block in strips.py does NOT affect this sentinel):
#   aeclass             FirstOrderSAE
#   U=40 A=2 P=20       layer=200 (default)
#   lr                  0.001
#   max_temperature     1.0
#   min_temperature     0.7  (current default in strips.py:44)
#   zerosuppress        0.05
#   zerosuppress_delay  0.05
#   dropout             0.05
#   noise               0.05
#   preencoder_layers   0    (set by puzzle() automatically)
#   epoch               1000
#   batch_size          1000 (current default in strips.py:41)
#   NO_EARLYSTOP=1
#   num_examples        5000 (override with NUM_EXAMPLES env)
#
# Usage
# -----
#   bash sh/baseline_verify.sh                    # NUM_EXAMPLES=5000
#   NUM_EXAMPLES=6500 bash sh/baseline_verify.sh  # match archive exactly
#
# After the job COMPLETED:
#   J=<JOBID-from-output>
#   F=$(ls logs/baseline-mnist-sentinel-*.${J}.out | head -1)
#   OUT=$(grep '\[info\] Output dir:' "$F" | head -1 | awk '{print $NF}')
#   tail -1 "$OUT/training_history.csv" | awk -F, '{print "val_BCE=", $7}'
#   # Expect val_BCE ~ 1e-7 (within an order of magnitude). If much higher
#   # (e.g. 0.1+) the codebase has regressed — bisect against the May 19
#   # archive's working state.

set -eo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NUM_EXAMPLES="${NUM_EXAMPLES:-5000}"

echo "[baseline_verify] submitting MNIST 8-puzzle sentinel (NUM_EXAMPLES=${NUM_EXAMPLES})"
echo "[baseline_verify] reference: archive val_BCE = 1.038e-07 @ epoch 999"

NO_EARLYSTOP=1 \
EPOCH=1000 \
AUTO_RESOURCES=0 \
AECLASS=FirstOrderSAE \
DOMAIN=puzzle PUZZLE_TYPE=mnist WIDTH=3 HEIGHT=3 \
U=40 A=2 P=20 \
LR=0.001 \
MAX_TEMPERATURE=1.0 \
ZEROSUPPRESS=0.05 \
ZEROSUPPRESS_DELAY=0.05 \
DROPOUT=0.05 \
NOISE=0.05 \
NUM_EXAMPLES=${NUM_EXAMPLES} \
MEM=16G TIME=0:30:00 \
JOB_NAME="baseline-mnist-sentinel-$(date +%Y%m%d-%H%M%S)" \
bash "${PROJECT_DIR}/sh/submit.sh"

echo
echo "[baseline_verify] After 'seff <JOBID>' shows COMPLETED, check val_BCE via:"
echo
cat <<'POST'
  J=<JOBID>
  F=$(ls logs/baseline-mnist-sentinel-*.${J}.out | head -1)
  OUT=$(grep '\[info\] Output dir:' "$F" | head -1 | awk '{print $NF}')
  python3 tools/summarize_run.py "$OUT"
  # Reference (May 19 archive): val_BCE min ~ 1e-7. PASS if within an order of magnitude.
POST
