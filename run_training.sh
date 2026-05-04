#!/bin/bash
# run_training.sh — Generic SLURM batch script for any FOSAE training run.
#
# Usage (direct sbatch with default vidvrd training):
#   sbatch run_training.sh
#
# Usage (custom command via env var):
#   sbatch --export=ALL,TRAIN_CMD="python3 strips.py learn vidvrd FirstOrderSAE 40 2 20 None None sequential 5000 None None 3 dog" run_training.sh
#
# Usage (preferred — through the wrapper):
#   bash sh/submit.sh                              # all-cat vidvrd
#   CATEGORY=person bash sh/submit.sh              # per-category
#   FPS=5 EPOCH=8000 GPUS=2 bash sh/submit.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/<job-name>.<JOBID>.out
#   seff <JOBID>

# ── Default SLURM directives (override via sbatch CLI / wrapper) ─────────────
#SBATCH --job-name=fosae-train
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/fosae-train.%j.out
#SBATCH --error=logs/fosae-train.%j.err
# GPU SKU constraint: uncomment + adjust if your account requires it
# Run `node_feat -p gpu | grep GPU_SKU` to list available SKUs.
##SBATCH --constraint="GPU_SKU:TESLA_V100_PCIE|GPU_SKU:TESLA_V100_SXM2"
##SBATCH --constraint="GPU_MEM:16GB"
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=yourname@stanford.edu

set -eo pipefail  # no -u: venv activate references unset vars

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Default training command (overridable via TRAIN_CMD env var) ─────────────
DEFAULT_TRAIN_CMD="python3 strips.py learn vidvrd FirstOrderSAE 40 2 20 None None sequential 5000 None None 3"
TRAIN_CMD="${TRAIN_CMD:-${DEFAULT_TRAIN_CMD}}"

# ── Header ────────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Job ID    : ${SLURM_JOB_ID:-N/A}"
echo "  Node      : ${SLURMD_NODENAME:-N/A}"
echo "  Started   : $(date)"
echo "  Dir       : ${PROJECT_DIR}"
echo "  TRAIN_CMD : ${TRAIN_CMD}"
echo "================================================================"

# ── Bootstrap environment ─────────────────────────────────────────────────────
source "${PROJECT_DIR}/sh/sherlock_env.sh"

# ── Verify GPU is visible ─────────────────────────────────────────────────────
echo ""
echo "--- GPU info ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    || echo "WARNING: nvidia-smi unavailable"
python3 - <<'PYEOF'
from tensorflow.python.client import device_lib
devs = [d.name for d in device_lib.list_local_devices()]
print("TF visible devices:", devs)
assert any("GPU" in d for d in devs), "No GPU detected -- job will be very slow on CPU"
PYEOF

# ── Prepare output / logs directories ────────────────────────────────────────
mkdir -p "${PROJECT_DIR}/out" "${PROJECT_DIR}/logs"

# ── Run training ──────────────────────────────────────────────────────────────
echo ""
echo "--- Training start: $(date) ---"
cd "${PROJECT_DIR}"
eval "${TRAIN_CMD}"
echo ""
echo "--- Training end: $(date) ---"

# ── Summarise output ──────────────────────────────────────────────────────────
echo ""
echo "--- Output files ---"
ls -lhR "${PROJECT_DIR}/out" 2>/dev/null | head -40
echo "================================================================"
