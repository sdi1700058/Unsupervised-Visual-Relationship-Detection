#!/bin/bash
# run_training.sh — Submit with: sbatch run_training.sh
#
# Runs:
#   python strips.py learn labeled_objects FirstOrderSAE \
#       10 2 100 5 None None None sequential 5000 5000
#
# Before first submission, create the logs directory:
#   mkdir -p logs/
#
# Monitor after submission:
#   squeue -u $USER                          # check queue position / status
#   tail -f logs/fosae-learn.<JOBID>.out     # watch live output
#   seff <JOBID>                             # resource usage report (after job ends)
#   sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,MaxRSS

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=fosae-learn
#SBATCH --partition=gpu
#SBATCH --gpus=1
# GPU SKU constraint: disabled by default. The exact SKU names depend on which
# nodes your account can access — verify with:  node_feat -p gpu | grep GPU_SKU
# Then uncomment and fill in, e.g.:
##SBATCH --constraint="GPU_SKU:TESLA_V100_PCIE|GPU_SKU:TESLA_V100_SXM2|GPU_SKU:TESLA_P100_PCIE"
# Alternatively use GPU_MEM to require at least 16 GB VRAM:
##SBATCH --constraint="GPU_MEM:16GB"
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/fosae-learn.%j.out
#SBATCH --error=logs/fosae-learn.%j.err
# Uncomment and fill in to receive email alerts:
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=yourname@stanford.edu

set -eo pipefail  # no -u: venv activate script references unset vars

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Header ────────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Job ID   : ${SLURM_JOB_ID}"
echo "  Node     : ${SLURMD_NODENAME}"
echo "  Started  : $(date)"
echo "  Dir      : ${PROJECT_DIR}"
echo "================================================================"

# ── Bootstrap environment ─────────────────────────────────────────────────────
# Loads gcc/python/cuda/cuDNN modules and activates the latplan venv.
# sherlock_env.sh is tolerant of missing modules (prints warnings, does not exit).
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

# ── Prepare output directory ─────────────────────────────────────────────────
mkdir -p "${PROJECT_DIR}/out"

# ── Run training ──────────────────────────────────────────────────────────────
echo ""
echo "--- Training start: $(date) ---"
cd "${PROJECT_DIR}"
python3 strips.py learn labeled_objects FirstOrderSAE \
    10 2 100 5 None None None sequential 5000 5000 9000
echo ""
echo "--- Training end: $(date) ---"

# ── Summarize output ──────────────────────────────────────────────────────────
echo ""
echo "--- Output files ---"
ls -lh "${PROJECT_DIR}/out/labeled_objects/" 2>/dev/null \
    || echo "(out/labeled_objects/ not found)"
echo "================================================================"
