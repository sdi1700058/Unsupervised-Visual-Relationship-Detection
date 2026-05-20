#!/bin/bash
# run_training.sh — Generic SLURM batch script for any FOSAE training run.
#
# Preferred entry point: sh/submit.sh (sets all env vars + sbatch flags).
# Direct usage:
#   sbatch run_training.sh                                    # default vidvrd
#   sbatch --export=ALL,TRAIN_CMD="python3 strips.py learn ..." run_training.sh
#
# Env vars consumed by post-train hooks (set automatically by sh/submit.sh):
#   DOMAIN     — vidvrd | labeled_objects | puzzle | blocks (others skip hooks)
#   AECLASS    — model class (default FirstOrderSAE)
#   U, A, P    — hyperparameters that compose run_tag
#   CATEGORY   — vidvrd category filter (or None / empty)
#   EXTRACT_FOL=0  — disable extract_fol.py post-step
#   VISUALIZE=0    — disable visualize_fol.py post-step
#
# Monitor:  squeue -u $USER  /  tail -f logs/<job>.<JOBID>.out  /  seff <JOBID>

# ── Default SLURM directives (sh/submit.sh overrides these via sbatch CLI) ───
#SBATCH --job-name=fosae-train
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
# B16: SLURM sends SIGUSR1 to the batch script 180s before walltime; we re-raise
# it as SignalInterrupt in latplan/util/tuning.py so nn_task saves a partial
# model before the process is killed by SIGKILL.
#SBATCH --signal=B:USR1@180
##SBATCH --constraint="GPU_MEM:16GB"
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=yourname@stanford.edu

set -eo pipefail  # no -u: venv activate references unset vars

PROJECT_DIR="${SLURM_SUBMIT_DIR:-.}"
if [[ ! -f "${PROJECT_DIR}/sh/sherlock_env.sh" ]]; then
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# ── Default training command (overridable via TRAIN_CMD env var) ─────────────
DEFAULT_TRAIN_CMD="python3 strips.py learn vidvrd FirstOrderSAE 40 2 20 None None sequential 5000 None None 3"
TRAIN_CMD="${TRAIN_CMD:-${DEFAULT_TRAIN_CMD}}"

# ── Header ────────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Job ID    : ${SLURM_JOB_ID:-N/A}"
echo "  Job Name  : ${SLURM_JOB_NAME:-N/A}"
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
try:
    from tensorflow.python.client import device_lib
    devs = [d.name for d in device_lib.list_local_devices()]
    print("TF visible devices:", devs)
    if not any("GPU" in d for d in devs):
        print("WARNING: No GPU detected — job will be very slow on CPU")
except Exception as e:
    print(f"WARNING: GPU check failed: {e}")
PYEOF

# ── Prepare output / logs directories ────────────────────────────────────────
mkdir -p "${PROJECT_DIR}/out" "${PROJECT_DIR}/logs"

# ── Run training ──────────────────────────────────────────────────────────────
echo ""
echo "--- Training start: $(date) ---"
cd "${PROJECT_DIR}"
TRAIN_RC=0
eval "${TRAIN_CMD}" || TRAIN_RC=$?
echo ""
echo "--- Training end: $(date) ---  (exit ${TRAIN_RC})"

if [[ "${TRAIN_RC}" -ne 0 ]]; then
    echo "[error] Training command failed. Skipping post-train hooks."
    exit "${TRAIN_RC}"
fi

# ── Resolve OUT_DIR ─────────────────────────────────────────────────────────
# Single source of truth: JOB_OUT_DIR composed by sh/submit.sh via
# latplan/util/paths.py::resolved_out_dir. strips.py wrote there during the
# train step (SPEC §C15, §V13). No flat-path reconstruction.
DOMAIN="${DOMAIN:-}"
CATEGORY="${CATEGORY:-}"
OUT_DIR="${JOB_OUT_DIR:-${OUT_DIR:-}}"

# ── Post-train: extract_fol + visualize_fol if model was saved ───────────────
echo ""
echo "--- Post-train hooks ---"
if [[ -z "${OUT_DIR}" ]]; then
    echo "[skip] JOB_OUT_DIR / OUT_DIR not set (sh/submit.sh did not compose). Manual:"
    echo "       python3 extract_fol.py    <out_dir> --domain <domain>"
    echo "       python3 visualize_fol.py  <out_dir> --domain <domain>"
elif [[ ! -f "${OUT_DIR}/net0.h5" ]]; then
    echo "[skip] ${OUT_DIR}/net0.h5 not found — training likely did not complete cleanly."
else
    echo "[info] Output dir: ${OUT_DIR}"
    CAT_FLAG=()
    # When NPZ_PATH is set, the actual trained category lives in
    # ${OUT_DIR}/loaded_videos.json — the submit.sh-supplied CATEGORY env
    # is a default-bicycle leftover that would mismatch. Let extract_fol /
    # visualize_fol auto-detect from the manifest by omitting --category.
    if [[ -n "${CATEGORY}" && "${CATEGORY}" != "None" \
          && "${DOMAIN}" =~ ^(vidvrd|actiongenome)$ \
          && ( -z "${NPZ_PATH}" || "${NPZ_PATH}" == "None" ) ]]; then
        CAT_FLAG=(--category "${CATEGORY}")
    fi

    if [[ "${EXTRACT_FOL:-1}" == "1" ]]; then
        echo "[run]  extract_fol.py"
        python3 extract_fol.py "${OUT_DIR}" --domain "${DOMAIN}" "${CAT_FLAG[@]}" \
            || echo "[warn] extract_fol failed (non-fatal)"
    else
        echo "[skip] EXTRACT_FOL=0"
    fi

    if [[ "${VISUALIZE:-1}" == "1" ]]; then
        echo "[run]  visualize_fol.py --num ${VIS_NUM:-6}"
        python3 visualize_fol.py "${OUT_DIR}" --domain "${DOMAIN}" \
            --num "${VIS_NUM:-6}" "${CAT_FLAG[@]}" \
            || echo "[warn] visualize_fol failed (non-fatal)"
    else
        echo "[skip] VISUALIZE=0"
    fi
fi

# ── Summarise output ──────────────────────────────────────────────────────────
echo ""
echo "--- Output files ---"
if [[ -n "${OUT_DIR}" && -d "${OUT_DIR}" ]]; then
    ls -lh "${OUT_DIR}/" | head -30
else
    ls -lhR "${PROJECT_DIR}/out" 2>/dev/null | head -40
fi
echo "================================================================"
