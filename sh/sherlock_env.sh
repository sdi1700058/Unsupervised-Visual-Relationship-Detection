#!/usr/bin/env bash
# sherlock_env.sh — Environment bootstrap for labeled-fosae on Stanford Sherlock.
#
# SOURCE this file; do not execute it directly:
#   source /path/to/labeled-fosae/sh/sherlock_env.sh
#
# To auto-load on every login, append to ~/.bashrc:
#   echo 'source /path/to/labeled-fosae/sh/sherlock_env.sh' >> ~/.bashrc
#
# Override module versions by exporting before sourcing, e.g.
#   PYTHON_MODULE=python/3.9.0 source sh/sherlock_env.sh
#
# Everything lives inside the project directory.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: source this script, do not execute it:  source sh/sherlock_env.sh" >&2
    exit 1
fi

# ── Load shared config (module versions + paths) ─────────────────────────────
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sherlock_config.sh"

export KERAS_HOME

# ── Optional tools (set to 1 to load) ────────────────────────────────────────
LOAD_CLAUDE_CODE="${LOAD_CLAUDE_CODE:-0}"
LOAD_COPILOT_CLI="${LOAD_COPILOT_CLI:-0}"

# ── Load modules ──────────────────────────────────────────────────────────────
# cudnn/7.6.5 declares cuda/10.2.89 as its dependency, so loading it pulls in the
# wrong CUDA for TF 1.15 (which needs libcudart.so.10.0 = cuda/10.0.x). The fix:
# load cudnn FIRST, then load CUDA_MODULE LAST so it swaps the dep cuda out for
# cuda/10.0.130 — the two then coexist (verified: TF opens libcudart.so.10.0 AND
# libcudnn.so.7). The saved `fosae` collection captures exactly this working set,
# so prefer `module restore fosae`; fall back to the explicit sequence.
if command -v module &>/dev/null; then
    if module restore fosae 2>/dev/null; then
        echo "[sherlock_env] module collection 'fosae' restored"
    else
        echo "[sherlock_env] collection 'fosae' not found — loading modules explicitly"
        module purge
        module load "${GCC_MODULE}" "${PYTHON_MODULE}" "${CUDNN_MODULE}" 2>/dev/null || \
            echo "[sherlock_env] WARNING: base module load failed"
        # CUDA last: swaps the cuda that cudnn pulled in for ${CUDA_MODULE}.
        module load "${CUDA_MODULE}" 2>/dev/null || \
            echo "[sherlock_env] WARNING: could not load ${CUDA_MODULE}"
        module load ffmpeg 2>/dev/null || true
        module save fosae 2>/dev/null && \
            echo "[sherlock_env] saved module collection 'fosae'"
    fi

    if [[ "${LOAD_CLAUDE_CODE}" == "1" ]]; then
        module load claude-code 2>/dev/null || \
            echo "[sherlock_env] WARNING: claude-code module not found"
    fi

    if [[ "${LOAD_COPILOT_CLI}" == "1" ]]; then
        module load copilot-cli 2>/dev/null || \
            echo "[sherlock_env] WARNING: copilot-cli module not found"
    fi
fi

# Module vars are consumed above; unset them so a re-source in the SAME shell
# re-derives the defaults from sherlock_config.sh instead of reusing the stale
# value left in the shell scope by the previous source.
unset GCC_MODULE PYTHON_MODULE CUDA_MODULE CUDNN_MODULE

# ── Activate the venv ─────────────────────────────────────────────────────────
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    source "${VENV_DIR}/bin/activate"
    echo "[sherlock_env] Activated venv: ${VENV_DIR}"
else
    echo "[sherlock_env] WARNING: venv not found at ${VENV_DIR}"
    echo "               Run: bash ${PROJECT_ROOT}/sh/sherlock_setup.sh"
fi

# ── Python settings ───────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

# ── GPU sanity check (honest — prints TF's real device list) ─────────────────
# A boolean "OK" hides the truth and gave false confidence. Print exactly what
# TF sees, in THIS shell, and do NOT swallow the libcudart/libcudnn dlopen
# errors — those W-lines are the actual diagnostic. TF 1.15 needs CUDA 10.0
# (libcudart.so.10.0) + cuDNN 7.x (libcudnn.so.7) — see sh/sherlock_config.sh.
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
    _gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "[sherlock_env] GPU on node (driver): ${_gpu_name}"
    TF_CPP_MIN_LOG_LEVEL=1 python3 - <<'PYEOF'
from tensorflow.python.client import device_lib
gpus = [d.name for d in device_lib.list_local_devices() if d.device_type == "GPU"]
if gpus:
    print("[sherlock_env] TF sees GPU:", gpus)
else:
    print("[sherlock_env] WARNING: TF sees NO GPU -> training WILL run on CPU.")
    print("               Look above for 'Could not load libcudart.so.10.0' etc.")
    print("               TF 1.15 needs CUDA 10.0 + cuDNN 7.x; check 'ml list'.")
PYEOF
else
    echo "[sherlock_env] (no GPU on this node — login or CPU partition)"
fi

# ── Convenience aliases ───────────────────────────────────────────────────────
alias py='python3'
alias gpu-shell='salloc -p gpu --gpus 1 --mem 32G --time 4:00:00'
alias cpu-shell='sh_dev'
