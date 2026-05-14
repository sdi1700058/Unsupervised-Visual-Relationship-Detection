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

# ── Load shared config (module versions + paths) ─────────────────────────────
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sherlock_config.sh"

export KERAS_HOME

# ── Optional tools (set to 1 to load) ────────────────────────────────────────
LOAD_CLAUDE_CODE="${LOAD_CLAUDE_CODE:-0}"
LOAD_COPILOT_CLI="${LOAD_COPILOT_CLI:-0}"

# ── Load modules ──────────────────────────────────────────────────────────────
if command -v module &>/dev/null; then
    module purge

    for _m in "${GCC_MODULE}" "${PYTHON_MODULE}" "${CUDA_MODULE}" "${CUDNN_MODULE}"; do
        module load "${_m}" 2>/dev/null || \
            echo "[sherlock_env] WARNING: could not load ${_m}"
    done

    if [[ "${LOAD_CLAUDE_CODE}" == "1" ]]; then
        module load claude-code 2>/dev/null || \
            echo "[sherlock_env] WARNING: claude-code module not found"
    fi

    if [[ "${LOAD_COPILOT_CLI}" == "1" ]]; then
        module load copilot-cli 2>/dev/null || \
            echo "[sherlock_env] WARNING: copilot-cli module not found"
    fi
fi

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

# ── GPU sanity check (loud) ──────────────────────────────────────────────────
# Loudly warn if TF cannot see the GPU — otherwise users silently fall back to
# CPU and wonder why training crawls. CUDA 10 + cuDNN 7 are required by TF 1.15.
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
    _gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "[sherlock_env] GPU detected: ${_gpu_name}"
    _tf_gpu="$(python3 -c 'import os; os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"; from tensorflow.python.client import device_lib; print(any("GPU" in d.name for d in device_lib.list_local_devices()))' 2>/dev/null)"
    if [[ "${_tf_gpu}" == "True" ]]; then
        echo "[sherlock_env] TF GPU support: OK"
    else
        echo "[sherlock_env] WARNING: TF cannot use the GPU (CUDA/cuDNN mismatch)."
        echo "               TF 1.15 requires CUDA 10.0–10.2 + cuDNN 7.x."
        echo "               Loaded modules: CUDA=${CUDA_MODULE} cuDNN=${CUDNN_MODULE}"
        echo "               Check 'ml list' / 'ml av cuda' to confirm versions."
    fi
else
    echo "[sherlock_env] (no GPU on this node — login or CPU partition)"
fi

# ── Convenience aliases ───────────────────────────────────────────────────────
alias py='python3'
alias gpu-shell='salloc -p gpu --gpus 1 --mem 32G --time 4:00:00'
alias cpu-shell='sh_dev'
