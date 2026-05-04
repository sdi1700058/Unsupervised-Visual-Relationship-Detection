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

# ── Convenience aliases ───────────────────────────────────────────────────────
alias py='python3'
alias gpu-shell='salloc -p gpu --gpus 1 --mem 32G --time 4:00:00'
alias cpu-shell='sh_dev'
