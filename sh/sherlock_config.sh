#!/usr/bin/env bash
# sherlock_config.sh — Single source of truth for Sherlock module + path config.
# SOURCE this file from sherlock_setup.sh / sherlock_env.sh / run_training.sh.
#
# Override any value before sourcing, e.g.
#   PYTHON_MODULE=python/3.12.1 CUDA_MODULE=cuda/12.4.0 source sh/sherlock_config.sh
#
# To check current Sherlock module strings:
#   ml av python   ml av cuda   ml av cudnn   ml av gcc

# ── Project root (file lives in sh/, so go one level up) ────────────────────
_CFG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${_CFG_DIR}/.." && pwd)}"

# ── Module versions (env-overridable) ────────────────────────────────────────
# Defaults reflect older toolchain compatible with TF 1.15 + Python 3.6.
# Newer Sherlock images may have replaced these — re-check with `ml av`.
PYTHON_MODULE="${PYTHON_MODULE:-python/3.6.1}"
CUDA_MODULE="${CUDA_MODULE:-cuda/10.0.130}"
CUDNN_MODULE="${CUDNN_MODULE:-cudnn/7.6.5}"
GCC_MODULE="${GCC_MODULE:-gcc/8.1.0}"

# ── Project paths (all inside PROJECT_ROOT — never write outside) ────────────
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/venv}"
KERAS_HOME="${KERAS_HOME:-${PROJECT_ROOT}/.keras}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/out}"

export PROJECT_ROOT VENV_DIR KERAS_HOME LOG_DIR OUT_DIR
export PYTHON_MODULE CUDA_MODULE CUDNN_MODULE GCC_MODULE
