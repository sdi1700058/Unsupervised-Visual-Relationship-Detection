#!/usr/bin/env bash
# activate.sh — top-level convenience for interactive Sherlock sessions.
#
# ALWAYS source this instead of `venv/bin/activate` directly, otherwise
# TF 1.15 will silently fall back to CPU (CUDA 10 / cuDNN 7 modules unloaded).
#
# Usage:
#   source ./activate.sh
#
# Equivalent to:  source sh/sherlock_env.sh

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_HERE}/sh/sherlock_env.sh"
unset _HERE
