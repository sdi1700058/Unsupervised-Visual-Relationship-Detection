#!/usr/bin/env bash
# tools/planner/install_fd.sh — bootstrap Fast Downward for SPEC §T H2.
#
# FD is required by tools/planner/plan_video.py (§T H1). Upstream
# /home/panoslat/Dev/Thesis/FOSAE/latplan/downward/ is empty (git submodule
# not initialized) and is C14 read-only, so we build our own copy out of git.
#
# Local  : builds under data/deps/fast-downward/ (gitignored via data/*)
# Sherlock: builds under $SCRATCH/panos/sgg-thesis/deps/fast-downward/
#
# Requires: cmake, gcc/g++ (≥7), python3, git.
#
# Usage
# -----
#   bash tools/planner/install_fd.sh              # builds w/ default release profile
#   FORCE=1 bash tools/planner/install_fd.sh      # rebuild even if binary present
#
# Gate: SPEC §P H2 — after run, expect `fast-downward.py --help | head -1`
# non-empty.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FD_URL="https://github.com/aibasel/downward.git"
FD_TAG="${FD_TAG:-release-24.06.0}"   # tagged release, pins reproducibility per V15

if [[ -n "${SCRATCH:-}" && -d "${SCRATCH}/panos/sgg-thesis" ]]; then
    DEPS_DIR="${SCRATCH}/panos/sgg-thesis/deps"
else
    DEPS_DIR="${PROJECT_DIR}/data/deps"
fi
FD_DIR="${DEPS_DIR}/fast-downward"
FD_BIN="${FD_DIR}/fast-downward.py"

if [[ -x "${FD_BIN}" && "${FORCE:-0}" != "1" ]]; then
    echo "[install_fd] present: ${FD_BIN}"
    "${FD_BIN}" --help | head -1
    exit 0
fi

mkdir -p "${DEPS_DIR}"
if [[ ! -d "${FD_DIR}" ]]; then
    echo "[install_fd] cloning ${FD_URL}@${FD_TAG} → ${FD_DIR}"
    git clone --depth 1 --branch "${FD_TAG}" "${FD_URL}" "${FD_DIR}"
fi

cd "${FD_DIR}"
echo "[install_fd] building (release)"
./build.py release

echo "[install_fd] symlinking → tools/planner/fast-downward.py"
ln -sf "${FD_BIN}" "${PROJECT_DIR}/tools/planner/fast-downward.py"

echo "[install_fd] OK"
"${FD_BIN}" --help | head -1
