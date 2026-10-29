#!/usr/bin/env bash
# planner/install_fd.sh — bootstrap Fast Downward for SPEC §T H2.
#
# FD is required by planner/plan_video.py (§T H1). Upstream
# /home/panoslat/Dev/Thesis/FOSAE/latplan/downward/ is empty (git submodule
# not initialized) and is C14 read-only, so we build our own copy out of git.
#
# Builds under data/deps/fast-downward/ (gitignored via data/*), where
# planner/pddl/planner.py:find_fast_downward looks for it. On Sherlock, run it
# inside a job after `ml gcc/12.4.0 cmake/3.24.2` (the system gcc is too old).
#
# Requires: cmake, gcc/g++ (≥7), python3, git.
#
# Usage
# -----
#   bash planner/install_fd.sh              # builds w/ default release profile
#   FORCE=1 bash planner/install_fd.sh      # rebuild even if binary present
#
# Gate: SPEC §P H2 — after run, expect `fast-downward.py --help | head -1`
# non-empty.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FD_URL="https://github.com/aibasel/downward.git"
FD_TAG="${FD_TAG:-release-24.06.0}"   # tagged release, pins reproducibility per V15

DEPS_DIR="${PROJECT_DIR}/data/deps"
FD_DIR="${DEPS_DIR}/fast-downward"
FD_BIN="${FD_DIR}/fast-downward.py"

# The driver script comes with the clone, so test the compiled search binary.
if [[ -x "${FD_DIR}/builds/release/bin/downward" && "${FORCE:-0}" != "1" ]]; then
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
# The C++ runtime is linked in, so the binary runs without the gcc module
# (the system libstdc++ of Sherlock is too old for it).
cmake -S src -B builds/release -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc"
cmake --build builds/release -j "${JOBS:-${SLURM_CPUS_ON_NODE:-2}}"


echo "[install_fd] OK"
"${FD_BIN}" --help | head -1
