#!/usr/bin/env bash
# sherlock_setup.sh — ONE-TIME venv creation on a Sherlock GPU compute node.
#
# Run from an interactive GPU session, NOT from a login node:
#
#   salloc -p gpu --gpus 1 --time 2:00:00 --mem 16G
#   cd /path/to/labeled-fosae
#   bash sh/sherlock_setup.sh
#
# To override the default module versions before running:
#   PYTHON_MODULE=python/3.9.0 CUDA_MODULE=cuda/11.7.1 \
#     bash sh/sherlock_setup.sh
#
# Everything is kept inside the project directory — nothing is written
# to $HOME, $GROUP_HOME, or anywhere else outside the project.

set -eo pipefail

# ── Load shared config (module versions + paths) ─────────────────────────────
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sherlock_config.sh"

# ── Sanity check (login node warning) ────────────────────────────────────────
if [[ "${SLURM_JOB_PARTITION:-}" != *"gpu"* && -z "${SLURMD_NODENAME:-}" ]]; then
    echo "WARNING: This does not appear to be a GPU compute node."
    echo "         Run 'salloc -p gpu --gpus 1' first, then re-run this script."
    echo "         Continuing anyway..."
fi

# ── Load modules ──────────────────────────────────────────────────────────────
echo "[1/6] Loading Sherlock modules..."
module purge
module load "${GCC_MODULE}"
module load "${PYTHON_MODULE}"
module load "${CUDA_MODULE}"
module load "${CUDNN_MODULE}"
module load ffmpeg

echo "      Python: $(python3 --version)"
echo "      nvcc:   $(nvcc --version 2>/dev/null | tail -1 || echo 'NOT FOUND')"

# ── Create virtual environment inside the project ─────────────────────────────
echo "[2/6] Creating venv at ${VENV_DIR} ..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# ── Upgrade pip / setuptools ──────────────────────────────────────────────────
echo "[3/6] Upgrading pip, wheel, setuptools..."
pip install --upgrade "pip<23" "setuptools<67" wheel
# pip<23 keeps the legacy resolver needed by old Keras/TF pins.

# ── Install project dependencies ──────────────────────────────────────────────
echo "[4/6] Installing requirements..."
pip install -r "${PROJECT_ROOT}/requirements.txt"

# ── Install the latplan package (editable) ────────────────────────────────────
echo "[5/6] Installing latplan in editable mode..."
pip install -e "${PROJECT_ROOT}" --no-deps

# ── Write Keras config inside the project ─────────────────────────────────────
echo "[6/6] Writing Keras config (${KERAS_HOME}/keras.json)..."
mkdir -p "${KERAS_HOME}"
cat > "${KERAS_HOME}/keras.json" <<'EOF'
{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
EOF

deactivate

echo ""
echo "Done. Venv at: ${VENV_DIR}"
echo "Keras config at: ${KERAS_HOME}/keras.json"
echo ""
echo "To activate the environment:"
echo "   source ${PROJECT_ROOT}/sh/sherlock_env.sh"
