#!/usr/bin/env bash
# sh/setup_venv_detect.sh — one-time sidecar venv for bbox preprocessing.
#
# The main FOSAE TF venv is py3.6 (locked by tf 1.15). Modern detection
# stacks (MediaPipe 0.10+, Grounded-DINO via transformers) need py>=3.9.
# This sidecar lives at PROJECT_DIR/venv-detect and is activated ONLY when
# running tools/synth_bbox.py — never touched at training time.
#
# Usage:
#   bash sh/setup_venv_detect.sh
#   source venv-detect/bin/activate
#   python tools/synth_bbox.py ...
#   deactivate
#
# Re-running is safe: pip --upgrade is no-op when already at latest.

set -eo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/venv-detect"

# Pin pip's wheel cache + HF model cache inside the workspace too.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_DIR}/.pip-cache}"
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf}"
mkdir -p "${PIP_CACHE_DIR}" "${HF_HOME}"

# Pick the newest python available. Order: explicit env > module > $PATH.
PY="${PYTHON:-}"
if [[ -z "${PY}" ]]; then
    for cand in python3.11 python3.10 python3.9 python3.8 python3; do
        if command -v "${cand}" &>/dev/null; then
            V=$("${cand}" -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
            MAJ=$(echo "${V}" | cut -d. -f1)
            MIN=$(echo "${V}" | cut -d. -f2)
            if (( MAJ >= 3 )) && (( MIN >= 8 )); then
                PY="${cand}"
                break
            fi
        fi
    done
fi
if [[ -z "${PY}" ]]; then
    echo "ERROR: no python>=3.8 found. On Sherlock: module load python/3.9.0" >&2
    exit 2
fi
echo "[venv] using ${PY} ($(${PY} --version))"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[venv] creating ${VENV_DIR}"
    "${PY}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[venv] upgrading pip"
pip install --quiet --upgrade pip wheel

echo "[venv] installing detection deps (CPU torch wheels)..."
# torch CPU wheel first (avoids the default CUDA build's 2 GB download)
pip install --quiet --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.0,<2.5" "torchvision>=0.15,<0.20" || \
    pip install --quiet "torch>=2.0,<2.5" "torchvision>=0.15,<0.20"

pip install --quiet \
    "mediapipe>=0.10" \
    "transformers>=4.40" \
    "pillow" \
    "numpy<2" \
    "pyarrow"

echo
echo "[venv] verifying imports..."
python -c "import mediapipe, transformers, torch, PIL, numpy, pyarrow; \
print(f'  mediapipe   {mediapipe.__version__}'); \
print(f'  transformers {transformers.__version__}'); \
print(f'  torch       {torch.__version__} (cuda={torch.cuda.is_available()})'); \
print(f'  PIL         {PIL.__version__}'); \
print(f'  numpy       {numpy.__version__}'); \
print(f'  pyarrow     {pyarrow.__version__}')"

deactivate
echo
echo "[ok] venv-detect ready. Activate with:  source venv-detect/bin/activate"
