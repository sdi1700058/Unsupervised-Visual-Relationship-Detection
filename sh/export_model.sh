#!/usr/bin/env bash
# Export a trained model to a planner export, on a compute node.
#
#   sbatch sh/export_model.sh <model_dir> [<model_dir> ...]
#
# This has to be a batch job. Running export_latents.py on a login node dies
# with:
#
#     cuInit: UNKNOWN ERROR (303)
#     terminate called after throwing an instance of 'std::system_error'
#       what():  Resource temporarily unavailable
#
# The cuInit line is a log, not the fault. The abort is TensorFlow failing to
# create its thread pool against the login node's process limit, and no
# environment variable avoids it — the work has to move off the login node.
# One CPU node is plenty; encoding a clip is seconds of compute.
#
# Writes eval/exports/<basename-of-model-dir>.npz unless the directory name is
# one of the long hashed ones, in which case pass -o yourself by editing the
# call below.
#
#SBATCH --job-name=fosae-export
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/export.%j.out
#SBATCH --error=logs/export.%j.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs eval/exports

source venv/bin/activate 2>/dev/null || source activate.sh

if [[ $# -eq 0 ]]; then
    echo "usage: sbatch sh/export_model.sh <model_dir> [<model_dir> ...]" >&2
    exit 2
fi

# No GPU is needed to encode a few hundred frames, and asking for one only
# lengthens the queue wait.
export FOSAE_GPU=0
export CUDA_VISIBLE_DEVICES=""

rc=0
for MODEL_DIR in "$@"; do
    if [[ ! -d "${MODEL_DIR}" ]]; then
        echo "SKIP ${MODEL_DIR}: not a directory"
        rc=1
        continue
    fi

    # The run directories are named FirstOrderSAE_U<n>_A<n>_P<n>_cat<npz>_...
    # Keep the U/A/P and the npz stem, drop the parameter hash, so the export
    # name says what produced it.
    NAME="$(basename "${MODEL_DIR}")"
    NAME="${NAME#FirstOrderSAE_}"
    OUT="eval/exports/${NAME}.npz"

    echo "=== ${MODEL_DIR}"
    if python3 tools/planner/export_latents.py "${MODEL_DIR}" -o "${OUT}"; then
        ls -lh "${OUT}"
    else
        echo "FAILED ${MODEL_DIR}"
        rc=1
    fi
done

echo
echo "done. commit what landed:"
echo "    git add eval/exports/*.npz && git commit -m 'export: planner inputs' && git push"
exit "${rc}"
