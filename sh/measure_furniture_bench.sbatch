#!/usr/bin/env bash
#
# measure_furniture_bench.sbatch — every demonstration of one FurnitureBench release
#
#   sbatch --job-name=fb-all sh/measure_furniture_bench.sbatch
#
# WHY. FurnitureBench's code declares, for each furniture kind, which part
# pairs a finished piece joins (`should_be_assembled`), and the dataset page
# says a reward is 1 at the step a part is assembled. Together those make each
# demonstration a sequence of assembled(i, j) propositions turning true. That
# was read from the code and the page on 2026-09-28 and not from a file. This
# job counts, for every released demonstration, its length and its reward
# steps, and checks the count against the pairs the code declares.
#
# IT STREAMS AND KEEPS NOTHING. The tar.gz is read front to back, each pickle
# is summarised and dropped, and nothing lands on disk but the log.
#
# IT NEVER CALLS pickle.load. The restricted unpickler refuses every global
# outside a numpy allow list, and a test pins that a pickle naming os.system
# is refused.
#
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${ROOT}" || exit 2
# lamp at high randomness, 7,423,005,782 bytes, the smallest release.
URL="${FB_URL:-https://drive.usercontent.google.com/download?id=1Ia0SHIACIoqwzVjhc_dgzEMsiKffIO6T&export=download&confirm=t}"
PYTHONPATH="${ROOT}/workbench" "${ROOT}/.venv-local/bin/python" \
    "${ROOT}/workbench/tools/furniture_bench_probe.py" --all "${URL}"
