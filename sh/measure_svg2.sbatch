#!/usr/bin/env bash
#
# measure_svg2.sbatch — SVG2's pairs, succession change and slot pressure.
#
#   sbatch --job-name=svg2 sh/measure_svg2.sbatch
#
# WHY. SVG2 is the largest relation set in the field, 636,000 videos and 6.7
# million relations written by gpt-5-2025-08-07. Criterion D2 closed it for the
# model authorship alone, and the author withdrew D2 on 2026-09-26, so it needs
# the three numbers every row carries. One of its 61 annotation files ran as a
# smoke test on 2026-09-27, and one file is not a result.
#
# WHAT IT FETCHES. Only the CLEANED annotation: 60 `pvd` parquet files and one
# `sav`, about 263 MB, from a HuggingFace dataset that is not gated and is
# CC BY 4.0. Not the 1.83 TB mask tree and not the videos, which live at Meta
# and are not in this repository.
#
# TWO TRAPS MET ON 2026-09-27, both handled below.
#   1. `workbench/tools/queue.py` shadows the standard library's `queue` when a
#      tool runs as a script, and pyarrow then fails to import. So the tool
#      runs as `-m tools.svg2_relations` from `workbench`.
#   2. The pyarrow module sets PYTHONPATH itself, so it is APPENDED to here.
#
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
RAW="${ROOT}/data/video/svg2/raw"
BASE="https://huggingface.co/datasets/UWGZQ/Synthetic_Visual_Genome2/resolve/main/data/cleaned"
mkdir -p "${RAW}"

fetch () {
    local url="$1" out="$2"
    [[ -s "${out}" ]] && return 0
    curl -fsSL --retry 5 --retry-delay 10 -o "${out}.part" "${url}" \
        && mv "${out}.part" "${out}"
}

failed=0
for i in $(seq -w 0 59); do
    fetch "${BASE}/pvd/train-000${i}-of-00060.parquet" \
          "${RAW}/pvd-train-000${i}.parquet" || failed=$((failed + 1))
done
fetch "${BASE}/sav/train-00000-of-00001.parquet" \
      "${RAW}/sav-train-00000.parquet" || failed=$((failed + 1))
echo "fetched: $(ls "${RAW}"/*.parquet | wc -l) parquet files, ${failed} failed, $(du -sh "${RAW}" | cut -f1)"

module load python/3.12.1 py-numpy/1.26.3_py312 py-pyarrow/18.1.0_py312
cd "${ROOT}/workbench" || exit 2
PYTHONPATH="${ROOT}:${PYTHONPATH:-}" python3 -m tools.svg2_relations "${RAW}"/*.parquet
