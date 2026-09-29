#!/usr/bin/env bash
#
# unpack_charades.sbatch — unpack Charades 480p, verify it, read every frame rate
#
#   sbatch --job-name=charades-unpack sh/unpack_charades.sbatch
#
# WHY. Action Genome, row 15 of notes/THE-TABLE.md, labels frames of these
# videos. Its own `dump_frames.py` runs plain `ffmpeg -i video %06d.png`, so a
# frame number counts decoded frames from 1 at each video's own rate, which
# the repository says is not always 24. The frame-rate table this job writes
# is what turns row 15's spacing of 12 frames into seconds.
#
# THREE STEPS. Unpack with Python's zipfile. Verify with
# workbench/tools/verify_archive.py: every member against its CRC, and the
# unpacked copy name for name and size for size. Then ffprobe every video.
#
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SRC="${ROOT}/data/video/charades/raw/Charades_v1_480.zip"
DEST="${ROOT}/data/video/charades/ext"
PY="${ROOT}/.venv-local/bin/python"
mkdir -p "${DEST}"

"${PY}" - "${SRC}" "${DEST}" <<'PY'
import sys, zipfile
src, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(src) as z:
    names = z.namelist()
    print("members: %d" % len(names), flush=True)
    for i, n in enumerate(names, 1):
        z.extract(n, dest)
        if i % 2000 == 0:
            print("  %d/%d" % (i, len(names)), flush=True)
print("extracted %d" % len(names))
PY

cd "${ROOT}/workbench" || exit 2
PYTHONPATH="..:${PYTHONPATH:-}" "${PY}" -m tools.verify_archive "${SRC}" --root "${DEST}"
verdict=$?

module load system
module load ffmpeg/7.1
OUT="${ROOT}/data/video/charades/frame_rates.tsv"
printf 'video\tr_frame_rate\tavg_frame_rate\tnb_frames\tduration\n' > "${OUT}"
find "${DEST}" -name '*.mp4' | sort | while read -r v; do
    id="$(basename "${v}" .mp4)"
    # key=value output, because ffprobe's csv writer uses its own field
    # order and not the order asked for.
    meta="$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=r_frame_rate,avg_frame_rate,nb_frames,duration \
        -of default=noprint_wrappers=1 "${v}")"
    get () { printf '%s\n' "${meta}" | sed -n "s/^$1=//p" | head -1; }
    printf '%s\t%s\t%s\t%s\t%s\n' "${id}" "$(get r_frame_rate)" \
        "$(get avg_frame_rate)" "$(get nb_frames)" "$(get duration)" >> "${OUT}"
done
echo "frame rates: $(($(wc -l < "${OUT}") - 1)) videos in ${OUT}"
exit "${verdict}"
