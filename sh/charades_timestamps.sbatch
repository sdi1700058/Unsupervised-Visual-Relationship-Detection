#!/usr/bin/env bash
#
# charades_timestamps.sbatch — every frame's real time, and a full decode count
#
#   sbatch --job-name=charades-ts sh/charades_timestamps.sbatch
#
# WHY. Two adversaries found on 2026-09-29 that Action Genome's gaps in
# seconds rested on one frame rate per video, which assumes the rate never
# varies, and that "no annotated frame past a video's end" compared against
# the container's frame COUNT, which is metadata and not a decode. This job
# reads every video packet's presentation time, and decodes every video in
# full to count its frames and its decode errors.
#
# OUTPUT. data/video/charades/timestamps.tsv, one line per video:
# video, decoded frames, decode error lines, frame times in seconds.
# Per-video lines go to node-local $L_SCRATCH first and are joined at the end,
# so eight parallel writers never interleave.
#
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SRC="${ROOT}/data/video/charades/ext"
OUT="${ROOT}/data/video/charades/timestamps.tsv"
WORK="${L_SCRATCH:?no node-local scratch}/charades-ts"
mkdir -p "${WORK}"
module load system
module load ffmpeg/7.1

one () {
    v="$1"; work="$2"
    id="$(basename "${v}" .mp4)"
    pts="$(ffprobe -v error -select_streams v:0 -show_entries packet=pts_time \
        -of csv=p=0 "${v}" | tr '\n' ' ')"
    errs="${work}/${id}.err"
    decoded="$(ffprobe -v error -count_frames -select_streams v:0 \
        -show_entries stream=nb_read_frames -of csv=p=0 "${v}" 2>"${errs}")"
    printf '%s\t%s\t%s\t%s\n' "${id}" "${decoded:-0}" \
        "$(wc -l < "${errs}")" "${pts}" > "${work}/${id}.tsv"
}
export -f one

find "${SRC}" -name '*.mp4' | sort | xargs -P "${SLURM_CPUS_PER_TASK:-8}" -I{} \
    bash -c 'one "$1" "$2"' _ {} "${WORK}"

printf 'video\tdecoded\terrors\tpts\n' > "${OUT}"
cat "${WORK}"/*.tsv >> "${OUT}"
echo "videos timed: $(($(wc -l < "${OUT}") - 1))"
echo "videos with decode errors: $(awk -F'\t' 'NR > 1 && $3 > 0' "${OUT}" | wc -l)"
echo "videos decoded to 0 frames: $(awk -F'\t' 'NR > 1 && $2 == 0' "${OUT}" | wc -l)"
