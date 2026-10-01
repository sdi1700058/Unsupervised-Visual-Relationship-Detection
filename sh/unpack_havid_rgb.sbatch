#!/usr/bin/env bash
#
# unpack_havid_rgb.sbatch — unpack HA-ViD's 3,222 videos and read every frame rate
#
#   sbatch --job-name=havid-rgb sh/unpack_havid_rgb.sbatch
#
# WHY. Axis 2 of THE-TABLE asks whether the frames are measurably on disk.
# HAViD_rgb.zip passed its CRC on 2026-09-30, Slurm 46146634. This unpacks it,
# checks the copy against the archive, and writes each video's frame rate and
# decoded frame count, because the action timestamps are frame numbers and
# one sample video read 15 fps.
#
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SRC="${ROOT}/data/video/ha_vid/raw/HAViD_rgb.zip"
DEST="${ROOT}/data/video/ha_vid/ext/rgb"
mkdir -p "${DEST}"
module load python/3.9.0
python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2]); print('extracted')" "${SRC}" "${DEST}"
cd "${ROOT}/workbench" || exit 2
PYTHONPATH="..:${PYTHONPATH:-}" python3 -m tools.verify_archive "${SRC}" --root "${DEST}"
verdict=$?
module load system
module load ffmpeg/7.1
OUT="${ROOT}/data/video/ha_vid/frame_rates.tsv"
one () {
    v="$1"
    printf '%s\t%s\t%s\n' "$(basename "${v}" .mp4)" \
        "$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=nw=1:nk=1 "${v}")" \
        "$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "${v}")"
}
export -f one
printf 'video\tr_frame_rate\tdecoded\n' > "${OUT}"
find "${DEST}" -name '*.mp4' | sort | xargs -P "${SLURM_CPUS_PER_TASK:-8}" -I{} bash -c 'one "$1"' _ {} | sort >> "${OUT}"
echo "videos: $(($(wc -l < "${OUT}") - 1))"
awk -F'\t' 'NR > 1 {print $2}' "${OUT}" | sort | uniq -c
echo "decoded to 0 frames: $(awk -F'\t' 'NR > 1 && ($3 == "" || $3 == 0)' "${OUT}" | wc -l)"
exit "${verdict}"
