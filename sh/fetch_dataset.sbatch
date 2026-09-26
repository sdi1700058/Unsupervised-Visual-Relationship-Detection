#!/usr/bin/env bash
#
# fetch_dataset.sbatch — pull one dataset's archives on a compute node.
#
#   sbatch --job-name=dl-vidor sh/fetch_dataset.sbatch vidor
#   sbatch --job-name=dl-vidvrd sh/fetch_dataset.sbatch vidvrd
#
# WHY A JOB AND NOT A LOGIN-NODE LOOP. Two reasons, and both are measured.
# The cluster policy in /etc/claude-code/CLAUDE.md forbids heavy work on the
# login node, and 29 GB over eight archives is heavy. And `curl` is DENIED to
# the session by managed settings, while a compute node runs an ordinary
# shell, so the job can fetch what the session cannot.
#
# EVERY DATASET HERE IS ON A HOST THAT NEEDS NO NEW PERMISSION. huggingface.co
# already serves VidOR and VidVRD in full, video included. That was measured on
# 2026-09-21 from the HuggingFace API rather than assumed from a project page.
#
# RESUMABLE BY DESIGN. `curl -C -` continues a part-file, so a job that hits
# the wall clock is re-submitted rather than restarted. Nothing is unpacked
# here: unpacking is a separate step, because a half-unpacked archive is
# harder to detect than a half-downloaded one.
#
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --output=logs/fetch-%x-%j.out
#SBATCH --error=logs/fetch-%x-%j.err

set -uo pipefail

DS="${1:?usage: sbatch sh/fetch_dataset.sbatch <dataset>}"
# SLURM COPIES THE SCRIPT TO ITS SPOOL DIRECTORY, so BASH_SOURCE points at
# /var/spool/slurmd and NOT at the repository. A probe job on 2026-09-21
# failed with "no manifest at /var/spool/slurmd/eval/datasets/routes/..."
# for exactly that reason, which is why a 20-minute probe runs before a
# 12-hour job. SLURM_SUBMIT_DIR is where sbatch was invoked, and the fallback
# keeps the script runnable as a plain bash script too.
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
[[ -d "${ROOT}/eval/datasets/routes" ]] || {
    echo "fetch: ${ROOT} is not the repository root (no eval/datasets/routes)." >&2
    echo "       Submit from the repository root, or set THESIS_ROOT." >&2
    exit 2
}
DEST="${ROOT}/data/video/${DS}/raw"
MANIFEST="${ROOT}/eval/datasets/routes/${DS}.manifest"

say  () { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn () { printf '[%s] WARNING: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

[[ -r "${MANIFEST}" ]] || { warn "no manifest at ${MANIFEST}"; exit 2; }
mkdir -p "${DEST}" "${ROOT}/logs"

say "dataset ${DS}"
say "into    ${DEST}"
say "manifest $(grep -cvE '^\s*(#|$)' "${MANIFEST}") file(s)"

ok=0; failed=0; skipped=0
while read -r url; do
    [[ -z "${url}" || "${url}" =~ ^[[:space:]]*# ]] && continue
    # A LINE MAY CARRY AN EXPLICIT OUTPUT NAME: "<url> <name>". Without one the
    # name is the URL's basename, and on 2026-09-26 that silently lost three of
    # four files: every Google Drive URL is ".../download?id=...", so all four
    # Something-Else parts resolved to "download" and the last three were
    # skipped as "already present".
    read -r url name <<< "${url}"
    [[ -n "${name}" ]] || name="$(basename "${url%%\?*}")"
    out="${DEST}/${name}"

    # A complete file is left alone. A part-file is continued.
    if [[ -s "${out}" && ! -s "${out}.part" ]]; then
        say "have ${name} ($(du -h "${out}" | cut -f1))"
        skipped=$((skipped + 1)); continue
    fi

    say "fetching ${name}"
    if curl -fL --retry 5 --retry-delay 10 -C - "${url}" -o "${out}"; then
        say "got ${name} ($(du -h "${out}" | cut -f1))"
        ok=$((ok + 1))
    else
        warn "failed ${name}"
        failed=$((failed + 1))
    fi
done < "${MANIFEST}"

say "done: ${ok} fetched, ${skipped} already present, ${failed} failed"
say "total on disk: $(du -sh "${DEST}" 2>/dev/null | cut -f1)"

# THE COUNT IS THE RESULT, so it is written where the lock board can read it.
{
    echo "dataset=${DS}"
    echo "measured_on=$(date +%Y-%m-%d)"
    echo "fetched=${ok} skipped=${skipped} failed=${failed}"
    echo "bytes=$(du -sb "${DEST}" 2>/dev/null | cut -f1)"
    echo "files=$(find "${DEST}" -maxdepth 1 -type f | wc -l)"
} > "${ROOT}/eval/datasets/routes/${DS}.fetched"

[[ "${failed}" -eq 0 ]] || exit 1
