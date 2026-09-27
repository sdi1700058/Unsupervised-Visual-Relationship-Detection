#!/usr/bin/env bash
#
# probe_furniture_bench.sbatch — does a released FurnitureBench file record poses?
#
#   sbatch --job-name=fb-probe sh/probe_furniture_bench.sbatch
#
# WHY. An adversary found on 2026-09-27 that Open X-Embodiment's mirror of
# FurnitureBench dropped `parts_poses`, the per-frame pose of every furniture
# part, measured on a real robot with AprilTags. The dataset page does not
# list that field. The collector code writes it. Only a released file settles
# which describes the release.
#
# IT READS A FRACTION OF 7.4 GB. `lamp` at high randomness is the smallest
# release, 7,423,005,782 bytes over a direct Google Drive link with no form. The
# probe streams the tar.gz and stops at the end of the FIRST pickle.
#
# IT NEVER CALLS pickle.load. The pickle comes from the internet, so the probe
# uses a restricted unpickler that refuses any global outside a numpy allow
# list. A test pins that a pickle naming os.system is refused.
#
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${ROOT}" || exit 2
URL="https://drive.usercontent.google.com/download?id=1Ia0SHIACIoqwzVjhc_dgzEMsiKffIO6T&export=download&confirm=t"
PYTHONPATH="${ROOT}/workbench" "${ROOT}/.venv-local/bin/python" \
    "${ROOT}/workbench/tools/furniture_bench_probe.py" "${URL}"
