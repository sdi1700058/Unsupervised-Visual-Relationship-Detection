#!/usr/bin/env bash
#
# list_ssv2.sbatch — count Something-Something V2's videos WITHOUT unpacking.
#
#   sbatch --job-name=ssv2-list sh/list_ssv2.sbatch
#
# WHY. Something-Else ranks first on notes/THE-TABLE.md, and its video is SSv2.
# The lock board reads it ORACLE ARM, which is TRUE in the board's own terms:
# nobody has opened SSv2, and axis 2 means measurably on disk. The 20 parts
# `20bn-something-something-v2-00` to `-19` carry no suffix, and their magic
# bytes are `1f 8b 08 00`, gzip, so concatenated they form one tar.gz.
#
# WHY NOT UNPACK. SSv2 holds about 220,000 videos, and the cluster rule is to
# prefer fewer, larger files over many small ones. Listing the stream answers
# the axis-2 question, that the videos are there and readable, without writing
# a single one.
#
# WHAT IT PROVES AND WHAT IT DOES NOT. A clean listing proves the archive is
# complete and every member header parses. It does not prove each video
# decodes, which only reading the frames would.
#
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
RAW="${ROOT}/data/video/ssv2/raw"

cat "${RAW}"/20bn-something-something-v2-?? | tar -tzvf - > "${TMPDIR:-/tmp}/ssv2.list"
status=( "${PIPESTATUS[@]}" )
echo "cat exit ${status[0]}, tar exit ${status[1]}"
echo "members: $(wc -l < "${TMPDIR:-/tmp}/ssv2.list")"
echo "by extension:"
awk '{print $NF}' "${TMPDIR:-/tmp}/ssv2.list" | sed -n 's/.*\.\([A-Za-z0-9]*\)$/\1/p' | sort | uniq -c | sort -rn | head
echo "total member bytes: $(awk '{s+=$3} END {print s}' "${TMPDIR:-/tmp}/ssv2.list")"
echo "smallest five members:"
sort -k3,3n "${TMPDIR:-/tmp}/ssv2.list" | awk '$3>0' | head -5 | awk '{print $3, $NF}'
