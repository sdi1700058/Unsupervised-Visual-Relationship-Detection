#!/usr/bin/env bash
# download_datasets.sh — fetch a candidate corpus, one at a time, on the cluster.
#
#   bash sh/download_datasets.sh list
#   bash sh/download_datasets.sh something_else
#   bash sh/download_datasets.sh vidor
#   bash sh/download_datasets.sh open_x
#   bash sh/download_datasets.sh language_table
#   bash sh/download_datasets.sh robovqa
#
# Every target is independent and resumable. Nothing here downloads video unless
# the target says so, because the oracle needs boxes and not frames: a corpus
# with annotations alone is already measurable.
#
# Sources and provenance are in notes/lit/dataset_sources.json, where each url
# records the date it was last fetched. Nothing is rejected for lacking boxes or
# a detector; that is a cost, recorded per target below.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA_DIR:-${ROOT}/data/video}"

say()  { printf '[download] %s\n' "$*"; }
need() {
    if ! command -v "$1" &>/dev/null; then
        say "MISSING TOOL: $1 — $2"
        return 1
    fi
}

# ---------------------------------------------------------------------------

target_list() {
    cat <<'LIST'
target            video?  tool needed     what arrives
----------------  ------  --------------  ------------------------------------
something_else    no      gdown           per-frame boxes, 180,049 videos
vidor             no      wget            10k videos of relation triplets
open_x            no      gsutil          many robot corpora, one RLDS format
language_table    no      gsutil          442,226 real robot episodes
robovqa           no      git             long-horizon robotics episodes
epic_kitchens     yes     python          egocentric video, boxes, verb-noun
kinetics          yes     wget            650k clips, 700 action classes

Priority order and the reason for each: notes/lit/dataset_sources.json
LIST
}

# --- 1. Something-Else -----------------------------------------------------
# Boxes only, no video. The single cheapest genuine second corpus: 180,049
# videos of per-frame boxes, plus the compositional splits M2 needs.
target_something_else() {
    local out="${DATA}/something_else/raw"
    mkdir -p "${out}"
    need gdown "pip install gdown, inside the venv only" || return 1
    say "Google Drive folder, four parts, into ${out}"
    gdown --folder "https://drive.google.com/drive/folders/1XqZC2jIHqrLPugPOVJxCH_YWa275PBrZ" \
          -O "${out}" --continue \
        || say "if gdown fails, open the link in a browser and place the parts in ${out}"
    ls -la "${out}" 2>/dev/null | tail -6
}

# --- 2. VidOR --------------------------------------------------------------
# Same annotation format as VidVRD, by the same author, 12 times larger.
# Annotations are a small download; the video is optional and not fetched here.
target_vidor() {
    local out="${DATA}/vidor/annotations"
    mkdir -p "${out}"
    say "VidOR annotations into ${out}"
    say "source page: https://xdshang.github.io/docs/vidor.html"
    for f in training_annotation.zip validation_annotation.zip; do
        wget -c -P "${out}" "https://zenodo.org/record/4084152/files/${f}" \
            || say "could not fetch ${f}; check the source page for the current link"
    done
    (cd "${out}" && for z in *.zip; do [ -e "$z" ] && unzip -n "$z"; done)
    say "annotation json files: $(find "${out}" -name '*.json' | wc -l)"
}

# --- 3. Open X-Embodiment --------------------------------------------------
# Dozens of robot corpora in one RLDS format, so one loader serves all of them.
# Every episode was recorded under a task, which is the highest structure
# available. Boxes come from a detector at bake time.
target_open_x() {
    local out="${DATA}/open_x"
    mkdir -p "${out}"
    need gsutil "part of google-cloud-sdk; on the cluster try 'module load google-cloud-sdk'" || return 1
    say "listing the available corpora rather than pulling all of them"
    gsutil ls gs://gresearch/robotics/ | head -40
    say ""
    say "pick one and fetch it, for example:"
    say "  gsutil -m cp -r gs://gresearch/robotics/bridge/0.1.0 ${out}/"
    say "sizes vary from a few GB to several hundred; check with 'gsutil du -sh' first"
}

# --- 4. Language Table -----------------------------------------------------
# 442,226 real robot episodes on a public bucket: no agreement, no form.
# Blocks on a table with an instruction per episode, which is the closest real
# analogue of blocksworld, where FOSAE worked.
target_language_table() {
    local out="${DATA}/language_table"
    mkdir -p "${out}"
    need gsutil "part of google-cloud-sdk" || return 1
    say "measuring before pulling"
    gsutil du -sh gs://gresearch/robotics/language_table/0.0.1/ || true
    say "to fetch a shard rather than the whole set:"
    say "  gsutil -m cp -r gs://gresearch/robotics/language_table/0.0.1/ ${out}/"
}

# --- 5. RoboVQA ------------------------------------------------------------
target_robovqa() {
    local out="${DATA}/robovqa"
    mkdir -p "${out}"
    say "cloning the release repository into ${out}"
    git clone --depth 1 https://github.com/google-deepmind/robovqa "${out}/repo" \
        || say "already cloned, or the clone failed"
    say "read ${out}/repo/README.md for the data location"
}

# --- 6. EPIC-KITCHENS ------------------------------------------------------
target_epic_kitchens() {
    local out="${DATA}/epic_kitchens"
    mkdir -p "${out}"
    say "cloning the official download scripts"
    git clone --depth 1 \
        https://github.com/epic-kitchens/epic-kitchens-download-scripts "${out}/scripts" \
        || say "already cloned"
    say "then, for annotations only, which is all the oracle needs:"
    say "  python3 ${out}/scripts/epic_downloader.py --output-path ${out} --annotations"
}

# --- 7. Kinetics -----------------------------------------------------------
target_kinetics() {
    local out="${DATA}/kinetics"
    mkdir -p "${out}"
    say "cloning the download helper; the corpus is large, so fetch one split"
    git clone --depth 1 https://github.com/cvdfoundation/kinetics-dataset "${out}/scripts" \
        || say "already cloned"
    say "read ${out}/scripts/README.md and run one split at a time"
}

# ---------------------------------------------------------------------------

main() {
    local target="${1:-list}"
    case "${target}" in
        list)            target_list ;;
        something_else)  target_something_else ;;
        vidor)           target_vidor ;;
        open_x)          target_open_x ;;
        language_table)  target_language_table ;;
        robovqa)         target_robovqa ;;
        epic_kitchens)   target_epic_kitchens ;;
        kinetics)        target_kinetics ;;
        *) say "unknown target '${target}'"; target_list; return 2 ;;
    esac
}

main "$@"
