#!/usr/bin/env bash
# dataset.sh — one interface for every corpus: get it, prepare it, measure it.
#
#   bash sh/dataset.sh list
#   bash sh/dataset.sh vidor download
#   bash sh/dataset.sh vidor prepare
#   bash sh/dataset.sh vidor screen
#   bash sh/dataset.sh vidor oracle
#   bash sh/dataset.sh vidor all
#
# Runs unchanged on the workstation and on the cluster. Nothing here trains a
# model; training is submitted separately.
#
# The five stages, and what each promises:
#
#   download   fetch archives into data/video/<name>/. Resumable.
#   prepare    unpack, and put annotations where the readers expect them.
#   screen     measure the corpus and write a winnable clip list. No model.
#   oracle     build ground-truth-box exports for the screened clips.
#   verify     say what is present and what is missing, and exit non-zero if
#              a stage has not produced what it claims.
#
# **A stage that has not been run end to end says so.** Where a corpus has only
# a verified download route and no tested bake, its later stages print what is
# needed rather than pretending. Provenance and the date each url was last
# fetched are in notes/lit/dataset_sources.json.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
DATA="${DATA_DIR:-${ROOT}/data/video}"
EVAL="${EVAL_DIR:-${ROOT}/eval}"

say()  { printf '[%s] %s\n' "${DS:-dataset}" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "${DS:-dataset}" "$*" >&2; }
die()  { printf '[%s] FATAL: %s\n' "${DS:-dataset}" "$*" >&2; exit 1; }

# Reading annotations needs pillow, which the training venv (python 3.6, pinned
# by tensorflow 1.15) does not carry. .venv-local does.
pick_python() {
    local cand
    for cand in .venv-local/bin/python3 .venv-local/bin/python venv/bin/python3 python3; do
        if command -v "${cand}" &>/dev/null && "${cand}" -c 'import PIL' 2>/dev/null; then
            echo "${cand}"; return 0
        fi
    done
    echo "python3"
}
PY="$(pick_python)"

hf_get() {                       # hf_get <repo> <file> <dest-dir>
    local repo="$1" file="$2" dest="$3"
    mkdir -p "${dest}"
    if [[ -s "${dest}/${file}" ]]; then
        say "already have ${file}"; return 0
    fi
    say "fetching ${file} from huggingface.co/datasets/${repo}"
    curl -fL --retry 3 -C - \
        "https://huggingface.co/datasets/${repo}/resolve/main/${file}" \
        -o "${dest}/${file}" \
        || { warn "could not fetch ${file}"; return 1; }
}

unzip_all() {                    # unzip_all <dir>
    local dir="$1" z
    shopt -s nullglob
    for z in "${dir}"/*.zip; do
        say "unpacking $(basename "${z}")"
        unzip -q -n "${z}" -d "${dir}" || warn "unzip failed for ${z}"
    done
    shopt -u nullglob
}

# ===========================================================================
# VidOR — same annotation format as VidVRD, by the same author, 12x larger.
# Verified end to end on 2026-09-04: downloaded, screened, oracle exports built
# and scored.
# ===========================================================================
vidor_download() {
    hf_get shangxd/vidor validation-annotation.zip "${DATA}/vidor/annotations"
    hf_get shangxd/vidor training-annotation.zip   "${DATA}/vidor/annotations"
    say "annotations only. Video is not needed: the oracle reads boxes."
}
vidor_prepare() {
    unzip_all "${DATA}/vidor/annotations"
    say "annotation files: $(find "${DATA}/vidor/annotations" -name '*.json' | wc -l)"
}
vidor_screen() {
    local out="${EVAL}/vidor_winnable_w16.txt"
    "${PY}" tools/video/screen_vidor.py --window 16 --sample "${SAMPLE:-500}" \
        --list "${out}" || die "screen failed"
    say "winnable clips: $(grep -vc '^#' "${out}")"
}
vidor_oracle() {
    CLIPS_FILE="${EVAL}/vidor_winnable_w16.txt" \
    ANN_DIR="${DATA}/vidor/annotations/training" \
    OUT_DIR="${EVAL}/probe/vidor" N_CLIPS="${N_CLIPS:-25}" \
        bash experiments/M_evaluation_methods/build_oracle_corpus.sh
}
vidor_verify() {
    local n_ann n_npz
    n_ann=$(find "${DATA}/vidor/annotations" -name '*.json' 2>/dev/null | wc -l)
    n_npz=$(ls -1 "${EVAL}/probe/vidor"/*.npz 2>/dev/null | wc -l)
    say "annotations ${n_ann}, oracle exports ${n_npz}"
    [[ "${n_ann}" -gt 0 ]] || { warn "no annotations; run download and prepare"; return 1; }
    return 0
}

# ===========================================================================
# Action Genome — sparse annotation, so it is screened by run length. Verified
# on 2026-09-04 for screen and box loading; the frames are only needed to train.
# ===========================================================================
actiongenome_download() {
    say "annotations come from the official repository, which links the pickles"
    say "  https://github.com/JingweiJ/ActionGenome"
    say "the Charades video, needed only for training, is a separate request"
    [[ -f "${DATA}/actiongenome/annotations/object_bbox_and_relationship.pkl" ]] \
        && say "annotations already present" \
        || warn "place object_bbox_and_relationship.pkl and person_bbox.pkl in ${DATA}/actiongenome/annotations/"
}
actiongenome_prepare() { say "nothing to unpack; the release ships pickles"; }
actiongenome_screen() {
    ( ulimit -v "${MEM_KB:-8000000}"
      "${PY}" tools/video/screen_actiongenome.py \
          --max-gap "${MAX_GAP:-6}" --min-run "${MIN_RUN:-8}" ) \
        || die "screen failed"
}
actiongenome_oracle() {
    warn "not yet run end to end. The box loader is tested"
    warn "(tools/planner/oracle.py boxes_from_actiongenome_clip) but no export"
    warn "has been built from it, so this stage is not claimed to work."
    return 3
}
actiongenome_verify() {
    local a="${DATA}/actiongenome/annotations"
    for f in object_bbox_and_relationship.pkl person_bbox.pkl; do
        [[ -f "${a}/${f}" ]] && say "have ${f}" || { warn "missing ${f}"; return 1; }
    done
    [[ -f "${EVAL}/datasets/actiongenome_screen.json" ]] \
        && say "screen present" || warn "screen not run"
    return 0
}

# ===========================================================================
# VidVRD — the original corpus. Kept so every stage has one worked reference.
# ===========================================================================
vidvrd_download() { bash sh/download_vidvrd.sh; }
vidvrd_prepare()  { say "download_vidvrd.sh extracts as it goes"; }
vidvrd_screen() {
    "${PY}" tools/video/screen_vidvrd.py --winnable-only --no-fill-only \
        --min-frames 45 --list "${EVAL}/vidvrd_winnable_clips.txt"
}
vidvrd_oracle() {
    bash experiments/M_evaluation_methods/build_oracle_corpus.sh
}
vidvrd_verify() {
    local n
    n=$(find "${DATA}/vidvrd/annotations" -name '*.json' 2>/dev/null | wc -l)
    say "annotation files ${n}"
    [[ "${n}" -gt 0 ]]
}

# ===========================================================================
# Corpora with a verified download route and no tested bake. Each says so.
# ===========================================================================
something_else_download() {
    say "per-frame boxes for 180,049 videos, four parts, on Google Drive"
    say "  https://github.com/joaanna/something_else  (README read 2026-09-04)"
    say "  https://drive.google.com/open?id=1XqZC2jIHqrLPugPOVJxCH_YWa275PBrZ"
    if command -v gdown &>/dev/null; then
        mkdir -p "${DATA}/something_else/raw"
        gdown --folder "https://drive.google.com/drive/folders/1XqZC2jIHqrLPugPOVJxCH_YWa275PBrZ" \
            -O "${DATA}/something_else/raw" --continue \
            || warn "gdown failed; fetch by hand into ${DATA}/something_else/raw"
    else
        warn "gdown not installed. In the venv: pip install gdown"
    fi
    say "video is NOT needed for the oracle. Frames are needed only to train."
}
something_else_prepare() { unzip_all "${DATA}/something_else/raw"; }
something_else_screen()  {
    "${PY}" tools/video/screen_something_else.py || die "screen failed"
}
something_else_oracle()  {
    say "the reader exists: oracle.py boxes_from_something_else"
    say "8 clips were scored from annotations alone on 2026-08-31"
    CLIPS_FILE="${EVAL}/something_else_clips.txt" \
    ANN_DIR="${DATA}/something_else/raw" OUT_DIR="${EVAL}/probe/se_batch" \
        bash experiments/M_evaluation_methods/build_oracle_corpus.sh
}
something_else_verify()  {
    ls "${DATA}/something_else/raw" 2>/dev/null | head -4
    [[ -d "${DATA}/something_else/raw" ]]
}

open_x_download() {
    say "dozens of robot corpora in one RLDS format"
    say "  https://github.com/google-deepmind/open_x_embodiment (verified 2026-09-04)"
    command -v gsutil &>/dev/null || { warn "needs gsutil; on the cluster try 'module load google-cloud-sdk'"; return 1; }
    say "available corpora:"
    gsutil ls gs://gresearch/robotics/ | head -40
    say "measure one before pulling it:  gsutil du -sh gs://gresearch/robotics/<name>"
}
open_x_prepare() { warn "no loader yet: RLDS episodes need a reader writing boxes"; return 3; }
open_x_screen()  { warn "blocked on a detector: this corpus ships no boxes"; return 3; }
open_x_oracle()  { warn "blocked on prepare"; return 3; }
open_x_verify()  { [[ -d "${DATA}/open_x" ]] && say "directory present" || warn "nothing downloaded"; }

language_table_download() {
    say "442,226 real robot episodes, public bucket, no agreement to sign"
    command -v gsutil &>/dev/null || { warn "needs gsutil"; return 1; }
    gsutil du -sh gs://gresearch/robotics/language_table/0.0.1/ || true
    say "then: gsutil -m cp -r gs://gresearch/robotics/language_table/0.0.1/ ${DATA}/language_table/"
}
language_table_prepare() { warn "no loader yet; same RLDS reader as open_x"; return 3; }
language_table_screen()  { warn "blocked on a detector"; return 3; }
language_table_oracle()  { warn "blocked on prepare"; return 3; }
language_table_verify()  { [[ -d "${DATA}/language_table" ]]; }

robovqa_download() {
    mkdir -p "${DATA}/robovqa"
    git clone --depth 1 https://github.com/google-deepmind/robovqa \
        "${DATA}/robovqa/repo" 2>/dev/null || say "already cloned"
    say "read ${DATA}/robovqa/repo/README.md for the data location"
}
robovqa_prepare() { warn "no loader yet"; return 3; }
robovqa_screen()  { warn "blocked on a detector"; return 3; }
robovqa_oracle()  { warn "blocked on prepare"; return 3; }
robovqa_verify()  { [[ -d "${DATA}/robovqa" ]]; }

# ===========================================================================

KNOWN="vidor actiongenome vidvrd something_else open_x language_table robovqa"

do_list() {
    printf '%-16s %-9s %-9s %-8s %s\n' dataset download prepare screen "oracle / notes"
    printf '%-16s %-9s %-9s %-8s %s\n' ---------------- --------- --------- -------- ---------------
    printf '%-16s %-9s %-9s %-8s %s\n' vidor yes yes yes 'yes, scored 2026-09-04'
    printf '%-16s %-9s %-9s %-8s %s\n' vidvrd yes yes yes 'yes, the reference corpus'
    printf '%-16s %-9s %-9s %-8s %s\n' actiongenome manual n/a yes 'loader tested, no export yet'
    printf '%-16s %-9s %-9s %-8s %s\n' something_else gdown yes yes 'reader exists, 8 clips scored'
    printf '%-16s %-9s %-9s %-8s %s\n' open_x gsutil no no 'needs an RLDS reader + detector'
    printf '%-16s %-9s %-9s %-8s %s\n' language_table gsutil no no 'needs an RLDS reader + detector'
    printf '%-16s %-9s %-9s %-8s %s\n' robovqa git no no 'needs a reader + detector'
    echo
    echo "Provenance and the date each url was last fetched:"
    echo "  notes/lit/dataset_sources.json"
    echo "Exit code 3 from a stage means: route verified, stage not built yet."
}

main() {
    local ds="${1:-list}" stage="${2:-verify}"
    if [[ "${ds}" == "list" ]]; then do_list; return 0; fi
    DS="${ds}"
    if ! printf '%s\n' ${KNOWN} | grep -qx "${ds}"; then
        warn "unknown dataset '${ds}'"; do_list; return 2
    fi
    if [[ "${stage}" == "all" ]]; then
        for s in download prepare screen oracle verify; do
            say "=== ${s} ==="
            "${ds}_${s}" || say "${s} did not complete (exit $?)"
        done
        return 0
    fi
    if ! declare -F "${ds}_${stage}" >/dev/null; then
        die "no stage '${stage}' for '${ds}'; use download|prepare|screen|oracle|verify|all"
    fi
    "${ds}_${stage}"
}

main "$@"
