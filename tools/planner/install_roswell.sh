#!/usr/bin/env bash
# tools/planner/install_roswell.sh — SPEC §T H2b.
#
# Installs Roswell (Common Lisp version manager) + SBCL for Route A
# (upstream AMA3 pipeline). Prerequisites for `ros lisp/ama3-*.ros` calls.
#
# Local  : builds Roswell under $HOME/.roswell (per Roswell convention).
# Sherlock: attempts `ml load sbcl` first; falls back to source build if
#           the module is not available.
#
# Usage
#   bash tools/planner/install_roswell.sh              # default install
#   FORCE=1 bash tools/planner/install_roswell.sh      # rebuild even if present
#
# Gate: SPEC §P H2b — after run, expect `~/.roswell/bin/ros --version` to
# print a version string.

set -euo pipefail

FORCE="${FORCE:-0}"
ROSWELL_HOME="${HOME}/.roswell"
ROSWELL_BIN="${ROSWELL_HOME}/bin/ros"

log() { echo "[install_roswell] $*" >&2; }

# --- 0. Short-circuit if already installed ---
if [[ -x "${ROSWELL_BIN}" && "${FORCE}" != "1" ]]; then
    log "Roswell already installed at ${ROSWELL_BIN}"
    "${ROSWELL_BIN}" --version || true
    log "SBCL check:"
    "${ROSWELL_BIN}" run --eval '(format t "SBCL ~a~%" (lisp-implementation-version))' --eval '(sb-ext:exit)' 2>&1 | tail -3 || true
    exit 0
fi

mkdir -p "${ROSWELL_HOME}"

# --- 1. SBCL check ---
SBCL_BIN=""
if command -v sbcl >/dev/null 2>&1; then
    SBCL_BIN="$(command -v sbcl)"
    log "found system sbcl at ${SBCL_BIN}"
elif command -v ml >/dev/null 2>&1; then
    # Sherlock Lmod path.
    log "attempting `ml load sbcl` (Sherlock)"
    if ml load sbcl 2>/dev/null; then
        SBCL_BIN="$(command -v sbcl)"
        log "loaded module sbcl → ${SBCL_BIN}"
    else
        log "no sbcl module available on Sherlock; will use Roswell's bundled sbcl-bin"
    fi
fi

# --- 2. Install dependencies for building Roswell (Debian/Ubuntu heuristic) ---
if [[ "${SKIP_DEPS:-0}" != "1" ]] && command -v apt-get >/dev/null 2>&1; then
    log "checking build deps (autoconf, automake, libcurl, git)"
    for pkg in autoconf automake libcurl4-openssl-dev git; do
        if ! dpkg -s "${pkg}" >/dev/null 2>&1; then
            log "MISSING apt package: ${pkg}. Install with: sudo apt-get install ${pkg}"
            log "Set SKIP_DEPS=1 to bypass this check."
            exit 3
        fi
    done
fi

# --- 3. Fetch Roswell source ---
ROSWELL_SRC="${ROSWELL_HOME}/src"
if [[ ! -d "${ROSWELL_SRC}/.git" ]]; then
    log "cloning Roswell into ${ROSWELL_SRC}"
    git clone --depth 1 https://github.com/roswell/roswell.git "${ROSWELL_SRC}"
fi

# --- 4. Build + install Roswell ---
cd "${ROSWELL_SRC}"
log "bootstrap + configure --prefix=${ROSWELL_HOME}"
sh bootstrap
./configure --prefix="${ROSWELL_HOME}"
log "make + make install"
make -j"$(nproc 2>/dev/null || echo 2)"
make install

# --- 5. Install SBCL through Roswell if system SBCL was absent ---
export PATH="${ROSWELL_HOME}/bin:${PATH}"
if [[ -z "${SBCL_BIN}" ]]; then
    log "installing sbcl-bin via ros install"
    ros install sbcl-bin
fi
ros use sbcl-bin || true

# --- 6. Smoke ---
log "smoke: ros --version"
"${ROSWELL_BIN}" --version

log "smoke: SBCL banner"
"${ROSWELL_BIN}" run --eval '(format t "SBCL ~a~%" (lisp-implementation-version))' --eval '(sb-ext:exit)' 2>&1 | tail -3

# --- 7. Install upstream latplan Roswell packages (per upstream/install.sh:67) ---
# Required by Route A: arrival (PDDL trace validator), numcl (numeric CL),
# eazy-gnuplot, magicffi, dataloader.
if [[ "${SKIP_ROS_PKGS:-0}" != "1" ]]; then
    log "installing upstream latplan Roswell packages: numcl arrival eazy-gnuplot magicffi dataloader"
    "${ROSWELL_BIN}" dynamic-space-size=8000 install numcl arrival eazy-gnuplot magicffi dataloader
fi

# --- 8. Build upstream latplan lisp/*.bin from *.ros ---
UPSTREAM_LATPLAN="${UPSTREAM_LATPLAN:-/home/panoslat/Dev/Thesis/FOSAE/latplan}"
if [[ -d "${UPSTREAM_LATPLAN}/lisp" && "${SKIP_LISP_BUILD:-0}" != "1" ]]; then
    log "building upstream lisp/*.bin via ${UPSTREAM_LATPLAN}/lisp/Makefile"
    (cd "${UPSTREAM_LATPLAN}/lisp" && make) || {
        log "WARNING: upstream lisp build failed. Route A blocked until this succeeds."
        log "Run manually: cd ${UPSTREAM_LATPLAN}/lisp && make"
    }
fi

log "OK — Roswell installed at ${ROSWELL_HOME}"
log "NOTE: add \"export PATH=${ROSWELL_HOME}/bin:\\\$PATH\" to your shell rc"
