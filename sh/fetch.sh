#!/usr/bin/env bash
#
# Download a file, but only from a domain on this project's allowlist.
#
# Why this exists instead of plain curl: Claude Code has no way to restrict a
# Bash command to a set of domains. WebFetch(domain:...) rules govern the
# WebFetch tool and the sandbox's network proxy, and that proxy cannot run on
# Sherlock (see sh/claude_confined.sh). A rule like Bash(curl:*arxiv.org*) only
# substring-matches the command line, so a query parameter or a redirect would
# slip straight through it. Parsing the host and checking it is the only way to
# actually enforce the list.
#
# This keeps the user-level deny on curl and wget intact. Nothing here widens
# it: the project allows sh/fetch.sh, and sh/fetch.sh allows one host set.
#
# Source of truth for the allowlist is the WebFetch(domain:...) entries in
# .claude/settings.local.json, so there is one list to maintain. An entry may
# be an exact host (arxiv.org) or a wildcard (*.githubusercontent.com).
#
# Usage:  sh/fetch.sh <https-url> [output-path]
#         sh/fetch.sh --list

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SETTINGS=$REPO/.claude/settings.local.json

die() { echo "fetch: $*" >&2; exit 1; }

[ -r "$SETTINGS" ] || die "no allowlist: $SETTINGS is missing or unreadable.
       The list comes from that file's WebFetch(domain:...) entries."

# Pull the domains out of the settings file. Deliberately fails closed: an
# empty list rejects everything rather than defaulting to permissive.
mapfile -t ALLOWED < <(grep -o 'WebFetch(domain:[^")]*)' "$SETTINGS" \
                       | cut -d: -f2 | tr -d ')' | sort -u)
[ "${#ALLOWED[@]}" -gt 0 ] || die "allowlist is empty; nothing is permitted."

if [ "${1:-}" = "--list" ]; then
  printf 'allowed domains (%d):\n' "${#ALLOWED[@]}"
  printf '  %s\n' "${ALLOWED[@]}"
  exit 0
fi

URL=${1:-}
OUT=${2:-}
[ -n "$URL" ] || die "usage: sh/fetch.sh <https-url> [output-path]"

case $URL in
  https://*) ;;
  *) die "refusing non-https URL: $URL" ;;
esac

# host = everything between the scheme and the first /, ? or :
host=${URL#https://}
host=${host%%/*}
host=${host%%\?*}
host=${host%%:*}
host=${host,,}
[ -n "$host" ] || die "could not parse a host out of: $URL"

permitted=false
for d in "${ALLOWED[@]}"; do
  d=${d,,}
  case $d in
    '*') continue ;;                                  # too broad to honour here
    '*.'*) [ "$host" = "${d#\*.}" ] && permitted=true
           case $host in *".${d#\*.}") permitted=true ;; esac ;;
    *) [ "$host" = "$d" ] && permitted=true ;;
  esac
  $permitted && break
done

if ! $permitted; then
  echo "fetch: '$host' is not on the allowlist." >&2
  echo "       Add \"WebFetch(domain:$host)\" to .claude/settings.local.json" >&2
  echo "       to permit it. Current list: sh/fetch.sh --list" >&2
  exit 1
fi

# Output must land inside the repository, which is the only writable tree in a
# confined session anyway. This makes the same guarantee for unconfined ones.
if [ -n "$OUT" ]; then
  outdir=$(cd "$(dirname "$OUT")" 2>/dev/null && pwd) \
    || die "output directory does not exist: $(dirname "$OUT")"
  case $outdir/ in
    "$REPO"/*) ;;
    *) die "refusing to write outside the repository: $outdir" ;;
  esac
  OUT=$outdir/$(basename "$OUT")
else
  OUT=$REPO/$(basename "${URL%%\?*}")
  [ "$(basename "$OUT")" != "" ] || die "cannot infer a filename from $URL"
fi

echo "fetch: $host -> $OUT"

# Redirects are followed because real download URLs need it, but the final
# host is re-checked below: a chain that ends somewhere unlisted gets its
# download discarded.
final=$(curl --proto '=https' --tlsv1.2 --location --max-redirs 5 \
             --fail --show-error --silent \
             --write-out '%{url_effective}' \
             --output "$OUT.part" "$URL") || {
  rm -f "$OUT.part"
  die "download failed: $URL"
}

fhost=${final#https://}; fhost=${fhost%%/*}; fhost=${fhost%%\?*}; fhost=${fhost%%:*}
if [ "${fhost,,}" != "$host" ]; then
  echo "fetch: redirected to '$fhost'" >&2
  ok=false
  for d in "${ALLOWED[@]}"; do
    d=${d,,}
    case $d in
      '*') continue ;;
      '*.'*) case ${fhost,,} in *".${d#\*.}"|"${d#\*.}") ok=true ;; esac ;;
      *) [ "${fhost,,}" = "$d" ] && ok=true ;;
    esac
    $ok && break
  done
  if ! $ok; then
    rm -f "$OUT.part"
    die "redirect target '$fhost' is not on the allowlist; download discarded."
  fi
fi

mv "$OUT.part" "$OUT"
echo "fetch: saved $(du -h "$OUT" | cut -f1)  $OUT"
