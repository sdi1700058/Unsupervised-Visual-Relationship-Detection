#!/usr/bin/env bash
#
# Status line for this project: says on every prompt whether the session is
# confined to the repository. Registered in .claude/settings.local.json.
#
# Claude Code feeds session JSON on stdin; this ignores it.

cat >/dev/null 2>&1

D=$(dirname "${BASH_SOURCE[0]}")

if "$D/is_confined.sh"; then
  printf '\033[32m[CONFINED sgg-thesis]\033[0m [CAVEMAN:WENYAN-ULTRA]'
else
  printf '\033[1;31m[UNCONFINED - writes can escape to $HOME]\033[0m [CAVEMAN:WENYAN-ULTRA]'
fi
