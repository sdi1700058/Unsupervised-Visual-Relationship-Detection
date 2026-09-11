#!/usr/bin/env bash
#
# PreToolUse hook: refuse Bash, Write and Edit when the session is not inside
# the bubblewrap confinement. Registered in .claude/settings.local.json.
#
# Why: on 2026-09-10 a stale shell held the pre-edit `thesis` function, so
# `thesis` launched an unconfined Claude. Nothing on screen showed it, and a
# python one-liner wrote into $HOME before anyone noticed. The permission rules
# alone did not stop it, because Bash(python3:*) is allowed and a script can
# write anywhere the user can. This closes that path: no confinement, no tools.
#
# Exit 2 tells Claude Code to block the call and show stderr to Claude.
#
# Escape hatch, so a bwrap outage cannot lock you out of your own repository:
#   THESIS_ALLOW_UNCONFINED=1 claude
# Set it deliberately, never by habit.

D=$(dirname "${BASH_SOURCE[0]}")

"$D/is_confined.sh" && exit 0
[ "${THESIS_ALLOW_UNCONFINED:-0}" = "1" ] && exit 0

cat >&2 <<'EOF'
BLOCKED: this Claude session is not confined to the repository.

/ is mounted read-write, so a command here could write to $HOME, $OAK or the
sibling project directories. That account belongs to someone else.

Almost certainly a stale shell: a terminal opened before ~/.bashrc was edited
still holds the old `thesis` function, which launches plain `claude`.

To fix, in the terminal under this session:
    /exit
    type thesis | grep -q claude_confined && echo OK || echo STALE
    source ~/.bashrc      # if STALE
    thesis

Tell the user this; do not try to work around it.
EOF
exit 2
