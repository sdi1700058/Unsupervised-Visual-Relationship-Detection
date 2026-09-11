#!/usr/bin/env bash
#
# Launch Claude Code confined to this repository.
#
# Why this exists: Claude Code's built-in /sandbox cannot start on Sherlock.
# Its network layer needs a net namespace, and the cluster disables them
# (/proc/sys/user/max_net_namespaces = 0, on login and compute nodes alike),
# so bubblewrap fails before any command runs. This script uses the same tool,
# bubblewrap, without asking for the namespace the cluster refuses.
#
#   writable    this repository, plus $CLAUDE_CONFIG_DIR for session state
#   read-only   everything else, including $HOME, $OAK, $GROUP_HOME, /share
#
# The boundary is enforced by the kernel, so it covers Bash and every process
# it spawns, as well as the Read, Write and Edit tools.
#
# Two things it does NOT cover:
#   - network egress, which is unfilterable here for the same namespace reason
#   - jobs submitted with sbatch, which slurmd runs on a compute node outside
#     this process
#
# Usage:  sh/claude_confined.sh [any claude arguments]

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STATE=${CLAUDE_CONFIG_DIR:-$HOME/.claude}

if ! command -v bwrap >/dev/null 2>&1; then
  echo "claude_confined: bwrap not on PATH. On Sherlock: ml load system bubblewrap" >&2
  exit 1
fi

# Checked explicitly because thesis() silences Lmod's chatter, which would
# otherwise be the only sign that a module failed to load.
if ! command -v claude >/dev/null 2>&1; then
  echo "claude_confined: claude not on PATH. On Sherlock: ml load claude-code" >&2
  exit 1
fi

# Under .claude/, which .gitignore already excludes.
TMP=$REPO/.claude/tmp
mkdir -p "$TMP"

# Files that grant permissions or run code are pinned read-only inside the
# writable areas, so nothing running in the sandbox can widen its own policy.
# Only paths that exist can be bound, hence the test.
pins=()
for p in "$STATE/settings.json" \
         "$STATE/hooks" "$STATE/skills" "$STATE/agents" "$STATE/commands" \
         "$REPO/.claude/settings.json"; do
  [ -e "$p" ] && pins+=(--ro-bind "$p" "$p")
done

exec bwrap \
  --ro-bind / / \
  --dev /dev \
  --proc /proc \
  --tmpfs /tmp \
  --bind "$REPO" "$REPO" \
  --bind "$STATE" "$STATE" \
  ${pins[@]+"${pins[@]}"} \
  --setenv TMPDIR "$TMP" \
  --unshare-user \
  --unshare-pid \
  --unshare-ipc \
  --die-with-parent \
  claude --settings "$REPO/sh/confined-permissions.json" "$@"
