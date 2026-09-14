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
# Every path below is the pen: the author writes them from a terminal, and a
# session cannot. notes/docs/AGENTIC_DESIGN.md section 8.1 gives the reason
# for each.
#
# The list used to say "the seven below" and had grown past that. A count in a
# comment beside the thing it counts is the staleness this project keeps
# finding, so the list describes itself instead.
ROSTER=$REPO/workbench/agents/roster.json
SEATS=$REPO/workbench/agents

# Build the roster from its sources before pinning it, so a seat change needs
# no command from the author. Editing agents/foreman.md used to leave the
# built roster.json stale, the `roster` capability red, and the author holding
# a command to run before anything else could proceed.
#
# Pinning the directory covers roster.json inside it, so the file is not
# listed separately: two overlapping ro-binds on the same path are an
# untested failure mode, and a launcher that refuses to start is worse than
# the redundancy it was guarding against.
#
# The sources are pinned with the artefact, and that is the part that matters.
# Rebuilding from sources a session could write would hand a session the
# ability to grant its own seats new tools, one restart later, which is the
# single escalation the pen exists to stop. Pinning agents/ closes that: the
# author edits a seat from a terminal, the launcher rebuilds it, and a session
# can change neither the source nor the result.
#
# A seat has two halves. What it *may do* is the author's. How it *works* --
# model, effort, maxTurns, the prose -- is tuning, and pinning the directory
# locked both together. Two measured costs on the first real run came from
# tuning alone: a hand lost 45 tool calls to `maxTurns: 30`, and each
# verifier cost about 100,000 tokens because of its model tier. Neither
# needed a new tool, and both needed the author at a terminal.
#
# So `agents/tools.lock` records the capability half, and the seats are
# unpinned only while they still match it. The harness tier survives because
# the roster is built here, before the pen closes and before any session
# runs: a drifted seat file never reaches `--agents` at all.
#
# Three states, and the safe one is the default. With no lock on disk this
# behaves exactly as it did before, so installing the lock and installing
# this launcher can happen in either order without a broken night.
SEAT_LOCK=$REPO/workbench/agents/tools.lock

# seat_pins is a list because the two states pin different things, and the
# whole-directory case must not also name roster.json inside it.
seat_pins=("$SEATS")
rebuild=yes
if [ -d "$SEATS" ] && [ -f "$REPO/workbench/tools/roster.py" ]; then
  if [ ! -f "$SEAT_LOCK" ]; then
    : # No lock. The directory stays pinned, as it was.
  elif python3 "$REPO/workbench/tools/roster.py" --lock-matches \
      --lock "$SEAT_LOCK" >/dev/null 2>&1; then
    seat_pins=("$SEAT_LOCK" "$ROSTER")
  else
    echo "claude_confined: a seat's tools, permission mode or spawn" >&2
    echo "                 rights differ from agents/tools.lock. The" >&2
    echo "                 seats stay pinned. Run, from a terminal," >&2
    echo "                 python3 workbench/tools/roster.py --lock-matches" >&2
    # And the roster is not rebuilt. Building it from sources that failed
    # the lock would hand the drifted tool list straight to --agents, which
    # is the one escalation this gate exists to stop.
    rebuild=no
  fi
  if [ "$rebuild" = yes ] && \
      ! python3 "$REPO/workbench/tools/roster.py" --out "$ROSTER" \
        >/dev/null; then
    echo "claude_confined: the seat sources do not build, keeping the" >&2
    echo "                 roster that is already on disk" >&2
  fi
fi

pins=()
for p in "$STATE/settings.json" \
         "$STATE/hooks" "$STATE/skills" "$STATE/agents" "$STATE/commands" \
         "$REPO/.claude/settings.json" \
         "$REPO/sh/confined-permissions.json" \
         "${seat_pins[@]}" \
         "$REPO/workbench/notes/queue/current.json" \
         "$REPO/workbench/notes/queue/rejected.json" \
         "$REPO/workbench/notes/queue/canaries.json" \
         "$REPO/workbench/notes/queue/revoked.json" \
         "$REPO/workbench/notes/accepted.json" \
         "$REPO/workbench/notes/supervisor_docs.txt"; do
  [ -e "$p" ] && pins+=(--ro-bind "$p" "$p")
done

# A missing roster must be loud. `claude --agents ""` exits 0, so an empty
# string starts a session with no seats, and that looks the same as a session
# where nobody configured them. Guard on the file, never on the variable.
AGENTS=()
if [ -f "$ROSTER" ]; then
  AGENTS=(--agents "$(cat "$ROSTER")")
else
  echo "claude_confined: no roster at $ROSTER, starting with no seats" >&2
fi

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
  claude --settings "$REPO/sh/confined-permissions.json" \
         ${AGENTS[@]+"${AGENTS[@]}"} "$@"
