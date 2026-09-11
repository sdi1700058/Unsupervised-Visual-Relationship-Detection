# `sh/`

Shell entry points: dataset acquisition and preparation, the detector
environment, and the launcher set described below.

## The launcher set — keep these six together, and keep them here

`claude_confined.sh`, `confined-permissions.json`, `is_confined.sh`,
`statusline.sh`, `guard_confined.sh`, `fetch.sh`.

They restrict a development session to this repository with bubblewrap, so no
write can land outside it. Three properties are not visible from the code:

1. **They must stay one directory below the repository root, together.**
   `claude_confined.sh` derives the root from its own location and reaches its
   siblings by that path.
2. **The registration does not travel.** `.claude/` is gitignored, so the
   status line and the `PreToolUse` hook have to be re-added on any new clone.
   `fetch.sh` reads its domain allowlist from the same place and refuses every
   download until it exists.
3. **The hook fails open.** Only exit code 2 blocks a call. If
   `guard_confined.sh` is moved or loses its executable bit, it stops guarding
   without saying so. The status line is the backstop; it reads kernel state.

Renaming or moving this repository breaks the launcher, and the fix is outside
the repository.

**The full reasoning, the verification commands and the JSON to re-add are in
`notes/docs/SANDBOX.md`.** Read that before reorganising anything here.
