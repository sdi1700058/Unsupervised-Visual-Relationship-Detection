#!/usr/bin/env bash
#
# Exit 0 if this process is inside the bubblewrap confinement, 1 if not.
#
# The test reads kernel state directly: sh/claude_confined.sh binds / read-only,
# so a writable / means nothing is confining this process. An unconfined session
# looks identical to a confined one from the outside, which is how a stale shell
# went unnoticed on 2026-09-10 and put two files in $HOME.

opts=$(awk '$5=="/" {print $6; exit}' /proc/self/mountinfo 2>/dev/null)
case ",$opts," in
  *,ro,*) exit 0 ;;
esac
exit 1
