# Route C — BFS in Latent Space

**SPEC.md §T H1a.** Smoke baseline. No PDDL, no Fast Downward, no external dependencies beyond the already-present numpy + latplan stack.

## Idea

Breadth-first search over `{0,1}^(U*P)` with action deltas mined from the training transitions. Every training pair `(z_pre, z_suc)` produces one action delta `d = z_suc XOR z_pre`. BFS applies deltas to the current state until it reaches the goal or exhausts the time budget.

## Determinism (SPEC.md §V V15)

- Deltas are sorted lexicographically after dedupe.
- BFS expands children in delta index order.
- Result: same model + same start/goal ⇒ same plan across reruns.

## Files

| File | Purpose |
|------|---------|
| `bfs.py` | `mine_action_deltas`, `bfs_plan`, `run` entry called by `plan_video.py --route c`. |
| `README.md` | This file. |

## Dependencies

- `numpy`, `keras`, `tensorflow` (already in the project env).
- `tools/planner/common/{encode,decode,metrics}.py` for the shared pipeline.
- No PDDL, no FD, no Roswell.

## Smoke command (H1a gate)

```bash
python3 tools/planner/plan_video.py <model_dir> --route c --plan-only
```

The gate passes when `eval/planner/<model_stem>/<video_stem>/route_c/plan_0_-1/metrics.json` contains `"reachability": true`.

## Limits

- BFS complexity: `O(K^d)` for `K` deltas and plan length `d`. Small overfit models (2-3 objects, 100-200 states) give `K ≈ 100`; plans of length ≤ 5 finish in seconds. Longer plans hit the time budget.
- BFS finds ONE plan; not necessarily the shortest in a semantic sense (it is shortest in delta count).
- Uses raw XOR deltas; ignores per-delta preconditions. A delta that changes bit `i` gets applied even in a state where the paper's action-schema would not admit it. This is the price of the "no PDDL" simplification.
