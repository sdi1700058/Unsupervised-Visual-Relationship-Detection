# Route B — Python-Native PDDL + Fast Downward

**SPEC.md §T H1b.** Fallback if Route A (Roswell + upstream lisp) is blocked. Same PDDL semantics; python emitter.

## Status

**Skeleton only.** `emit_domain_pddl` and `emit_problem_pddl` raise `NotImplementedError`. The plan orchestrator calls `run` which also raises. Implementation queued for the next loop iteration after Route C smoke passes and the encode/decode/metrics pipeline is validated.

## Idea

- `emit_domain.py::emit_domain_pddl` — write a STRIPS PDDL domain with one `:action` operator per unique action delta.
    - Each latent bit `i ∈ {0..U*P-1}` becomes one propositional atom `bit_i`.
    - For each delta `d = z_suc XOR z_pre`:
        - `:precondition` = observed `z_pre` bit values at every position in `d`'s support.
        - `:effect` = `(and (bit_i)…)` for bits going `0→1`; `(not (bit_i))…` for bits going `1→0`.
- `emit_problem.py::emit_problem_pddl` — write start + goal states as ground atom sets.
- Fast Downward runs on the emitted PDDL with `--search "astar(lmcut())"` (V15).

## Dependencies

- Fast Downward binary from `tools/planner/install_fd.sh` (H2a gate).
- `tools/planner/common/{encode,decode,metrics}.py`.
- No Roswell, no upstream latplan lisp code.

## Smoke command (H1b gate)

```bash
python3 tools/planner/plan_video.py <model_dir> --route b --plan-only
```

Gate: `domain.pddl` + `problem.pddl` + `sas_plan` written; `metrics.json` `reachability=true`.

## Why not just use Route A?

- Route A (upstream AMA3 via Roswell) is thesis-primary; Route B exists in case Roswell install fails on the workstation or Sherlock.
- Route B keeps the PDDL narrative alive without the lisp toolchain.
