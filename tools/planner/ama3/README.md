# Route A — AMA3 via Upstream Roswell + Lisp

**SPEC.md §T H1c.** Primary route for the thesis. Reproduces the paper's plannability evaluation using the original author's pipeline. No local reimplementation; orchestration only per SPEC.md §C C14 + C18.

## Status

**Skeleton only.** Implementation queued for the loop iteration after Route C smoke passes AND after the Sherlock lisp-tooling check clears.

## Idea

Delegate to upstream `/home/panoslat/Dev/Thesis/FOSAE/latplan/`:

1. `sys.path.insert(0, "/home/panoslat/Dev/Thesis/FOSAE/latplan/")` — see `upstream_bridge.py`.
2. Encode start + goal latents locally via `common/encode.py`. Write to `latent_start.csv`, `latent_goal.csv`.
3. `ros latplan/lisp/ama3-domain.ros <model_dir>/actions.csv <out_dir>/domain.pddl`
    - Requires `dump_actions` to have been called during training. If missing, generate on the fly.
4. `ros latplan/lisp/ama3-problem.ros latent_start.csv latent_goal.csv <out_dir>/problem.pddl`
5. `latplan/helper/fd-sasgz.sh --search astar(lmcut()) <out_dir>/problem.pddl` → `<out_dir>/sas_plan`
6. `ros latplan/lisp/ama3-read-latent-state-traces.ros <sas_plan>` → intermediate latents.
7. `common/decode.py::decode_trace_to_bboxes` + `common/metrics.py::bbox_mse` → metrics.json.

## Dependencies

| Dep | Provided by | Gate |
|-----|-------------|------|
| Fast Downward | `tools/planner/install_fd.sh` | H2a |
| Roswell + SBCL | `tools/planner/install_roswell.sh` (pending) | H2b |
| Upstream latplan sys.path bridge | `upstream_bridge.py` | H2c |

## Sherlock feasibility (open)

- Roswell requires SBCL. Sherlock module availability unknown. User must paste:
    ```bash
    ml spider sbcl 2>&1 | head -20
    ml spider roswell 2>&1 | head -10
    ml spider lisp 2>&1 | head -10
    which sbcl
    ```
- If Sherlock lacks SBCL/Roswell, planning eval runs **local CPU only** (planning is CPU-cheap; SPEC.md §C C17).

## Files

| File | Purpose |
|------|---------|
| `run.py` | Entry called by `plan_video.py --route a`. Orchestrator. |
| `upstream_bridge.py` | `sys.path.insert` for upstream latplan + import smoke. |
| `README.md` | This file. |

## Smoke command (H1c gate)

```bash
python3 tools/planner/plan_video.py <model_dir> --route a --plan-only
```

Gate: `sas_plan` written via upstream AMA3; `metrics.json` `reachability=true`.

## Rationale

Route A preserves the paper's plannability evaluation contract. bbox MSE comparisons across the two routes (A vs B vs C) surface differences between the paper's PDDL semantics and the simpler python schema.
