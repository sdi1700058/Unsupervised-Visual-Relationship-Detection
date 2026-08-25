# bfs

Breadth-first search over the latent space. No PDDL, no Fast Downward, no lisp.

Mines the distinct XOR deltas from the training transitions and searches over
them. The deltas carry no preconditions, so an operator applies in any state,
even one where the model never produced that transition. That makes this a
lower bound, not a reading of the learned schema.

Use it as the first smoke test: it exercises the whole search and scoring path
with nothing installed beyond numpy.

    python3 tools/planner/plan_video.py export.npz --method bfs --init 0 --goal 4

Search cost is `O(K^d)` for `K` deltas and plan length `d`, so it only suits
short windows. See `../README.md` for the task, the export format and the
metrics.
