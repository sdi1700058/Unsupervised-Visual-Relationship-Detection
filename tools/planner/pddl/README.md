# pddl

Emits propositional STRIPS from the learned latents, then calls Fast Downward.

One predicate per latent bit, one operator per distinct `(add, delete)` pair
found in consecutive export rows (all clips, all splits). Preconditions come
from the bits the operator touches, so an operator that sets bit 7 requires
bit 7 to be off. That stops operators firing where the model never saw them
fire.

The goal pins every bit, positive and negative. A partial goal would let the
planner leave the untouched bits anywhere, which would make the reconstructed
frames meaningless.

Fast Downward is not installed on Sherlock; `install_fd.sh` builds it inside
a job, not on a login node.

    bash tools/planner/install_fd.sh
    python3 tools/planner/plan_video.py export.npz --method pddl --init 0 --goal 4

This is our own encoding, not the paper's. For the paper-faithful path use
`ama3`. See `../README.md` for the task, the export format and the metrics.
