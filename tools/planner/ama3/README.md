# ama3

Builds the PDDL with the upstream latplan lisp code, so the domain and problem
come from the author's own emitter rather than our reimplementation. That makes
a result here the one to quote.

What runs where:

    lisp/ama3-domain.bin      effect CSVs -> domain.pddl    upstream
    helper/ama3-problem.sh    init + goal -> problem.pddl   upstream
    fast-downward.py          both files  -> a plan         ours
    replay_plan()             plan        -> latent trace   ours

The last two steps depart from upstream on purpose:

- Upstream starts the planner through `helper/fd-latest.sh`, which shells out
  to a `planner-scripts/` layout this checkout does not have. We call Fast
  Downward directly with the same search configuration, so the search is
  identical.
- Upstream turns the plan into a state trace with `arrival` and a second lisp
  binary. `arrival` hangs here with no output, and the step is redundant: Fast
  Downward guarantees the plan fits the domain it solved, and we wrote the add
  and delete sets, so replaying them gives the same trace exactly.

`ama3-domain.bin` wants the action list, the add effects and the delete
effects as three files. That is the shape the action autoencoder dumps.
FOSAE's own `dump_actions` writes only `pre|suc` rows (`latplan/model.py:1051`),
so `write_domain` derives the three files from the export.

Setup:

    sudo apt-get install -y libpng-dev      # one lisp dependency needs it
    bash tools/planner/install_fd.sh
    bash tools/planner/install_roswell.sh   # Roswell, SBCL, lisp binaries

    python3 tools/planner/plan_video.py export.npz --method ama3 --init 0 --goal 4

Sherlock has no `sbcl` and no `roswell` module, so this method runs locally
only. Planning is cheap on CPU, so that is not a problem: export on the
cluster, plan on the workstation. See `../README.md`.
