# Superseded sweeps

These ran and produced results. They are kept because they are the record of
which script made which directory under `out/`, and a result whose command is
lost cannot be reproduced or defended.

**Do not run them.** They carry their own forked copies of `bake` and
`submit`, which have drifted from the shared ones in `sh/sweep_lib.sh`. Start
from `sh/sweep3.sh` instead.

| Script | Ran | What it swept | Why it was replaced |
|---|---|---|---|
| `overnight_sweep.sh` | August 2026 | 16 runs. `ZEROSUPPRESS` off and 0.2, `MAX_TEMPERATURE` 0.5 and 5.0, `U/A/P` at 10/2/8 and 20/2/16, batch 32, `all_pairs`, four clips, two whole categories. | Every result landed between 0.488 and 0.597 while MNIST reached 1.0e-07. When no knob moves the number, the swept knobs are not the cause. It also never varied patch size, which turned out to matter. |
| `sweep2.sh` | August 2026 | Patch size 4 to 64, larger categories, multi-clip overfit, capacity at `U=80, P=40`. | Found the answer it was built to find: only the runs with the pre-encoder on learned. `sweep3.sh` holds the pre-encoder on and moves everything else around it. Its baked npz are reused, so it is still worth reading for the `clip_stem` naming. |

## The bug worth remembering

`overnight_sweep.sh` writes its `submit` as:

```bash
env "$@" DOMAIN=vidvrd EPOCH="${EPOCH}" NO_EARLYSTOP=1 \
    MEM=16G TIME=1:00:00 AUTO_RESOURCES=0 \
    bash sh/submit.sh
```

`"$@"` comes first, so the caller's variables are overwritten by the ones that
follow rather than the other way round. Its own `TIME=2:00:00` on line 175 was
discarded in silence, and every arm ran with `MEM=16G TIME=1:00:00` whatever it
asked for. `sh/sweep_lib.sh` puts `"$@"` last for this reason.
