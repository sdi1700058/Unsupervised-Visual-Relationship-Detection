# E2 — would a temporal-distance objective fix what FOSAE fails at?

**Status: PROPOSAL. Not submitted, and it needs the user's decision first,
because it trains a model with a different objective.** `PROGRESS.md` D1 holds
changes to FOSAE's internals at the user's direction. This is a **separate
baseline alongside** FOSAE rather than a change to it, but it is close enough
to that line that it should not be run without a word.

Written 2026-08-31 because `THESIS_MAP.md` listed "one MAD-style baseline" as
the only unblocked way to strengthen the thesis, and that was **wrong** — a
fair comparison needs FOSAE's own inputs, and no baked `npz` exists on the
workstation. It is a cluster experiment.

---

## Why it is worth running

H14 answered the thesis question with a negative and a mechanism: a trained
FOSAE plans badly because its latent **compresses time**. Frames 7 apart sit 4
to 6 observed transitions apart where a positional code puts them exactly 7
(`SPEC.md` V35).

That property has a name and a method. `RELATED_WORK.md` **P1** learns an
embedding where distance *is* the minimum number of actions between states,
*"solely from state trajectories, requiring neither reward signals nor the
actions executed by the agent"* — which is exactly what video gives.

**The thesis currently says FOSAE fails and names what it lacks. It cannot say
whether the gap is closable.** One baseline changes that: an error analysis
becomes a direction.

## The experiment

Train two encoders on **identical data** — the 86 clips of
`eval/vidvrd_winnable_w16.txt`, same frames, same patches, same box encoding.

| arm | objective |
|---|---|
| **A** (control) | FOSAE as it stands. Reconstruction. Already have this: H14. |
| **B** | The same encoder, with a **temporal-distance loss** added: for frames *i* and *j* of one clip, push `‖f(i) − f(j)‖` towards `|i − j|` capped at the window. |

Arm B needs no new data and no labels. The supervision is the frame index,
which every clip already carries.

## What each outcome means, decided in advance

| result | reading |
|---|---|
| **B's `time_fid` → 1.0 and its `floor_ratio` approaches the oracle's 1.56** | The gap is closable, and the objective was the problem. The thesis gains a direction and a demonstration. |
| **B's `time_fid` → 1.0 but `floor_ratio` stays near FOSAE's 71.94** | Temporal fidelity is necessary and not sufficient. Still informative: it eliminates the leading explanation and sends the next reader elsewhere. |
| **B's `time_fid` does not improve** | The loss did not take, which is a training problem rather than a result. Check the loss actually decreased before concluding anything. |
| **B reconstructs much worse** | The objectives conflict. That is itself the finding A4 predicts — abstractions that serve planning need not serve reconstruction. |

**All four are worth having.** None of them requires B to win.

## What it does NOT test

- Not P1's method. P1 is a full quasimetric architecture; this is one auxiliary
  loss on the existing encoder. It tests **whether the property matters**, not
  whether P1 is the right way to get it.
- Not held-out generalisation, unless the clip split is held out — which it
  should be, and which the current H14 numbers are not.

## Cost

Two training runs on the data H14 already used. Arm A exists, so the marginal
cost is **one run**. The scoring is local and takes minutes.

## Before it can be submitted

1. **The user's decision.** It trains with a modified objective.
2. A held-out clip split, so B is not scored on its own training data — the
   caveat that qualifies every current number.
3. `sh/h14.sh` as the template; the bake and the export chain are already
   written and tested.

**No script is provided deliberately.** Writing one would make it easy to run
before the decision is taken.
