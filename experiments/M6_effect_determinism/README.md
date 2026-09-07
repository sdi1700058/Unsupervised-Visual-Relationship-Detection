# M6 — does the same action have the same effect wherever it applies?

**Pre-registered before the first run. Everything below the line "Run it" was
written before any number existed.**

## The hypothesis, in one sentence

A FOSAE latent transition is not writable as a STRIPS operator, because the
same mined effect, applied in the other states where its precondition holds,
does not produce the successor the dataset recorded.

## Why this question and not another one

A STRIPS operator says: *in any state where the precondition holds, these
propositions become true and these become false*. The quantifier is the whole
point. It is what lets a classical planner chain operators it never saw
chained, and it is the only reason search is possible at all. Asai and Muise
build that quantifier into the architecture in the Cube-Space AutoEncoder
(entry `A6` in `notes/docs/RELATED_WORK.md`): each action becomes a constant
vector added in latent space, so the same action shifts the same bits wherever
it fires.

FOSAE has no such prior. `tools/planner/pddl/planner.py` therefore *mines*
operators after the fact: it takes every observed transition, reduces it to the
pair of bit sets that turn on and turn off, and calls each distinct pair one
operator. The precondition it writes is the weakest one the effect implies —
every add bit off, every delete bit on. Nothing in training makes that operator
true anywhere except at the transition it came from.

M6 measures how far from true it is. The measurement uses the planner's own
operator definition, so the number describes the action model the planner
actually reads and not a different one built for the occasion.

Two further reasons this unit is worth its cost. It runs on exports that are
already on disk, so it needs no training run, no GPU and no change to the
model. And it asks a question no other metric here asks: every other number
measures how accurately positions come back, and this one measures whether the
transition structure obeys a law.

## What is measured

Take a latent export. A transition is a pair of consecutive latents inside one
clip. Cross-clip pairs are dropped, because the export concatenates clips and a
pair that steps over the join records a cut and not a motion.

For a transition from `s` to `s'`:

* the **effect** is `(add, delete)`: the bits that turn on, and the bits that
  turn off;
* the **precondition** is the weakest one the effect implies: every add bit off
  in `s`, every delete bit on in `s`;
* two transitions that flip the same bits the same way are **one operator**.

Then, for each operator `e`:

* `support(e)` is how many transitions produced it;
* `applicable(e)` is how many transitions in the whole dataset start from a
  state where the precondition of `e` holds. Every member of the group is one
  of these, so `applicable(e)` is at least `support(e)`;
* `agree(e)` counts those applicable transitions whose recorded successor is
  within the tolerance of what `e` predicts. The prediction is
  `(s or add) and not delete`.

**A transition that changes nothing counts in the denominator and not in the
operator set.** A no-op is a real contradiction of an operator whose
precondition holds there: the operator says bits must move and the dataset shows
they did not. Dropping no-ops from the scan would hide that, so the count of
no-ops is reported next to the result.

An operator is **testable** when `applicable(e)` is greater than one: its
precondition holds in more than one state of the dataset, so those states could
have disagreed with each other. An operator whose precondition holds in exactly
one state is satisfied only where it fired and was never given the chance to
disagree.

The headline, `determinism`, is the pooled rate over **testable operators
alone**: `sum(agree) / sum(applicable)` restricted to them. The same pooled
rate over every operator is reported beside it as `determinism_naive`, under
that name, because the untestable operators each contribute a free 1.0 to it.
The per-operator mean is reported as well.

### The tolerance

Zero by default: the predicted successor and the recorded successor must be the
same word. The tolerance is a Hamming distance in bits, so `--tolerance 4`
accepts a successor that is four bits away from the prediction. Zero is the
STRIPS reading and it is the one the table reports. Any other value has to be
stated in the sentence that quotes the number.

## The degenerate reading, and how it is ruled out

**This metric has one obvious way to lie.** If every transition in the dataset
has its own unique effect, then every operator's precondition is satisfied only
in the one state that produced it, `applicable(e)` equals `support(e)`
everywhere, and the pooled rate is exactly 1.0. A perfect score would then mean
the model learned a lookup table of the dataset and nothing that generalises.
That is the opposite of the reading a perfect score invites.

Five guards, all reported next to the headline and all checked **before** it:

0. **A collapsed export.** An encoder that emitted one state for every frame
   produces no transition at all, so every rate is `0/0` and a measurement that
   divided by the transition count would report the emptiest export as the most
   lawful one. `n_distinct_states` is on every row and the verdict says `DEAD
   EXPORT` first, before anything else is read. This guard was added after the
   first run, which found four such exports on disk (see the section at the
   end).


1. **Effect reuse**, transitions divided by distinct effects. A value of 1.0
   means every transition is its own operator.
2. **The testable share**: the fraction of operators whose precondition holds
   in more than one state. Only those could have been contradicted. The rest
   each contribute a free 1.0 to the naive rate, and they are kept out of the
   headline.
3. **Concentration**: the share of transitions taken by the single most common
   effect, the share taken by the top ten, and the effective number of effects,
   `exp(H)` over the support distribution. An export whose effects are all
   singletons and an export with ten operators used everywhere can report the
   same pooled rate, and these numbers separate them.
4. **A shuffled control.** The successors are permuted at random across the
   dataset with a fixed seed, which keeps the state distribution and destroys
   the pairing. The control runs through the same measurement. If the measured
   headline does not beat the control's headline, the number describes the
   geometry of the state set and not anything the model learned. When the
   control has no testable operator at all, it cannot be compared; the code
   says so in the verdict rather than passing or failing the row in silence.

5. **A positive control.** A synthetic dataset built to have the Cube-Space
   property: a one-hot mode over six operators, a payload block that differs
   from clip to clip and never moves, and one operator per mode step. Each
   operator's effect is a constant two-bit vector and every state whose mode
   matches takes it. A metric that comes back negative on every dataset is worth
   nothing, and the shuffled control cannot show that this one can return a
   high rate. This control was added after the first run, for the reason given
   at the end.

The verdict function checks 0, then 1, then 2, then 4, and only then grades the
rate. The order is in the code, not in the reader.

## Run it

**Locally. No GPU, no training run, no cluster.**

```bash
bash experiments/M6_effect_determinism/run.sh
```

That builds the oracle exports it needs under `eval/exports/m6/`, then writes
`eval/m6/m6_determinism.json`, `eval/m6/m6_determinism.svg` and a table on
standard output. It runs under `ulimit -v 6000000`, because an unbounded local
run crashed this workstation on 2026-08-28.

To rerun one row by hand, on an export whose encoder did not collapse:

```bash
.venv-local/bin/python tools/planner/m6_determinism.py \
    eval/exports/U40_A2_P10_catH14-winnable88-30fps-mo3-nofill-p8_fps30_0cab2f.npz \
    --out-dir eval/m6
```

`M6_CLIPS=4 bash experiments/M6_effect_determinism/run.sh` shortens the oracle
half for a smoke run.

### The rows

Seven, over two datasets and two kinds of latent.

| row | latent | dataset |
|---|---|---|
| `FOSAE U40 P5` | trained, 200 bits | VidVRD, the 88 screened clips |
| `FOSAE U40 P10` | trained, 400 bits | the same 88 |
| `FOSAE U40 P20` | trained, 800 bits, **collapsed** | the same 88 |
| `FOSAE U20 P10` | trained, 200 bits, **collapsed** | the same 88 |
| `oracle onehot VidVRD` | hand-built, 600 bits | the first `M6_CLIPS` of the 88 |
| `oracle binary VidVRD` | hand-built, 72 bits | the same clips |
| `oracle binary VidOR` | hand-built, 72 bits | VidOR, screened clips |
| `synthetic constant-effect control` | built to be lawful | none |

Two of the four trained exports hold **one distinct state over all 8,610
frames, every bit zero**. They stay in the table as the negative control for
guard 0 rather than being replaced, because a table that quietly dropped them
would hide that half the trained exports on disk carry no state at all.

The two oracle codes are in the table as an **anchor for the metric itself**.
`tools/planner/box_geometry.py` states that one-hot gives the motion "one bin to the
right" a different effect at every start position and that the binary code
gives it five. That is an M6 statement about a code nobody trained, and its
value is known from the code's construction. If M6 does not rank the binary
oracle above the one-hot oracle on effect reuse, M6 is measuring something
other than what it claims, and the FOSAE rows say nothing.

The oracle rows also carry the second dataset. A trained FOSAE export exists for
VidVRD alone, so VidOR enters through the oracle, where no training is needed.

### Expected runtime

| stage | expectation | worst case |
|---|---|---|
| oracle exports, 24 clips | 2 to 4 minutes *(inferred: the reader parses one annotation JSON per clip)* | 15 minutes |
| the measurement, per trained export | under 2 minutes *(inferred: the scan is one pass over 8,610 transitions for each distinct effect, and the effect count cannot exceed the transition count)* | 20 minutes |
| the figure | seconds | seconds |
| everything | under 20 minutes | about an hour |

Nothing here needs a GPU and nothing writes into `git`.

## What each outcome means — decided before the run

Every threshold below is also a constant in `tools/planner/m6_determinism.py`.
Moving one after reading the data turns the measurement into a search for a
story.

| constant | value | what it is |
|---|---|---|
| `TOLERANCE` | 0 bits | the predicted and the recorded successor must match exactly |
| `MIN_REUSE` | 2.0 | below this many transitions per distinct effect, the operator set is a lookup table |
| `MIN_TESTABLE_SHARE` | 0.10 | at least this share of operators must have a precondition holding in more than one state |
| `CONTROL_MARGIN` | 0.10 | the measured rate must beat the shuffled control by this much |
| `DETERMINISTIC` | 0.90 | at or above this pooled rate the action model is lawful |
| `NON_DETERMINISTIC` | 0.50 | at or below this it is not |

### The two readings that answer nothing, checked first

| result | reading | what follows |
|---|---|---|
| **reuse below 2.0** | Nearly every transition is its own operator. The pooled rate is near 1.0 for a reason that has nothing to do with lawfulness | Report the reuse, not the rate. The action model memorised the dataset, which is itself a finding about the latent code, and the determinism question stays open |
| **testable share below 0.10** | Almost no operator can be contradicted anywhere | The same. Report the share and withhold the rate |

### The result, if the measurement is not vacuous

"The rate" below is the headline: the pooled rate over testable operators.

| result | reading | what follows |
|---|---|---|
| **rate at or above 0.90, and above the control by 0.10** | The mined operators are lawful. The transition structure is writable as STRIPS | The A6 concern does not bind on this data. Look for the planner's failures elsewhere, in the reconstruction half |
| **rate at or below 0.50** | The same operator produces different successors in different states. The symbolic model is a fiction whatever the reconstruction loss says | This is the measured form of the A6 argument, inside this project's own data. A Cube-Space prior stops being a suggestion from the literature and becomes the indicated fix. Quote it wherever `mse_ratio` is quoted |
| **between 0.50 and 0.90** | Partly lawful | Report the rate and the operator-level spread. Do not call the model either STRIPS or not |
| **rate does not beat the control by 0.10** | Whatever lawfulness the number shows is a property of the state set, not of learning | Report the control beside the rate everywhere. The row carries no evidence about the model |

### The anchor row

| result | reading | what follows |
|---|---|---|
| **binary oracle above one-hot oracle on reuse** | M6 recovers a property of these two codes that is known from their construction | Read the FOSAE rows |
| **binary oracle at or below one-hot oracle on reuse** | M6 disagrees with the code it was pointed at | Suspend the FOSAE rows and repair the metric first |

## Two limitations, stated before the run

**The trained rows come from one dataset.** VidVRD trained the only FOSAE
exports on disk. VidOR enters through the oracle, so the second dataset tests
the metric and the representation, and not the model.

**A mined operator's precondition is the weakest one, and that is a choice.**
Adding the bits that happen to be constant across a group would narrow every
operator, raise the rate and shrink `applicable`. The weakest precondition is
used because it is the one `tools/planner/pddl/planner.py` writes into the
domain file, so the number describes the planner that exists. The count of
constant bits beyond the mined precondition is reported for each row, so a
reader can see how much room the other choice would have had.
