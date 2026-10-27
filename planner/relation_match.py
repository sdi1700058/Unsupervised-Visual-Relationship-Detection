#!/usr/bin/env python3
"""The relation-matching metric the supervisors sketched on 2026-09-15.

THE RECORD, `notes/thesis/SUPERVISOR.md` section 4, quoted and not paraphrased:

    Πώς αντιστοιχίζουμε σχέσεις;(σχέση ομοιότητας η οποία μεγιστοποιείται
    οταν όλα τα αντικειμενα που συμμετέχουν στην r συμμετέχουν επίσης στην
    r', και μόνο αυτά)
    Έχουμε ένα σύνολο από εκτιμώμενες σχέσεις R(O) και από αληθινές R'(O)
    Προσπάθησε να βρεις μια αντιστοίχιση σχέσεων στα R, R' που να
    μεγιστοποιεί τις ομοιότητες μεταξύ των σχέσεων
    (Μπορεί να λυθεί και ως ισομορφισμός γράφων)
    Σε αυτή την υλοποίηση: η μέση ομοιότητα των αντιστοιχισμένων ζευγαριών
    σχέσεων είναι η μετρική αξιολόγησης της εξαγωγής σχέσεων.

WHY THE PUBLISHED ROUTE CANNOT BE USED, and S3 asks for a published metric
first so this has to be settled rather than assumed. Measured 2026-09-21:
`planner/m7_map.py:356` compares triplets by exact tuple equality, and
that tuple carries the predicate as a STRING such as `next_to`, while
`model/util/fol.py:100` names every FOSAE predicate `pred_{p}`. **A numbered
predicate never equals a named one**, and no quantity of data changes it.

**THE SUPERVISORS' SKETCH IS NOT A WORKAROUND FOR THAT. IT IS THE RIGHT TOOL,
and the reason is in their own sentence.** It scores a relation by WHICH
OBJECTS TAKE PART, never by what the relation is called. FOSAE's predicates
are unsupervised and anonymous by construction, so a metric that ignores the
name is the only kind that can score them without first fitting a name
mapping. Fitting a mapping is itself a model, and it would have to be held
out, which turns one metric into an experiment.

TWO SCORES LIVE HERE AND THE DIFFERENCE MATTERS.

`mean_matched_similarity` is the record's metric, implemented exactly as
written. **It has a hole, and the hole is large.** The mean runs over MATCHED
pairs alone, and only `min(len(R), len(R_true))` pairs can ever match. So a
system that finds ONE relation out of a hundred and gets it right scores a
perfect 1.0, the same as a system that finds all hundred. It rewards silence.
Flooding the estimate with every possible pair scores 1.0 too, so it rewards
noise as well. `test_relation_match.py` holds both worlds.

`coverage_weighted_score` divides the same total by `max(len(R),
len(R_true))`, which closes both holes: the lucky single guess scores 0.01
and the flood scores under 0.02.

**Which one goes in the thesis is the author's, not this module's.** The
record asks for the first. The second answers the question a reader thinks
they are being told. Both are computed and both are reported.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))


def _participants(relation, ordered):
    """What a relation is, for the purpose of comparing two of them.

    The record says "every object taking part", which is participation and
    therefore a SET: `on(a, b)` and `on(b, a)` involve the same objects. A
    caller who needs subject and object kept apart asks for `ordered`.
    """
    if ordered:
        return tuple(relation)
    return frozenset(relation)


def similarity(estimated, truth, ordered=False):
    """Maximised when the same objects take part, and only those.

    The Jaccard index over the participating objects is the graded reading of
    that sentence: it reaches 1.0 exactly when the two sets are equal, and it
    falls both when a true object is missing AND when a spurious one is
    added. The second half is the record's "και μόνο αυτά", so a relation
    that covers every true object and brings an extra one scores below 1.0.
    """
    if not estimated and not truth:
        # 0/0 is not 1.0. Scoring two empty relations as a perfect match
        # would hand out a free point for saying nothing.
        raise ValueError("two empty relations have no similarity")
    if ordered:
        return 1.0 if tuple(estimated) == tuple(truth) else 0.0
    a = _participants(estimated, False)
    b = _participants(truth, False)
    union = a | b
    if not union:
        raise ValueError("two empty relations have no similarity")
    return float(len(a & b)) / float(len(union))


def _assign(cost):
    """Minimum-cost one-to-one assignment, reusing the project's own solver.

    `planner/common/metrics.py` already carries this, with a scipy path
    and a fallback for when scipy is absent. Rule 4 of the dataset criterion
    is built over buildable, and a second implementation here would be a
    second thing to keep true.
    """
    from planner.common.metrics import _assign as project_assign
    return project_assign(cost)


def match(estimated, truth, ordered=False):
    """The pairing that maximises total similarity.

    Returns a list of (index in estimated, index in truth, similarity). At
    most `min(len(estimated), len(truth))` pairs come back, because the
    matching is one to one.

    The record notes this can also be read as a graph isomorphism. It is
    solved here as an assignment problem, which is the same thing for a
    complete bipartite graph and has a polynomial solver.
    """
    import numpy as np
    if not estimated or not truth:
        return []
    sims = np.zeros((len(estimated), len(truth)), dtype=float)
    for i, e in enumerate(estimated):
        for j, t in enumerate(truth):
            try:
                sims[i, j] = similarity(e, t, ordered=ordered)
            except ValueError:
                sims[i, j] = 0.0
    # The solver minimises, and the record asks to maximise.
    rows, cols = _assign(-sims)
    return [(int(i), int(j), float(sims[i, j])) for i, j in zip(rows, cols)]


def mean_matched_similarity(estimated, truth, ordered=False):
    """THE RECORD'S METRIC, implemented as written.

    The mean similarity of the matched pairs. **Read the module docstring
    before quoting this number**: it rewards a system that reports one
    relation and stays silent about ninety-nine others.
    """
    pairs = match(estimated, truth, ordered=ordered)
    if not pairs:
        return 0.0
    return sum(s for _, _, s in pairs) / float(len(pairs))


def coverage_weighted_score(estimated, truth, ordered=False):
    """The same total, divided by the larger of the two sets.

    This is the repair for the hole in the record's metric. Dividing by
    `max(len(estimated), len(truth))` charges for every true relation that
    went unmatched AND for every guess that matched nothing, so neither
    silence nor noise buys a point.

    An empty truth is refused rather than scored, because there is nothing to
    be right or wrong about.
    """
    if not truth:
        raise ValueError("an empty truth set cannot be scored")
    pairs = match(estimated, truth, ordered=ordered)
    total = sum(s for _, _, s in pairs)
    return total / float(max(len(estimated), len(truth)))


def report(estimated, truth, ordered=False):
    """Both scores and the counts behind them, for a caller that writes prose."""
    pairs = match(estimated, truth, ordered=ordered)
    return {
        "n_estimated": len(estimated),
        "n_truth": len(truth),
        "n_matched": len(pairs),
        "mean_matched_similarity": (mean_matched_similarity(
            estimated, truth, ordered=ordered)),
        "coverage_weighted_score": (coverage_weighted_score(
            estimated, truth, ordered=ordered) if truth else None),
        "ordered": ordered,
    }
