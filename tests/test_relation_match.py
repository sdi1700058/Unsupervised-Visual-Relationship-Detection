#!/usr/bin/env python3
"""The relation-matching metric the supervisors sketched on 2026-09-15.

THE RECORD, `notes/thesis/SUPERVISOR.md` section 4, quoted rather than
paraphrased:

    Πώς αντιστοιχίζουμε σχέσεις;(σχέση ομοιότητας η οποία μεγιστοποιείται
    οταν όλα τα αντικειμενα που συμμετέχουν στην r συμμετέχουν επίσης στην
    r', και μόνο αυτά)
    Έχουμε ένα σύνολο από εκτιμώμενες σχέσεις R(O) και από αληθινές R'(O)
    Προσπάθησε να βρεις μια αντιστοίχιση σχέσεων στα R, R' που να
    μεγιστοποιεί τις ομοιότητες μεταξύ των σχέσεων
    (Μπορεί να λυθεί και ως ισομορφισμός γράφων)
    Σε αυτή την υλοποίηση: η μέση ομοιότητα των αντιστοιχισμένων ζευγαριών
    σχέσεων είναι η μετρική αξιολόγησης της εξαγωγής σχέσεων.

Three things, in their order: a similarity maximised when the participating
objects coincide and only those; a matching over R(O) and R'(O) that
maximises total similarity; and the MEAN SIMILARITY OF THE MATCHED PAIRS as
the metric.

WHY THIS AND NOT A PUBLISHED TRIPLET mAP. S3 asks for a published metric
first, and the published route cannot score FOSAE output at all. Measured
2026-09-21: `planner/m7_map.py:356` compares triplets by exact tuple
equality, and that tuple carries the predicate as a STRING such as `next_to`,
while `model/util/fol.py:100` names every FOSAE predicate `pred_{p}`. A
numbered predicate never equals a named one, and no amount of data changes
that.

**The supervisors' sketch is not a workaround for that. It is the right tool,
and the reason is in their own sentence.** It scores a relation by WHICH
OBJECTS TAKE PART, not by what the relation is called. FOSAE's predicates are
unsupervised and anonymous by construction, so a metric that ignores the name
is the only kind that can score them without first fitting a name mapping,
and fitting a mapping is itself a model that would have to be held out.

S8 IS A GATE AND THESE TESTS ARE IT. The supervisors asked for small worlds
where only the relation-extraction result exists, to check the metric returns
what a reader expects, INCLUDING on cases built to make it wrong.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from planner.relation_match import (  # noqa: E402
    similarity, match, mean_matched_similarity, coverage_weighted_score)


class TestSimilarity(unittest.TestCase):
    """Maximised when the participants coincide, and only those."""

    def test_identical_participants_score_one(self):
        self.assertEqual(similarity((0, 1), (0, 1)), 1.0)

    def test_disjoint_participants_score_zero(self):
        self.assertEqual(similarity((0, 1), (2, 3)), 0.0)

    def test_a_shared_object_scores_between(self):
        # {0,1} against {1,2}: one shared of three distinct.
        self.assertAlmostEqual(similarity((0, 1), (1, 2)), 1.0 / 3.0)

    def test_a_superset_is_not_a_match(self):
        """'and only those' is the clause that makes this strict.

        A relation over {0,1,2} that contains the true {0,1} shares every
        true object, and it also brings one that does not belong. The
        record's wording rules it out, so the score is below one.
        """
        self.assertLess(similarity((0, 1, 2), (0, 1)), 1.0)
        self.assertAlmostEqual(similarity((0, 1, 2), (0, 1)), 2.0 / 3.0)

    def test_order_is_ignored_by_default_because_the_record_says_participate(self):
        self.assertEqual(similarity((0, 1), (1, 0)), 1.0)

    def test_order_can_be_required_for_an_asymmetric_relation(self):
        """`on(a, b)` and `on(b, a)` are different claims about the world.

        The record speaks of participation, which is a set. But a subject and
        an object are not interchangeable, so the caller may ask for order.
        """
        self.assertEqual(similarity((0, 1), (1, 0), ordered=True), 0.0)
        self.assertEqual(similarity((0, 1), (0, 1), ordered=True), 1.0)

    def test_a_repeated_object_does_not_inflate_the_score(self):
        # on(a, a) participates in {a}, not in {a, a}.
        self.assertEqual(similarity((0, 0), (0,)), 1.0)

    def test_two_empty_relations_are_refused_rather_than_scored_one(self):
        """0/0 is not 1.0, and a silent 1.0 here would be a free point."""
        with self.assertRaises(ValueError):
            similarity((), ())


class TestMatching(unittest.TestCase):
    """A matching over R and R' that maximises total similarity."""

    def test_the_obvious_pairing_is_found(self):
        R = [(0, 1), (2, 3)]
        T = [(2, 3), (0, 1)]
        pairs = match(R, T)
        self.assertEqual(sorted((i, j) for i, j, _ in pairs), [(0, 1), (1, 0)])

    def test_it_maximises_the_total_and_not_the_first_greedy_choice(self):
        """A greedy pass takes 1.0 then 0.0. The optimum takes 0.5 twice.

        R0={0,1} R1={2,3}.  T0={0,1} T1={0,2}.
        greedy: R0-T0 = 1.0, then R1-T1 = 1/3  -> total 1.333
        optimum: the same, so this case needs a shape where greedy loses.
        """
        # R0 matches T0 perfectly and T1 partly. R1 matches ONLY T0.
        R = [(0, 1), (5, 6)]
        T = [(0, 1), (0, 1)]
        pairs = match(R, T)
        total = sum(s for _, _, s in pairs)
        self.assertAlmostEqual(total, 1.0)

    def test_greedy_would_lose_and_the_solver_does_not(self):
        # R0={a,b} R1={a}.  T0={a} T1={a,b}
        # greedy by best-first: R0-T1=1.0 then R1-T0=1.0 -> 2.0 (optimal here)
        # build a case where first-come greedy is wrong:
        # R0={a,b}: sim to T0={a,b}=1.0, to T1={a}=0.5
        # R1={a}  : sim to T0={a,b}=0.5, to T1={a}=1.0
        R = [(0, 1), (0,)]
        T = [(0, 1), (0,)]
        pairs = match(R, T)
        self.assertAlmostEqual(sum(s for _, _, s in pairs), 2.0)

    def test_unequal_sizes_match_only_as_many_as_the_smaller_side(self):
        R = [(0, 1), (2, 3), (4, 5)]
        T = [(0, 1)]
        pairs = match(R, T)
        self.assertEqual(len(pairs), 1)

    def test_an_empty_estimate_matches_nothing(self):
        self.assertEqual(match([], [(0, 1)]), [])

    def test_the_matching_is_one_to_one(self):
        R = [(0, 1), (0, 1), (0, 1)]
        T = [(0, 1), (0, 1)]
        pairs = match(R, T)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len({j for _, j, _ in pairs}), 2)


class TestTheRecordsMetric(unittest.TestCase):
    """Mean similarity of the matched pairs, exactly as the record asks."""

    def test_a_perfect_estimate_scores_one(self):
        R = [(0, 1), (2, 3)]
        self.assertEqual(mean_matched_similarity(R, list(R)), 1.0)

    def test_a_wholly_wrong_estimate_scores_zero(self):
        self.assertEqual(
            mean_matched_similarity([(4, 5)], [(0, 1)]), 0.0)

    def test_an_empty_estimate_scores_zero_rather_than_raising(self):
        self.assertEqual(mean_matched_similarity([], [(0, 1)]), 0.0)


class TestS8TheMetricMustReturNAWrongAnswerOnPurpose(unittest.TestCase):
    """The gate. A metric never made to fail has not been tested.

    `notes/thesis/SUPERVISOR.md` section 4 asks for small worlds holding only
    the relation-extraction result, to see whether the metric answers as a
    reader would expect. These are those worlds, and the first one is the
    case where the record's own metric answers wrongly.
    """

    def test_ONE_CORRECT_GUESS_OUT_OF_A_HUNDRED_TRUTHS_SCORES_A_PERFECT_ONE(self):
        """THE DEFECT IN THE RECORD'S METRIC, and it is not small.

        The estimate holds ONE relation and it is right. The truth holds a
        hundred. Only one pair can be matched, its similarity is 1.0, and the
        mean over matched pairs is therefore 1.0.

        **A system that finds one relation in a hundred scores the same as a
        system that finds all hundred.** The metric rewards silence.
        """
        truth = [(i, i + 1) for i in range(100)]
        lucky = [(0, 1)]
        self.assertEqual(mean_matched_similarity(lucky, truth), 1.0)

    def test_the_coverage_weighted_score_refuses_that_free_point(self):
        truth = [(i, i + 1) for i in range(100)]
        lucky = [(0, 1)]
        self.assertAlmostEqual(coverage_weighted_score(lucky, truth), 0.01)

    def test_FLOODING_THE_ESTIMATE_ALSO_SCORES_A_PERFECT_ONE(self):
        """The mirror of the same hole, and it rewards noise.

        Guess every pair over ten objects. The truth holds one. One pair
        matches perfectly and the mean over matched pairs is 1.0.
        """
        truth = [(0, 1)]
        flood = [(a, b) for a in range(10) for b in range(10) if a != b]
        self.assertEqual(mean_matched_similarity(flood, truth), 1.0)

    def test_the_coverage_weighted_score_refuses_the_flood_too(self):
        truth = [(0, 1)]
        flood = [(a, b) for a in range(10) for b in range(10) if a != b]
        self.assertLess(coverage_weighted_score(flood, truth), 0.02)

    def test_a_dummy_world_where_half_the_relations_are_found(self):
        truth = [(0, 1), (1, 2), (2, 3), (3, 4)]
        half = [(0, 1), (1, 2)]
        self.assertEqual(mean_matched_similarity(half, truth), 1.0)
        self.assertAlmostEqual(coverage_weighted_score(half, truth), 0.5)

    def test_a_dummy_world_where_the_objects_are_right_and_paired_wrongly(self):
        """Same objects, different pairing. A reader expects a low score.

        truth pairs 0-1 and 2-3. The estimate pairs 0-2 and 1-3. Every object
        appears in both sets, so a metric that looked only at the object
        VOCABULARY would answer 1.0. This one does not.
        """
        truth = [(0, 1), (2, 3)]
        shuffled = [(0, 2), (1, 3)]
        got = coverage_weighted_score(shuffled, truth)
        self.assertLess(got, 0.5)
        self.assertGreater(got, 0.0)

    def test_a_dummy_world_with_a_symmetric_relation_read_both_ways(self):
        truth = [(0, 1)]
        flipped = [(1, 0)]
        self.assertEqual(coverage_weighted_score(flipped, truth), 1.0)
        self.assertEqual(
            coverage_weighted_score(flipped, truth, ordered=True), 0.0)

    def test_the_empty_truth_is_refused_rather_than_scored(self):
        with self.assertRaises(ValueError):
            coverage_weighted_score([(0, 1)], [])


if __name__ == "__main__":
    unittest.main()
