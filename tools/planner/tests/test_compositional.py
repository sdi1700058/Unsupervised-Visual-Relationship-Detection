#!/usr/bin/env python3
"""Tests for M2, compositional generalisation.

The split is the whole experiment, so most of these test the split rather than
the probe: a compositional split that accidentally leaks the held-out
combination measures nothing, and would do so silently.

    python3 -m unittest tools/planner/tests/test_compositional.py
"""

import unittest
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _meta(triples):
    """A stand-in for one clip: the (subj_cat, predicate, obj_cat) it holds."""
    return {"triples": set(triples)}


class TestCompositionalSplit(unittest.TestCase):

    def _corpus(self):
        # `chase` appears with dog and with cat; `ride` only with person.
        return [
            _meta([("dog", "chase", "ball")]),
            _meta([("dog", "chase", "frisbee")]),
            _meta([("cat", "chase", "ball")]),
            _meta([("cat", "run_left", "ball")]),
            _meta([("person", "ride", "bicycle")]),
            _meta([("person", "ride", "horse")]),
            _meta([("dog", "run_left", "ball")]),
            _meta([("horse", "run_left", "ball")]),
        ]

    def test_held_out_combination_never_appears_in_training(self):
        from tools.planner.compositional import compositional_split

        corpus = self._corpus()
        tr, te, held = compositional_split(corpus, held_out_categories={"cat"})

        train_cats = {t[0] for i in tr for t in corpus[i]["triples"]}
        self.assertNotIn("cat", train_cats)
        self.assertTrue(len(te) > 0)

    def test_the_predicate_itself_must_be_seen_in_training(self):
        """Otherwise it is a novel-label test, not a compositional one.

        Compositional generalisation asks whether a predicate learned on one
        object transfers to another. If the predicate never appears in
        training at all, a failure says nothing about composition.
        """
        from tools.planner.compositional import compositional_split

        corpus = self._corpus()
        tr, te, held = compositional_split(corpus, held_out_categories={"cat"})

        train_preds = {t[1] for i in tr for t in corpus[i]["triples"]}
        for p in held["transferable_predicates"]:
            self.assertIn(p, train_preds)

        # "chase" qualifies: seen with dog in training, with cat in test.
        self.assertIn("chase", held["transferable_predicates"])

    def test_a_predicate_unique_to_the_held_out_category_is_excluded(self):
        from tools.planner.compositional import compositional_split

        corpus = self._corpus()
        tr, te, held = compositional_split(corpus,
                                           held_out_categories={"person"})
        # `ride` only ever occurs with person, so holding person out removes
        # the predicate entirely. It cannot test composition.
        self.assertNotIn("ride", held["transferable_predicates"])

    def test_empty_split_is_refused_rather_than_returned(self):
        from tools.planner.compositional import compositional_split

        corpus = self._corpus()
        with self.assertRaises(ValueError):
            compositional_split(corpus, held_out_categories={"dog", "cat",
                                                             "person", "horse"})

    def test_random_split_is_the_same_size(self):
        """The control has to be size-matched or the comparison is unfair."""
        from tools.planner.compositional import (compositional_split,
                                                 matched_random_split)

        corpus = self._corpus()
        tr, te, _ = compositional_split(corpus, held_out_categories={"cat"})
        rtr, rte = matched_random_split(len(corpus), len(te), seed=0)
        self.assertEqual(len(te), len(rte))
        self.assertEqual(len(tr), len(rtr))
        self.assertEqual(set(rtr) & set(rte), set())


class TestTripleExtraction(unittest.TestCase):

    def test_triples_pair_categories_with_predicates(self):
        from tools.planner.compositional import clip_triples

        doc = {
            "subject/objects": [{"tid": 0, "category": "dog"},
                                {"tid": 1, "category": "frisbee"}],
            "relation_instances": [
                {"subject_tid": 0, "object_tid": 1, "predicate": "chase",
                 "begin_fid": 0, "end_fid": 5},
            ],
        }
        self.assertEqual(clip_triples(doc), {("dog", "chase", "frisbee")})

    def test_a_relation_naming_an_unknown_tid_is_dropped(self):
        from tools.planner.compositional import clip_triples

        doc = {
            "subject/objects": [{"tid": 0, "category": "dog"}],
            "relation_instances": [
                {"subject_tid": 0, "object_tid": 9, "predicate": "chase",
                 "begin_fid": 0, "end_fid": 5},
            ],
        }
        self.assertEqual(clip_triples(doc), set())


class TestVerdict(unittest.TestCase):

    def test_a_drop_on_the_compositional_split_means_memorisation(self):
        from tools.planner.compositional import verdict

        v = verdict({"compositional_mAP": 0.20, "random_mAP": 0.60,
                     "prior_mAP": 0.15})
        self.assertIn("memoris", v.lower())

    def test_matching_scores_mean_the_rule_transferred(self):
        from tools.planner.compositional import verdict

        v = verdict({"compositional_mAP": 0.58, "random_mAP": 0.60,
                     "prior_mAP": 0.15})
        self.assertIn("transfer", v.lower())

    def test_no_signal_at_all_is_reported_as_such(self):
        """Both splits at the prior means the probe never worked.

        Reporting "it generalises!" when neither split beat the base rate
        would be the worst possible reading, and it is the one a naive
        difference test gives.
        """
        from tools.planner.compositional import verdict

        v = verdict({"compositional_mAP": 0.15, "random_mAP": 0.16,
                     "prior_mAP": 0.15})
        self.assertIn("no signal", v.lower())


if __name__ == "__main__":
    unittest.main()
