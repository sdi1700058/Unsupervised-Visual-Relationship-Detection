"""Datasets are scored before any is chosen, and the order is enforced.

The availability bias survived four corrections because it moved upstream. The
author struck availability from the score and set its weight to zero, and the
bias continued, because only three datasets were ever scored and those three
were the three already on disk. An unscored dataset cannot win a comparison.

So the guard is not another instruction. Every candidate is scored before any
is chosen, and a selection out of score order fails unless the plan carries a
written reason.
"""

import os
import shutil
import sys
import tempfile
import unittest
import xml.dom.minidom

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, os.pardir))

from tools import score_datasets


def candidate(name, **kw):
    base = {"name": name, "structure": 0.5, "relations": 0.5,
            "purpose": 0.5, "usage": 0.5, "density": 0.5, "volume": 0.5}
    base.update(kw)
    return base


class TestWeights(unittest.TestCase):

    def test_the_weights_are_the_authors(self):
        self.assertEqual(score_datasets.WEIGHTS["structure"], 0.30)
        self.assertEqual(score_datasets.WEIGHTS["relations"], 0.25)
        self.assertEqual(score_datasets.WEIGHTS["purpose"], 0.14)
        self.assertEqual(score_datasets.WEIGHTS["usage"], 0.06)
        self.assertEqual(score_datasets.WEIGHTS["density"], 0.15)
        self.assertEqual(score_datasets.WEIGHTS["volume"], 0.10)

    def test_purpose_outweighs_usage(self):
        """The author's instruction of 2026-09-05. How close the other work is
        to this task matters more than how much of it there is."""
        self.assertGreater(score_datasets.WEIGHTS["purpose"],
                           score_datasets.WEIGHTS["usage"])

    def test_availability_is_not_a_criterion(self):
        """Struck by the author. Its absence is the point of the module."""
        self.assertNotIn("availability", score_datasets.WEIGHTS)

    def test_the_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(score_datasets.WEIGHTS.values()), 1.0)


class TestScore(unittest.TestCase):

    def test_an_all_half_candidate_scores_a_half(self):
        self.assertAlmostEqual(score_datasets.score(candidate("x")), 0.5)

    def test_structure_moves_the_score_most(self):
        """Structure carries 0.30, the heaviest weight the author set."""
        by_structure = score_datasets.score(candidate("a", structure=1.0))
        by_volume = score_datasets.score(candidate("b", volume=1.0))
        self.assertGreater(by_structure, by_volume)

    def test_a_missing_criterion_gives_none_rather_than_zero(self):
        """Scoring a gap as zero is how an unmeasured dataset loses for being
        new, which is the bias wearing a different hat."""
        self.assertIsNone(score_datasets.score({"name": "p", "structure": 1.0}))

    def test_a_perfect_candidate_scores_one(self):
        perfect = dict((k, 1.0) for k in score_datasets.WEIGHTS)
        perfect["name"] = "best"
        self.assertAlmostEqual(score_datasets.score(perfect), 1.0)


class TestRank(unittest.TestCase):

    def test_rank_puts_the_best_first(self):
        ranked = score_datasets.rank([candidate("low", structure=0.1),
                                      candidate("high", structure=0.9)])
        self.assertEqual(ranked[0]["name"], "high")

    def test_unscored_candidates_come_last_and_are_marked(self):
        """An unscored dataset is not a bad dataset. It is an unmeasured one,
        and the ranking must say so rather than implying a verdict."""
        ranked = score_datasets.rank([{"name": "unmeasured"},
                                      candidate("scored")])
        self.assertEqual(ranked[-1]["name"], "unmeasured")
        self.assertIsNone(ranked[-1]["score"])

    def test_an_empty_list_ranks_to_nothing(self):
        self.assertEqual(score_datasets.rank([]), [])


class TestOutOfOrder(unittest.TestCase):

    def setUp(self):
        self.ranked = score_datasets.rank([candidate("best", structure=0.9),
                                           candidate("worse", structure=0.1)])

    def test_selecting_a_lower_ranked_dataset_is_reported(self):
        plan = {"datasets": [{"name": "worse", "selected": True}]}
        bad = score_datasets.out_of_order(plan, self.ranked)
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0]["dataset"], "worse")

    def test_a_written_reason_makes_it_pass(self):
        """The rule is not that the top must win. It is that a departure is
        argued rather than drifted into."""
        plan = {"datasets": [{"name": "worse", "selected": True,
                             "reason": "it ships compositional splits"}]}
        self.assertEqual(score_datasets.out_of_order(plan, self.ranked), [])

    def test_selecting_the_top_ranked_dataset_passes(self):
        plan = {"datasets": [{"name": "best", "selected": True}]}
        self.assertEqual(score_datasets.out_of_order(plan, self.ranked), [])

    def test_an_unscored_selection_is_reported(self):
        """Selecting something nobody measured is the original failure."""
        ranked = score_datasets.rank([candidate("best"), {"name": "mystery"}])
        plan = {"datasets": [{"name": "mystery", "selected": True}]}
        bad = score_datasets.out_of_order(plan, ranked)
        self.assertEqual(len(bad), 1)


class TestFigure(unittest.TestCase):

    def test_the_figure_parses(self):
        ranked = score_datasets.rank([candidate("a"), candidate("b")])
        xml.dom.minidom.parseString(score_datasets.render_svg(ranked))

    def test_the_figure_survives_no_candidates(self):
        xml.dom.minidom.parseString(score_datasets.render_svg([]))

    def test_a_name_with_an_angle_bracket_does_not_break_the_figure(self):
        ranked = score_datasets.rank([candidate("a < b")])
        xml.dom.minidom.parseString(score_datasets.render_svg(ranked))


class TestMain(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_missing_candidates_file_reports_rather_than_passing(self):
        code = score_datasets.main(["--candidates",
                                    os.path.join(self.dir, "none.json")])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()


class TestMethodWeights(unittest.TestCase):
    """Evaluation methods are ingredients too, and were never compared."""

    def test_the_method_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(score_datasets.METHOD_WEIGHTS.values()), 1.0)

    def test_discriminating_outweighs_everything(self):
        """A measurement that scores everything alike answers nothing,
        however respectable it is."""
        for name, weight in score_datasets.METHOD_WEIGHTS.items():
            if name == "discriminates":
                continue
            self.assertGreater(score_datasets.METHOD_WEIGHTS["discriminates"],
                               weight)

    def test_purpose_outweighs_usage_for_methods_too(self):
        self.assertGreater(score_datasets.METHOD_WEIGHTS["purpose"],
                           score_datasets.METHOD_WEIGHTS["usage"])

    def test_a_method_missing_a_criterion_is_unscored(self):
        """A method with no result yet must not be ranked as though it had
        one. Absent is a statement about our knowledge."""
        partial = {"name": "untested", "purpose": 0.8, "usage": 0.5}
        self.assertIsNone(score_datasets.score(
            partial, score_datasets.METHOD_WEIGHTS))

    def test_the_figure_draws_method_criteria(self):
        method = dict((k, 0.5) for k in score_datasets.METHOD_WEIGHTS)
        method["name"] = "m"
        ranked = score_datasets.rank([method], score_datasets.METHOD_WEIGHTS)
        svg = score_datasets.render_svg(ranked, score_datasets.METHOD_WEIGHTS)
        xml.dom.minidom.parseString(svg)
        self.assertIn("discriminates", svg)
