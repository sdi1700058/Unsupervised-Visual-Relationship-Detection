"""The multi-dataset bonus is applied once, not twice.

Found on 2026-09-05 when the author doubted a strength of 0.94 and asked how
the evidence could be that strong. It could not.

`evidence_strength` multiplies by an independence factor that rewards spanning
several datasets: 0.5 for one, 0.8 for two, 1.0 for three or more. Separately,
`support` spreads each observation across the datasets it names and combines
them with a noisy-or, which rewards the same thing again.

An observation naming three datasets therefore collected the bonus twice, and
one naming a single dataset collected it once. The same measurement scored
0.94 written as one row and 0.60 written as three, which made the arithmetic
depend on bookkeeping rather than on evidence.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, os.pardir))

from tools import workplan


def obs(oid, n, datasets, paired=False):
    return {"id": oid, "n": n, "datasets": list(datasets), "paired": paired,
            "tier": "measured", "source": "setup.py", "supports": True,
            "what": "x", "experiment": "x"}


class TestBonusAppliedOnce(unittest.TestCase):

    def test_one_row_over_three_datasets_matches_three_rows(self):
        """The same measurement must score the same either way.

        Writing one observation that names three datasets, or three that name
        one each, is a bookkeeping choice. It must not change the strength.
        """
        together = [obs("A", 30, ("d1", "d2", "d3"))]
        apart = [obs("A1", 30, ("d1",)), obs("A2", 30, ("d2",)),
                 obs("A3", 30, ("d3",))]
        self.assertAlmostEqual(workplan.support(together),
                               workplan.support(apart), places=6)

    def test_the_per_dataset_share_never_carries_the_group_bonus(self):
        """Inside the spread, each dataset is worth what one dataset is worth.

        The cross-dataset credit is what the noisy-or is for; applying it to
        each share as well counts it twice.
        """
        spread = workplan._by_dataset([obs("A", 30, ("d1", "d2", "d3"))])
        single = workplan._by_dataset([obs("B", 30, ("d1",))])
        self.assertAlmostEqual(spread["d1"], single["d1"], places=6)

    def test_three_datasets_still_beat_one(self):
        """The fix must not remove the reward for breadth, only the double
        count. Three datasets are worth more than one."""
        three = workplan.support([obs("A", 30, ("d1", "d2", "d3"))])
        one = workplan.support([obs("B", 30, ("d1",))])
        self.assertGreater(three, one)

    def test_two_datasets_sit_between_one_and_three(self):
        one = workplan.support([obs("A", 30, ("d1",))])
        two = workplan.support([obs("B", 30, ("d1", "d2"))])
        three = workplan.support([obs("C", 30, ("d1", "d2", "d3"))])
        self.assertLess(one, two)
        self.assertLess(two, three)

    def test_an_unpaired_observation_is_still_discounted(self):
        paired = workplan.support([obs("A", 30, ("d1",), paired=True)])
        unpaired = workplan.support([obs("B", 30, ("d1",), paired=False)])
        self.assertGreater(paired, unpaired)

    def test_an_observation_naming_no_dataset_scores_nothing(self):
        """A measurement that cannot say what it ran on is not evidence."""
        self.assertEqual(workplan.support([obs("A", 30, ())]), 0.0)


if __name__ == "__main__":
    unittest.main()
