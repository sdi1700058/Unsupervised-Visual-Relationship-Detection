#!/usr/bin/env python3
"""The plan tool: evidence strength, claim licence, and the checks that guard them.

The design these tests lock down is in `notes/docs/DESIGN_WORKPLAN.md`. The
properties that matter most, and why each exists:

- **A score never decides anything.** It localises disagreement. So every score
  must be reproducible from stored fields alone, and no score may be stored.
- **`n` counts independent units, not rows.** E1 was scored at n=276, which was
  80 windows times 2 methods with the windows overlapping at stride 1.
- **Repeating a run on one corpus must not compound.** Otherwise five runs on
  one dataset look like strong support.
- **Wording is derived from licence.** A claim whose evidence does not reach the
  bar is re-worded down a tier rather than argued about.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "tools"))

import workplan  # noqa: E402


def obs(oid="O1", n=10, datasets=("vidvrd",), paired=True,
        source="eval/planner/x/summary.csv", tier=None, supports=True):
    o = {"id": oid, "n": n, "datasets": list(datasets), "paired": paired,
         "source": source, "supports": supports}
    if tier:
        o["tier"] = tier
    return o


class TestSampleWeight(unittest.TestCase):

    def test_the_scale_anchors(self):
        self.assertAlmostEqual(workplan.w_n(1), 0.202, places=2)
        self.assertAlmostEqual(workplan.w_n(8), 0.640, places=2)
        self.assertAlmostEqual(workplan.w_n(22), 0.913, places=2)

    def test_it_saturates_and_never_exceeds_one(self):
        self.assertEqual(workplan.w_n(30), 1.0)
        self.assertEqual(workplan.w_n(100000), 1.0)

    def test_no_sample_is_worth_nothing(self):
        self.assertEqual(workplan.w_n(0), 0.0)


class TestIndependence(unittest.TestCase):

    def test_one_dataset_is_capped_at_a_half(self):
        """This is the red card, expressed as arithmetic."""
        self.assertEqual(workplan.independence(["vidvrd"]), 0.5)

    def test_two_and_three(self):
        self.assertEqual(workplan.independence(["a", "b"]), 0.8)
        self.assertEqual(workplan.independence(["a", "b", "c"]), 1.0)
        self.assertEqual(workplan.independence(["a", "b", "c", "d"]), 1.0)

    def test_the_same_dataset_twice_is_still_one(self):
        self.assertEqual(workplan.independence(["vidvrd", "vidvrd"]), 0.5)


class TestTierIsDerivedFromProvenance(unittest.TestCase):

    def test_a_command_output_is_measured(self):
        self.assertEqual(workplan.tier_of(obs(source="eval/x/summary.csv")),
                         "measured")

    def test_no_source_is_inferred_not_measured(self):
        self.assertEqual(workplan.tier_of(obs(source=None)), "inferred")

    def test_an_explicit_guess_is_honoured(self):
        self.assertEqual(workplan.tier_of(obs(source=None, tier="guess")),
                         "guess")

    def test_provenance_cannot_be_upgraded_by_hand(self):
        """Claiming 'measured' with no source must not be believed."""
        self.assertEqual(workplan.tier_of(obs(source=None, tier="measured")),
                         "inferred")


class TestEvidenceStrength(unittest.TestCase):

    def test_the_recorded_headline_values(self):
        """Claim 1: measured, n=22, one dataset, paired."""
        e = workplan.evidence_strength(obs(n=22))
        self.assertAlmostEqual(e, 1.0 * 0.911 * 0.5 * 1.0, places=2)

    def test_unpaired_costs_what_V38_says_it_costs(self):
        a = workplan.evidence_strength(obs(n=22, paired=True))
        b = workplan.evidence_strength(obs(n=22, paired=False))
        self.assertAlmostEqual(b / a, 0.6, places=6)

    def test_the_worst_falsified_claim_scores_low(self):
        """'oracle beats by 12x' was one clip. It must score near the floor."""
        e = workplan.evidence_strength(obs(n=1))
        self.assertLess(e, 0.15)

    def test_a_guess_cannot_look_strong_however_large_the_sample(self):
        e = workplan.evidence_strength(
            obs(n=100000, datasets=["a", "b", "c"], source=None, tier="guess"))
        self.assertLessEqual(e, 0.1)


class TestSupportDoesNotCompoundWithinADataset(unittest.TestCase):

    def test_five_runs_on_one_corpus_do_not_beat_one_run(self):
        many = [obs("O%d" % i, n=22) for i in range(5)]
        one = [obs("O1", n=22)]
        self.assertAlmostEqual(workplan.support(many), workplan.support(one),
                               places=6)

    def test_a_second_dataset_does_raise_it(self):
        one = [obs("O1", n=22, datasets=["vidvrd"])]
        two = [obs("O1", n=22, datasets=["vidvrd"]),
               obs("O2", n=22, datasets=["actiongenome"])]
        self.assertGreater(workplan.support(two), workplan.support(one))

    def test_contested_needs_real_evidence_on_both_sides(self):
        for_ = [obs("O1", n=22)]
        against = [obs("O2", n=22, supports=False)]
        self.assertTrue(workplan.contested(for_ + against))
        self.assertFalse(workplan.contested(for_))


class TestClaimLicence(unittest.TestCase):

    def test_nothing_measured_so_far_licenses_more_than_scoped(self):
        """e peaks at 0.46 today, bounded by having one dataset."""
        self.assertEqual(workplan.licensed_tier(0.46), "scoped")

    def test_existential_needs_half(self):
        self.assertEqual(workplan.licensed_tier(0.55), "existential")

    def test_comparative_and_universal_bars(self):
        self.assertEqual(workplan.licensed_tier(0.65), "comparative")
        self.assertEqual(workplan.licensed_tier(0.85), "universal")

    def test_below_the_floor_licenses_nothing(self):
        self.assertIsNone(workplan.licensed_tier(0.2))

    def test_a_claim_is_reworded_down_never_up(self):
        claim = {"id": "C1", "claim_type": "universal", "evidence": ["O1"]}
        got = workplan.wording_tier(claim, e=0.46)
        self.assertEqual(got, "scoped")

    def test_a_claim_never_exceeds_what_it_asked_for(self):
        claim = {"id": "C1", "claim_type": "scoped", "evidence": ["O1"]}
        self.assertEqual(workplan.wording_tier(claim, e=0.95), "scoped")


class TestDecisionWeights(unittest.TestCase):

    WEIGHTS = {"structure": 0.30, "relations": 0.25, "literature": 0.20,
               "density": 0.15, "volume": 0.10}

    def test_the_dataset_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(self.WEIGHTS.values()), 1.0, places=6)

    def test_scoring_is_the_weighted_sum(self):
        scores = {"structure": 1.0, "relations": 0.0, "literature": 0.0,
                  "density": 0.0, "volume": 0.0}
        self.assertAlmostEqual(workplan.decide(scores, self.WEIGHTS), 0.30)

    def test_a_criteria_set_with_no_weights_refuses_to_score(self):
        with self.assertRaises(workplan.NoWeights):
            workplan.decide({"structure": 1.0}, None)

    def test_sensitivity_reports_what_would_flip_the_order(self):
        a = {"structure": 1.0, "relations": 0.0, "literature": 0.0,
             "density": 0.0, "volume": 0.0}
        b = {"structure": 0.0, "relations": 1.0, "literature": 0.0,
             "density": 0.0, "volume": 0.0}
        flip = workplan.sensitivity({"A": a, "B": b}, self.WEIGHTS)
        self.assertIsNotNone(flip)
        self.assertIn("structure", flip["criterion"])


class TestCheck(unittest.TestCase):

    def base(self):
        return {"units": [], "assumptions": [], "claims": [], "questions": [],
                "availability": [], "observations": [], "decisions": {}}

    def test_a_unit_naming_a_missing_assumption_fails(self):
        plan = self.base()
        plan["units"] = [{"id": "U1", "assumes": ["A9"], "state": "not_started"}]
        problems = workplan.check(plan)
        self.assertTrue(any("A9" in p for p in problems))

    def test_evidence_that_does_not_exist_on_disk_fails(self):
        plan = self.base()
        plan["units"] = [{"id": "U1", "state": "evidence_produced",
                          "evidence": ["eval/nope/nothing.csv"]}]
        self.assertTrue(any("nothing.csv" in p for p in workplan.check(plan)))

    def test_only_the_user_may_accept(self):
        plan = self.base()
        plan["units"] = [{"id": "U1", "state": "accepted", "accepted_by": None}]
        self.assertTrue(any("accepted" in p for p in workplan.check(plan)))

    def test_a_measurement_unit_needs_a_figure(self):
        plan = self.base()
        plan["units"] = [{"id": "U1", "state": "evidence_produced",
                          "produces": "measurement",
                          "evidence": ["tools/workplan.py"]}]
        self.assertTrue(any("figure" in p for p in workplan.check(plan)))

    def test_a_falsified_assumption_with_a_live_unit_is_a_contradiction(self):
        plan = self.base()
        plan["assumptions"] = [{"id": "A1", "claim": "x",
                                "falsified": "2026-08-30"}]
        plan["units"] = [{"id": "U1", "assumes": ["A1"], "state": "accepted",
                          "accepted_by": "user"}]
        self.assertTrue(any("A1" in p for p in workplan.contradictions(plan)))

    def test_a_clean_plan_has_no_problems(self):
        self.assertEqual(workplan.check(self.base()), [])


class TestNext(unittest.TestCase):

    def test_a_blocked_unit_is_never_proposed(self):
        plan = {"units": [
            {"id": "U1", "state": "not_started", "runs_on": "local",
             "blocked_on": {"who": "user", "what": "a decision"}},
            {"id": "U2", "state": "not_started", "runs_on": "local",
             "blocked_on": None}], "assumptions": [], "claims": [],
            "questions": [], "availability": [], "observations": [],
            "decisions": {}}
        self.assertEqual(workplan.next_unit(plan)["id"], "U2")

    def test_a_unit_whose_dependency_is_unfinished_is_not_proposed(self):
        plan = {"units": [
            {"id": "U1", "state": "not_started", "runs_on": "local",
             "depends_on": ["U0"], "blocked_on": None},
            {"id": "U0", "state": "not_started", "runs_on": "local",
             "blocked_on": None}], "assumptions": [], "claims": [],
            "questions": [], "availability": [], "observations": [],
            "decisions": {}}
        self.assertEqual(workplan.next_unit(plan)["id"], "U0")

    def test_nothing_actionable_returns_none(self):
        plan = {"units": [{"id": "U1", "state": "not_started",
                           "runs_on": "sherlock", "blocked_on": None}],
                "assumptions": [], "claims": [], "questions": [],
                "availability": [], "observations": [], "decisions": {}}
        self.assertIsNone(workplan.next_unit(plan, runs_on="local"))


if __name__ == "__main__":
    unittest.main()
