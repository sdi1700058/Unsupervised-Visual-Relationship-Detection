#!/usr/bin/env python3
"""Tests for M6, effect determinism.

The metric has one obvious way to lie: a dataset in which every transition has
its own effect scores a perfect 1.0 while it has learned nothing that
generalises. Most of these tests exist to hold that guard in place, because a
metric that reports 1.0 for the worst case is worse than no metric.

    .venv-local/bin/python -m unittest \\
        tools.planner.tests.test_m6_determinism
"""

import os
import shutil
import tempfile
import unittest
import xml.dom.minidom
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _states(rows):
    """A latent array from a list of bit lists."""
    return np.asarray(rows, dtype=np.int8)


def _clip_pairs(pairs):
    """Latents and clip names from a list of (source, successor) bit lists.

    Each pair becomes its own two-frame clip, so the dataset holds exactly the
    transitions written down and no accidental ones between them.
    """
    rows, clips = [], []
    for i, (s, t) in enumerate(pairs):
        rows.append(s)
        rows.append(t)
        name = "clip%d" % i
        clips += [name, name]
    return _states(rows), clips


# A dataset in which one operator fires from three different states and always
# does the same thing. This is what lawful looks like.
LAWFUL = [([0, 0, 0, 0], [1, 0, 0, 0]),
          ([0, 1, 0, 0], [1, 1, 0, 0]),
          ([0, 0, 1, 0], [1, 0, 1, 0])]

# The same, plus one state where the operator's precondition holds and the
# dataset does something else entirely.
CONTRADICTED = LAWFUL + [([0, 0, 0, 1], [0, 0, 0, 0])]

# Every transition has its own effect and every precondition holds in exactly
# one state. The naive rate is 1.0 and the dataset has learned nothing.
DEGENERATE = [([1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
              ([0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
              ([0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
              ([0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0])]


class TestEffects(unittest.TestCase):

    def test_effect_is_the_bits_that_turn_on_and_the_bits_that_turn_off(self):
        from tools.planner.m6_determinism import group_effects

        pre = _states([[1, 0, 1, 0]])
        suc = _states([[1, 1, 0, 0]])
        groups = group_effects(pre, suc)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["add"], [1])
        self.assertEqual(groups[0]["del"], [2])
        self.assertEqual(groups[0]["support"], 1)

    def test_a_transition_that_changes_nothing_is_not_an_operator(self):
        """A no-op operator would give the planner a free action.

        It is still a transition, so it stays in the denominator of the
        determinism scan. See the no-op test further down.
        """
        from tools.planner.m6_determinism import group_effects

        pre = _states([[1, 0], [1, 0]])
        suc = _states([[1, 0], [1, 1]])
        groups = group_effects(pre, suc)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["add"], [1])

    def test_two_transitions_that_flip_the_same_bits_are_one_operator(self):
        from tools.planner.m6_determinism import group_effects

        pre = _states([[0, 0, 0], [0, 0, 1]])
        suc = _states([[1, 0, 0], [1, 0, 1]])
        groups = group_effects(pre, suc)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["support"], 2)
        self.assertEqual(groups[0]["members"], [0, 1])

    def test_the_precondition_is_add_bits_off_and_delete_bits_on(self):
        """The weakest precondition the effect implies, which is the one
        `tools/planner/pddl/planner.py` writes into the domain file."""
        from tools.planner.m6_determinism import applicable_mask

        pre = _states([[0, 1, 0],     # add bit 0 off, delete bit 1 on: holds
                       [1, 1, 0],     # add bit already on: does not hold
                       [0, 0, 0]])    # delete bit already off: does not hold
        mask = applicable_mask(pre, [0], [1])
        self.assertEqual(mask.tolist(), [True, False, False])

    def test_an_operator_with_no_precondition_bits_applies_everywhere(self):
        from tools.planner.m6_determinism import applicable_mask

        pre = _states([[0, 1], [1, 1]])
        self.assertEqual(applicable_mask(pre, [], []).tolist(), [True, True])


class TestTransitions(unittest.TestCase):

    def test_a_pair_that_crosses_a_clip_boundary_is_not_a_transition(self):
        """The export concatenates clips, so a pair over the join is a cut."""
        from tools.planner.m6_determinism import transitions

        z = _states([[0], [1], [0], [1]])
        clips = ["a", "a", "b", "b"]
        pre, suc, dropped = transitions(z, clips)
        self.assertEqual(len(pre), 2)
        self.assertEqual(dropped, 1)

    def test_without_clip_names_the_export_is_one_clip(self):
        from tools.planner.m6_determinism import transitions

        z = _states([[0], [1], [0]])
        pre, suc, dropped = transitions(z, None)
        self.assertEqual(len(pre), 2)
        self.assertEqual(dropped, 0)

    def test_clip_names_come_from_the_frame_id_prefix(self):
        from tools.planner.m6_determinism import clips_from_frame_ids

        ids = ["ILSVRC2015_train_00150010/000000",
               "ILSVRC2015_train_00150010/000001",
               "ILSVRC2015_val_00159002/000193"]
        self.assertEqual(clips_from_frame_ids(ids),
                         ["ILSVRC2015_train_00150010",
                          "ILSVRC2015_train_00150010",
                          "ILSVRC2015_val_00159002"])


class TestDeterminism(unittest.TestCase):

    def _row(self, pairs, **kw):
        from tools.planner.m6_determinism import analyse

        z, clips = _clip_pairs(pairs)
        return analyse(z, clips, **kw)

    def test_a_lawful_corpus_scores_one(self):
        r = self._row(LAWFUL)
        self.assertEqual(r["n_transitions"], 3)
        self.assertEqual(r["n_effects"], 1)
        self.assertEqual(r["applicable_total"], 3)
        self.assertEqual(r["determinism"], 1.0)

    def test_a_contradicted_operator_lowers_the_rate(self):
        r = self._row(CONTRADICTED)
        # The operator "turn bit 0 on" has its precondition satisfied in all
        # four source states, and one of them does something else.
        self.assertEqual(r["n_effects"], 2)
        self.assertAlmostEqual(r["determinism"], 0.75)
        self.assertLess(r["determinism"], r["determinism_naive"])

    def test_a_no_op_contradicts_an_operator_whose_precondition_holds(self):
        """The operator says bits must move; the dataset shows they did not."""
        r = self._row([([0, 0], [1, 0]),
                       ([0, 1], [0, 1]),
                       ([0, 0], [1, 0])])
        self.assertEqual(r["n_noop"], 1)
        self.assertEqual(r["n_effects"], 1)
        # Three sources satisfy "bit 0 off"; two of them turn bit 0 on.
        self.assertEqual(r["applicable_total"], 3)
        self.assertAlmostEqual(r["determinism"], 2.0 / 3.0)

    def test_tolerance_admits_a_near_miss(self):
        pairs = [([0, 0, 0, 0], [1, 1, 0, 0]),
                 ([0, 0, 1, 0], [1, 1, 1, 0]),
                 ([0, 0, 0, 1], [1, 0, 0, 1])]
        exact = self._row(pairs, tolerance=0)
        loose = self._row(pairs, tolerance=1)
        self.assertLess(exact["determinism"], 1.0)
        self.assertEqual(loose["determinism"], 1.0)
        self.assertEqual(loose["tolerance"], 1)

    def test_the_exact_rate_matches_a_brute_force_reference(self):
        """The zero-tolerance path takes a shortcut, so it is checked."""
        from tools.planner.m6_determinism import (analyse, group_effects,
                                                  transitions,
                                                  applicable_mask)

        rng = np.random.RandomState(0)
        z = rng.randint(0, 2, size=(40, 12)).astype(np.int8)
        clips = ["c%d" % (i // 10) for i in range(40)]
        r = analyse(z, clips, tolerance=0)

        pre, suc, _ = transitions(z, clips)
        agree = applicable = 0
        for g in group_effects(pre, suc):
            mask = applicable_mask(pre, g["add"], g["del"])
            for t in np.flatnonzero(mask):
                predicted = pre[t].copy()
                predicted[g["add"]] = 1
                predicted[g["del"]] = 0
                applicable += 1
                agree += int(np.array_equal(predicted, suc[t]))
        self.assertEqual(r["applicable_total"], applicable)
        self.assertEqual(r["agree_total"], agree)


class TestTheDegenerateReading(unittest.TestCase):
    """The reading this metric would give if nobody guarded it."""

    def _row(self, pairs, **kw):
        from tools.planner.m6_determinism import analyse

        z, clips = _clip_pairs(pairs)
        return analyse(z, clips, **kw)

    def test_every_effect_unique_scores_a_perfect_naive_rate(self):
        r = self._row(DEGENERATE)
        self.assertEqual(r["n_effects"], 4)
        self.assertEqual(r["n_transitions"], 4)
        self.assertEqual(r["determinism_naive"], 1.0)

    def test_and_is_called_degenerate_rather_than_perfect(self):
        from tools.planner.m6_determinism import verdict

        r = self._row(DEGENERATE)
        self.assertEqual(r["reuse"], 1.0)
        self.assertIn("degenerate", verdict(r).lower())

    def test_an_operator_that_can_never_be_contradicted_is_not_evidence(self):
        """A precondition satisfied in one state alone was never tested."""
        r = self._row(DEGENERATE)
        self.assertEqual(r["n_testable"], 0)
        self.assertEqual(r["testable_share"], 0.0)
        self.assertIsNone(r["determinism"])

    def test_a_lawful_corpus_is_not_called_degenerate(self):
        from tools.planner.m6_determinism import verdict

        r = self._row(LAWFUL)
        self.assertEqual(r["n_testable"], 1)
        self.assertNotIn("degenerate", verdict(r).lower())

    def test_the_verdict_checks_reuse_before_it_grades_the_rate(self):
        """Grading first would report the worst dataset as the best one."""
        from tools.planner.m6_determinism import verdict, MIN_REUSE

        r = self._row(DEGENERATE)
        r["determinism"] = 1.0          # force the grader's best case
        self.assertLess(r["reuse"], MIN_REUSE)
        self.assertIn("degenerate", verdict(r).lower())

    def test_a_vacuous_corpus_is_reported_before_the_rate(self):
        from tools.planner.m6_determinism import verdict

        row = {"reuse": 5.0, "testable_share": 0.0, "n_testable": 0,
               "determinism": 1.0, "determinism_naive": 1.0,
               "control": {"determinism": 1.0}}
        self.assertIn("vacuous", verdict(row).lower())


class TestACollapsedExport(unittest.TestCase):
    """An export whose encoder collapsed has no transition to measure.

    It would otherwise arrive as the emptiest possible dataset and leave every
    rate at 0/0. Four exports on disk are in this state, and the collapse
    tracks the latent shape rather than its size: 200 bits is dead at U20 P10
    and alive at U40 P5.
    """

    # Measured 2026-09-05 over every file in eval/exports/. These are
    # gitignored, so a checkout without them skips the check rather than
    # failing it.
    DEAD = ("H14-U20_A2_P10-150010.npz",
            "H14-U40_A2_P20-150010.npz",
            "U20_A2_P10_catH14-winnable88-30fps-mo3-nofill-p8_fps30_f2b8c5.npz",
            "U40_A2_P20_catH14-winnable88-30fps-mo3-nofill-p8_fps30_78b02d.npz")

    LIVE = ("U40_A2_P10_catH14-winnable88-30fps-mo3-nofill-p8_fps30_0cab2f.npz",
            "U40_A2_P5_catH14-winnable88-30fps-mo3-nofill-p8_fps30_c95ea4.npz")

    def _analyse(self, name, limit=None):
        """One export, optionally only its first `limit` frames.

        The live exports are sliced. Scanning 8,610 transitions against every
        one of their thousands of operators costs the suite half a minute per
        file, and the question here is only whether the guard fires, which the
        first frames answer.
        """
        from tools.planner.m6_determinism import analyse, load_pool

        path = os.path.join(str(Path(__file__).resolve().parents[3]),
                            "eval", "exports", name)
        if not os.path.isfile(path):
            self.skipTest("%s is not on disk; eval/ is gitignored" % name)
        z, clips = load_pool([path])
        if limit:
            z, clips = z[:limit], clips[:limit]
        return analyse(z, clips, label=name)

    def test_a_single_state_corpus_is_called_dead_and_not_deterministic(self):
        from tools.planner.m6_determinism import analyse, verdict

        z = _states([[0, 0, 0]] * 6)
        row = analyse(z, ["c", "c", "c", "c", "c", "c"])
        self.assertEqual(row["n_distinct_states"], 1)
        self.assertTrue(row["all_zero"])
        self.assertIsNone(row["determinism"])
        self.assertIsNone(row["determinism_naive"])
        self.assertIn("dead export", verdict(row).lower())

    def test_the_dead_check_runs_before_the_rate_is_graded(self):
        from tools.planner.m6_determinism import verdict

        row = {"n_distinct_states": 1, "n_states": 105, "all_zero": True,
               "reuse": 9.0, "testable_share": 1.0, "determinism": 1.0,
               "determinism_naive": 1.0, "control": {"determinism": 0.0}}
        self.assertIn("dead export", verdict(row).lower())

    def test_the_four_collapsed_exports_are_named_and_caught(self):
        for name in self.DEAD:
            row = self._analyse(name)
            self.assertEqual(row["n_distinct_states"], 1, name)
            self.assertEqual(row["n_effects"], 0, name)
            self.assertIn("dead export", row["verdict"].lower(), name)

    def test_a_live_export_is_not_caught_by_the_dead_check(self):
        """The guard must not swallow the exports that carry the result."""
        for name in self.LIVE:
            row = self._analyse(name, limit=300)
            self.assertGreater(row["n_distinct_states"], 1, name)
            self.assertNotIn("dead export", row["verdict"].lower(), name)


class TestThePositiveControl(unittest.TestCase):
    """A metric that is negative on every dataset is worth nothing."""

    def test_a_constant_effect_corpus_scores_one(self):
        from tools.planner.m6_determinism import analyse, synthetic_cube_corpus

        z, clips = synthetic_cube_corpus(seed=0)
        row = analyse(z, clips, label="synthetic")
        self.assertEqual(row["determinism"], 1.0)
        self.assertGreater(row["reuse"], 10.0)
        self.assertEqual(row["testable_share"], 1.0)

    def test_and_is_read_as_lawful(self):
        from tools.planner.m6_determinism import analyse, synthetic_cube_corpus

        z, clips = synthetic_cube_corpus(seed=0)
        self.assertIn("lawful", analyse(z, clips)["verdict"].lower())

    def test_its_operators_fire_from_states_that_agree_on_little(self):
        """The STRIPS signature: wide applicability, narrow precondition."""
        from tools.planner.m6_determinism import analyse, synthetic_cube_corpus

        z, clips = synthetic_cube_corpus(n_ops=6, n_payload=26, seed=0)
        row = analyse(z, clips)
        # Only the six mode bits are constant across a group's sources.
        self.assertLess(row["mean_precondition_agreement"], 0.25)

    def test_the_shuffled_control_does_not_score_one(self):
        from tools.planner.m6_determinism import analyse, synthetic_cube_corpus

        z, clips = synthetic_cube_corpus(seed=0)
        row = analyse(z, clips)
        self.assertLess(row["control"]["determinism"], 0.5)


class TestConcentration(unittest.TestCase):

    def test_reuse_is_one_when_every_transition_has_its_own_effect(self):
        from tools.planner.m6_determinism import concentration

        c = concentration([1, 1, 1, 1])
        self.assertEqual(c["reuse"], 1.0)
        self.assertEqual(c["singleton_share"], 1.0)
        self.assertAlmostEqual(c["effective_effects"], 4.0)

    def test_one_dominant_effect_shows_up_as_a_top_share(self):
        from tools.planner.m6_determinism import concentration

        c = concentration([5, 1, 1, 1])
        self.assertEqual(c["n_effects"], 4)
        self.assertEqual(c["reuse"], 2.0)
        self.assertAlmostEqual(c["top1_share"], 0.625)
        self.assertAlmostEqual(c["singleton_share"], 0.75)
        self.assertLess(c["effective_effects"], 4.0)

    def test_no_effects_at_all_is_reported_and_not_divided_by(self):
        from tools.planner.m6_determinism import concentration

        c = concentration([])
        self.assertEqual(c["n_effects"], 0)
        self.assertIsNone(c["reuse"])


class TestPreconditionAgreement(unittest.TestCase):

    def test_the_aggregate_ignores_operators_that_fired_once(self):
        """A group of one agrees with itself, so averaging them in would
        report the sparsity of the latent code and nothing else."""
        from tools.planner.m6_determinism import analyse

        # Two clips share one effect from states that differ; a third clip
        # carries a singleton effect whose single source trivially agrees.
        z, clips = _clip_pairs([([0, 0, 0, 0], [1, 0, 0, 0]),
                                ([0, 1, 0, 0], [1, 1, 0, 0]),
                                ([0, 0, 0, 1], [0, 0, 0, 0])])
        row = analyse(z, clips)
        self.assertEqual(row["n_multi_effects"], 1)
        self.assertLess(row["mean_precondition_agreement"], 1.0)

    def test_the_corpus_wide_agreement_is_reported_beside_it(self):
        """A sparse code drives both up, so only their distance is a
        property of the grouping."""
        from tools.planner.m6_determinism import analyse

        z, clips = _clip_pairs(LAWFUL)
        row = analyse(z, clips)
        self.assertIn("corpus_agreement", row)
        self.assertLessEqual(row["corpus_agreement"],
                             row["mean_precondition_agreement"])

    def test_identical_sources_agree_on_every_bit(self):
        from tools.planner.m6_determinism import precondition_agreement

        pre = _states([[0, 1, 0], [0, 1, 0]])
        a = precondition_agreement(pre, [0, 1], add=[2], dele=[])
        self.assertEqual(a["agreement"], 1.0)
        self.assertEqual(a["constant_bits"], 3)
        self.assertEqual(a["extra_bits"], 2)

    def test_sources_that_differ_agree_on_fewer_bits(self):
        from tools.planner.m6_determinism import precondition_agreement

        pre = _states([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        a = precondition_agreement(pre, [0, 1, 2], add=[0], dele=[])
        self.assertAlmostEqual(a["agreement"], 1.0 / 3.0)
        self.assertEqual(a["constant_bits"], 1)
        self.assertEqual(a["extra_bits"], 0)


class TestTheControl(unittest.TestCase):

    def test_the_control_keeps_the_successors_and_breaks_the_pairing(self):
        from tools.planner.m6_determinism import permute_successors

        suc = _states([[0, 0], [0, 1], [1, 0], [1, 1]])
        out = permute_successors(suc, seed=0)
        self.assertEqual(sorted(map(tuple, out.tolist())),
                         sorted(map(tuple, suc.tolist())))

    def test_the_control_is_reproducible(self):
        from tools.planner.m6_determinism import permute_successors

        suc = _states([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.assertTrue(np.array_equal(permute_successors(suc, seed=3),
                                       permute_successors(suc, seed=3)))

    def test_every_row_carries_a_control(self):
        from tools.planner.m6_determinism import analyse

        z, clips = _clip_pairs(LAWFUL)
        r = analyse(z, clips)
        self.assertIn("control", r)
        for key in ("reuse", "n_effects", "determinism", "determinism_naive"):
            self.assertIn(key, r["control"])


class TestPooling(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="m6-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name, rows):
        path = os.path.join(self.tmp, name)
        np.savez_compressed(path, latents=_states(rows))
        return path

    def test_two_pooled_exports_never_share_a_transition(self):
        """Otherwise the join between two files invents a transition."""
        from tools.planner.m6_determinism import load_pool, transitions

        a = self._write("a.npz", [[0], [1]])
        b = self._write("b.npz", [[0], [1]])
        z, clips = load_pool([a, b])
        pre, suc, dropped = transitions(z, clips)
        self.assertEqual(len(z), 4)
        self.assertEqual(len(pre), 2)
        self.assertEqual(dropped, 1)

    def test_a_row_argument_may_name_a_label_and_several_files(self):
        from tools.planner.m6_determinism import parse_row

        a = self._write("a.npz", [[0], [1]])
        b = self._write("b.npz", [[0], [1]])
        label, files = parse_row("oracle VidOR=%s,%s" % (a, b))
        self.assertEqual(label, "oracle VidOR")
        self.assertEqual(files, [a, b])

    def test_a_bare_path_takes_its_label_from_the_file_name(self):
        from tools.planner.m6_determinism import parse_row

        label, files = parse_row("eval/exports/H14-P10-150010.npz")
        self.assertEqual(label, "H14-P10-150010")
        self.assertEqual(files, ["eval/exports/H14-P10-150010.npz"])


class TestFigure(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="m6-svg-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _rows(self):
        from tools.planner.m6_determinism import analyse

        out = []
        for label, pairs in (("lawful", LAWFUL),
                             ("contradicted", CONTRADICTED),
                             ("degenerate", DEGENERATE)):
            z, clips = _clip_pairs(pairs)
            out.append(analyse(z, clips, label=label))
        return out

    def test_the_figure_parses_as_xml(self):
        """This project has shipped an unopenable figure three times."""
        from tools.planner.m6_determinism import write_svg

        path = os.path.join(self.tmp, "m6.svg")
        write_svg(self._rows(), path)
        xml.dom.minidom.parse(path)

    def test_the_figure_names_every_row(self):
        from tools.planner.m6_determinism import write_svg

        path = os.path.join(self.tmp, "m6.svg")
        write_svg(self._rows(), path)
        with open(path) as handle:
            text = handle.read()
        for label in ("lawful", "contradicted", "degenerate"):
            self.assertIn(label, text)

    def test_a_label_holding_an_angle_bracket_is_escaped(self):
        from tools.planner.m6_determinism import analyse, write_svg

        z, clips = _clip_pairs(LAWFUL)
        rows = [analyse(z, clips, label="U40 <P20> & more")]
        path = os.path.join(self.tmp, "m6-escape.svg")
        write_svg(rows, path)
        with open(path) as handle:
            text = handle.read()
        self.assertIn("&lt;P20&gt;", text)
        xml.dom.minidom.parse(path)


class TestCommandLine(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="m6-cli-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_it_writes_a_json_and_a_figure(self):
        from tools.planner.m6_determinism import main

        z, clips = _clip_pairs(CONTRADICTED)
        path = os.path.join(self.tmp, "dataset.npz")
        np.savez_compressed(path, latents=z,
                            frame_ids=np.asarray(
                                ["%s/%06d" % (c, i)
                                 for i, c in enumerate(clips)]))
        out = os.path.join(self.tmp, "out")
        self.assertEqual(main([path, "--out-dir", out]), 0)
        self.assertTrue(os.path.isfile(
            os.path.join(out, "m6_determinism.json")))
        xml.dom.minidom.parse(os.path.join(out, "m6_determinism.svg"))


if __name__ == "__main__":
    unittest.main()
