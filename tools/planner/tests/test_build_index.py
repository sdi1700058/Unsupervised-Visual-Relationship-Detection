#!/usr/bin/env python3
"""Extracting datasets, metrics and real-versus-synthetic from the summaries.

Three mistakes are locked down here, all of them made while building this.

**Two list styles.** The five-point summaries use `- **Data.**` and
`3. **Data.**`. Matching only the dash lost 54 of 118 bullets and produced a
coverage figure that looked like a property of the literature.

**Substring matching.** "CLEVR" matched every mention of CLEVRER, and "mAP"
matched "mapping".

**Negation, which is the serious one.** Summaries state the negative explicitly
and often: *"All rendered. No real video."* Matching "real video" there records
the exact opposite of what the sentence says, on the one field that carries the
thesis's own argument.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from tools.lit import build_index as bi  # noqa: E402


DASH = """### Z1. A Paper — Someone, 2020 · [arXiv:1](https://arxiv.org/abs/1)

- **Problem.** Something.
- **Data.** VidVRD clips. **Real video.**
- **Evaluation.** mAP on relation triplets.
- **Performance.** Good.
"""

NUMBERED = """### Z2. Another — Someone, 2021

1. **Problem.** Something.
3. **Data.** Image-based 8-puzzle and Blocksworld. **All rendered. No real video.**
4. **Evaluation.** Whether a valid plan is found.
"""


class TestBulletStyles(unittest.TestCase):

    def test_a_dash_bullet_is_read(self):
        self.assertIn("VidVRD", bi._bullet(DASH, "Data"))

    def test_a_numbered_bullet_is_read(self):
        """The style that lost 54 of 118 entries."""
        self.assertIn("Blocksworld", bi._bullet(NUMBERED, "Data"))

    def test_a_missing_bullet_is_empty_not_an_error(self):
        self.assertEqual(bi._bullet(NUMBERED, "Performance"), "")

    def test_a_bullet_stops_at_the_next_one(self):
        self.assertNotIn("Evaluation", bi._bullet(DASH, "Data"))


class TestWordBoundaries(unittest.TestCase):

    def test_clevr_does_not_match_clevrer(self):
        self.assertEqual(bi._match("Trained on CLEVRER.", bi.DATASETS),
                         ["CLEVRER"])

    def test_map_does_not_match_mapping(self):
        self.assertEqual(bi._match("A mapping of features.", bi.METRICS), [])

    def test_map_still_matches_when_it_is_the_metric(self):
        self.assertIn("mAP", bi._match("Reported as mAP at 0.5.", bi.METRICS))

    def test_several_names_in_one_bullet_all_match(self):
        got = bi._match("VidVRD and VidOR clips.", bi.DATASETS)
        self.assertEqual(got, ["VidOR", "VidVRD"])


class TestRealnessNegation(unittest.TestCase):

    def test_no_real_video_is_not_real(self):
        """The FOSAE entry itself: 'All rendered. No real video.'"""
        got = bi.parse_entry(NUMBERED)
        self.assertNotIn("real", got["realness"])
        self.assertIn("synthetic", got["realness"])

    def test_plain_real_video_is_real(self):
        self.assertIn("real", bi.parse_entry(DASH)["realness"])

    def test_rendered_not_real_is_synthetic_only(self):
        entry = ("### Z3. T\n\n- **Data.** Billiards videos. "
                 "**Rendered, not real.**\n")
        got = bi.parse_entry(entry)
        self.assertIn("synthetic", got["realness"])
        self.assertNotIn("real", got["realness"])

    def test_a_negation_does_not_swallow_a_later_affirmation(self):
        entry = ("### Z4. T\n\n- **Data.** Not simulated. "
                 "Real robot interaction throughout.\n")
        self.assertIn("real", bi.parse_entry(entry)["realness"])


class TestParse(unittest.TestCase):

    def test_the_identifier_and_url_are_kept(self):
        got = bi.parse_entry(DASH)
        self.assertEqual(got["id"], "Z1")
        self.assertEqual(got["url"], "https://arxiv.org/abs/1")

    def test_datasets_and_metrics_land_on_the_record(self):
        got = bi.parse_entry(DASH)
        self.assertEqual(got["datasets"], ["VidVRD"])
        self.assertIn("mAP", got["metrics"])

    def test_entries_are_split_on_their_headings(self):
        self.assertEqual(len(bi.split_entries("\n" + DASH + "\n" + NUMBERED)), 2)

    def test_prose_between_entries_is_not_an_entry(self):
        text = "\n" + DASH + "\nSome commentary paragraph.\n\n" + NUMBERED
        self.assertEqual(len(bi.split_entries(text)), 2)


class TestAgainstTheRealCorpus(unittest.TestCase):
    """Live assertions. They fail if extraction regresses on the real notes."""

    @classmethod
    def setUpClass(cls):
        cls.ok = os.path.isfile(bi.SOURCE)
        cls.index = bi.build() if cls.ok else None

    def test_nearly_every_summary_yields_a_data_bullet(self):
        if not self.ok:
            self.skipTest("the notes are not present in this checkout")
        c = self.index["coverage"]
        self.assertGreaterEqual(c["with_data_bullet"], c["papers"] - 4)

    def test_fosae_itself_is_recorded_as_synthetic(self):
        """If this ever flips, the central argument has been corrupted."""
        if not self.ok:
            self.skipTest("the notes are not present in this checkout")
        a2 = [p for p in self.index["papers"] if p["id"] == "A2"]
        if not a2:
            self.skipTest("A2 is not in this copy of the notes")
        self.assertIn("synthetic", a2[0]["realness"])
        self.assertNotIn("real", a2[0]["realness"])


if __name__ == "__main__":
    unittest.main()
