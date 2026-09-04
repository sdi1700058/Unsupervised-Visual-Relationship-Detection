#!/usr/bin/env python3
"""Generated cross-reference blocks: replace the block, never the prose.

The whole point of generating these is that hand-written cross-references rot.
That only holds if regeneration is safe, so the properties worth locking down
are: the surrounding prose survives, a second run changes nothing, and a
document without markers is skipped rather than mangled.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from tools.lit import render_index as ri  # noqa: E402


def index(papers):
    return {"papers": papers}


def paper(pid, rank=None, datasets=(), metrics=(), realness=()):
    p = {"id": pid, "datasets": list(datasets), "metrics": list(metrics),
         "realness": list(realness), "title": pid, "url": ""}
    if rank:
        p["shortlist_rank"] = rank
    return p


DOC = """# A document

Hand-written prose above.

<!-- generated: papers-by-dataset -->

old content that must be replaced

<!-- /generated -->

Hand-written prose below.
"""


class TestReplaceBlock(unittest.TestCase):

    def test_the_body_is_replaced(self):
        new, found = ri.replace_block(DOC, "papers-by-dataset", "NEW BODY")
        self.assertTrue(found)
        self.assertIn("NEW BODY", new)
        self.assertNotIn("old content", new)

    def test_prose_on_both_sides_survives(self):
        new, _ = ri.replace_block(DOC, "papers-by-dataset", "NEW BODY")
        self.assertIn("Hand-written prose above.", new)
        self.assertIn("Hand-written prose below.", new)

    def test_a_document_without_markers_is_left_alone(self):
        text = "# Plain\n\nNothing generated here.\n"
        new, found = ri.replace_block(text, "papers-by-dataset", "X")
        self.assertFalse(found)
        self.assertEqual(new, text)

    def test_an_unclosed_marker_is_not_treated_as_a_block(self):
        text = "<!-- generated: papers-by-dataset -->\nno end marker\n"
        new, found = ri.replace_block(text, "papers-by-dataset", "X")
        self.assertFalse(found)
        self.assertEqual(new, text)

    def test_replacing_twice_is_stable(self):
        once, _ = ri.replace_block(DOC, "papers-by-dataset", "BODY")
        twice, _ = ri.replace_block(once, "papers-by-dataset", "BODY")
        self.assertEqual(once, twice)


class TestBlocks(unittest.TestCase):

    def test_a_dataset_gathers_its_papers_and_their_metrics(self):
        body = ri.papers_by_dataset(index([
            paper("B1", datasets=["VidVRD"], metrics=["mAP"]),
            paper("B3", datasets=["VidVRD", "VidOR"], metrics=["mAP"])]))
        self.assertIn("**VidVRD** (2)", body)
        self.assertIn("B1, B3", body)

    def test_a_metric_gathers_the_datasets_it_was_reported_on(self):
        body = ri.papers_by_metric(index([
            paper("M1", datasets=["Human3.6M"], metrics=["MPJPE"])]))
        self.assertIn("**MPJPE** (1)", body)
        self.assertIn("Human3.6M", body)

    def test_a_paper_naming_nothing_shows_a_dash_not_a_blank(self):
        body = ri.shortlist_usage(index([paper("A3", rank=1)]))
        self.assertIn("—", body)

    def test_the_shortlist_keeps_its_ranking(self):
        body = ri.shortlist_usage(index([paper("Z", rank=2), paper("A", rank=1)]))
        self.assertLess(body.index("**A**"), body.index("**Z**"))

    def test_only_shortlisted_papers_appear_in_that_block(self):
        body = ri.shortlist_usage(index([paper("A", rank=1), paper("B")]))
        self.assertIn("**A**", body)
        self.assertNotIn("**B**", body)


class TestRealness(unittest.TestCase):

    def test_real_wins_when_both_are_present(self):
        self.assertEqual(ri._realness(paper("X", realness=["real",
                                                           "synthetic"])),
                         "real")

    def test_simulated_reads_as_synthetic(self):
        self.assertEqual(ri._realness(paper("X", realness=["simulated"])),
                         "synthetic")

    def test_nothing_stated_is_unstated_not_synthetic(self):
        self.assertEqual(ri._realness(paper("X")), "unstated")


class TestChart(unittest.TestCase):

    def test_the_chart_splits_by_realness(self):
        path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "usage.svg")
        out = ri.render_chart(index([
            paper("A", datasets=["VidVRD"], realness=["real"]),
            paper("B", datasets=["CLEVR"], realness=["synthetic"])]), path)
        self.assertEqual(out, path)
        with open(path) as handle:
            svg = handle.read()
        self.assertIn("VidVRD", svg)
        self.assertIn("real", svg)

    def test_no_datasets_means_no_chart(self):
        path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "empty.svg")
        self.assertIsNone(ri.render_chart(index([paper("A")]), path))


if __name__ == "__main__":
    unittest.main()
