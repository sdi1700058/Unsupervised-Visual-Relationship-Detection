#!/usr/bin/env python3
"""A claim's wording is derived and rendered, never typed by hand.

This is the part of `DESIGN_WORKPLAN.md` that was designed and then not built.
The plan recorded `appears_in` for every claim and nothing read it, so the
licence system computed a tier and had no way to make a document obey it. The
documents were being reworded by hand, which is exactly the method the design
replaced: a sentence and its evidence could drift apart again with nothing to
notice.

So the sentence is generated into the document between markers, the way the
literature cross-references are, and a check fails when a listed document does
not carry it.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "tools"))

import workplan  # noqa: E402


def obs(oid="O1", n=22, datasets=("vidvrd",), paired=True,
        what="the oracle beat the baseline on 12 of 22 clips",
        source="eval/planner/x/summary.csv"):
    return {"id": oid, "n": n, "datasets": list(datasets), "paired": paired,
            "what": what, "source": source, "supports": True}


def claim(cid="C1", asserts="The task is winnable on real video",
          claim_type="existential", evidence=("O1",), appears_in=()):
    return {"id": cid, "asserts": asserts, "claim_type": claim_type,
            "evidence": list(evidence), "appears_in": list(appears_in),
            "rests_on": [], "approved_by": None}


def plan(claims, observations):
    return {"units": [], "assumptions": [], "claims": claims,
            "questions": [], "availability": [],
            "observations": observations, "decisions": {}}


class TestDerivedSentence(unittest.TestCase):

    def test_a_scoped_sentence_names_the_sample_and_the_corpus(self):
        p = plan([claim()], [obs()])
        text = workplan.claim_sentence(p["claims"][0], p)
        self.assertIn("22", text)
        self.assertIn("vidvrd", text.lower())
        self.assertIn("12 of 22", text)

    def test_a_scoped_sentence_does_not_generalise(self):
        """0.46 licenses scoped only, so the sentence must not assert more."""
        p = plan([claim()], [obs()])
        text = workplan.claim_sentence(p["claims"][0], p).lower()
        for forbidden in ("therefore", "in general", "always", "proves"):
            self.assertNotIn(forbidden, text)

    def test_the_tier_is_named_in_the_sentence(self):
        p = plan([claim()], [obs()])
        self.assertIn("scoped",
                      workplan.claim_sentence(p["claims"][0], p).lower())

    def test_more_datasets_license_a_stronger_sentence(self):
        strong = obs(n=30, datasets=("vidvrd", "vidor", "actiongenome"))
        granted = claim(claim_type="universal")
        granted["approved_by"] = "author"
        p = plan([granted], [strong])
        text = workplan.claim_sentence(p["claims"][0], p)
        self.assertIn("universal", text.lower())
        self.assertIn("3 datasets", text)

    def test_a_claim_with_no_evidence_says_so_rather_than_inventing(self):
        p = plan([claim(evidence=())], [])
        text = workplan.claim_sentence(p["claims"][0], p)
        self.assertIn("no evidence", text.lower())


class TestRenderIntoDocuments(unittest.TestCase):

    def setUp(self):
        self.tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "claims")
        if not os.path.isdir(self.tmp):
            os.makedirs(self.tmp)

    def doc(self, name, body):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as handle:
            handle.write(body)
        return path

    def test_the_sentence_lands_between_its_markers(self):
        path = self.doc("d1.md", "before\n\n<!-- claim: C1 -->\nold\n"
                                 "<!-- /claim -->\n\nafter\n")
        p = plan([claim(appears_in=[path])], [obs()])
        written = workplan.render_claims(p)
        self.assertIn(path, written)
        with open(path) as handle:
            text = handle.read()
        self.assertIn("12 of 22", text)
        self.assertNotIn("old", text)
        self.assertIn("before", text)
        self.assertIn("after", text)

    def test_rendering_twice_changes_nothing(self):
        path = self.doc("d2.md", "<!-- claim: C1 -->\n<!-- /claim -->\n")
        p = plan([claim(appears_in=[path])], [obs()])
        workplan.render_claims(p)
        with open(path) as handle:
            once = handle.read()
        workplan.render_claims(p)
        with open(path) as handle:
            twice = handle.read()
        self.assertEqual(once, twice)

    def test_a_document_without_the_marker_is_reported_not_mangled(self):
        path = self.doc("d3.md", "no marker here\n")
        p = plan([claim(appears_in=[path])], [obs()])
        problems = workplan.check_claims(p)
        self.assertTrue(any("C1" in x and path in x for x in problems))
        with open(path) as handle:
            self.assertEqual(handle.read(), "no marker here\n")


class TestCheckCatchesDrift(unittest.TestCase):

    def setUp(self):
        self.tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "claims")
        if not os.path.isdir(self.tmp):
            os.makedirs(self.tmp)

    def doc(self, name, body):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as handle:
            handle.write(body)
        return path

    def test_a_stale_rendered_block_fails(self):
        """The document says one thing and the evidence now says another."""
        path = self.doc("d4.md", "<!-- claim: C1 -->\nsomething else entirely\n"
                                 "<!-- /claim -->\n")
        p = plan([claim(appears_in=[path])], [obs()])
        self.assertTrue(any("C1" in x for x in workplan.check_claims(p)))

    def test_a_current_block_passes(self):
        path = self.doc("d5.md", "<!-- claim: C1 -->\n<!-- /claim -->\n")
        p = plan([claim(appears_in=[path])], [obs()])
        workplan.render_claims(p)
        self.assertEqual(workplan.check_claims(p), [])

    def test_a_claim_listing_no_document_is_flagged(self):
        """A claim nobody states is a claim nobody can check."""
        p = plan([claim(appears_in=[])], [obs()])
        self.assertTrue(any("no document" in x for x in workplan.check_claims(p)))


if __name__ == "__main__":
    unittest.main()


class TestClaimStrengthCombinesCorpora(unittest.TestCase):
    """A second corpus must raise a claim, or the red card is not enforced.

    The first implementation scored a claim by its single strongest
    observation, so evidence from a second dataset changed nothing and the
    independence multiplier — the whole arithmetic expression of "one dataset is
    not enough" — could never be escaped. Registering VidOR alongside VidVRD
    left C1 reading `datasets=1`, which is how this surfaced.
    """

    def test_two_corpora_beat_one(self):
        one = plan([claim(evidence=["O1"])], [obs("O1", n=22)])
        two = plan([claim(evidence=["O1", "O2"])],
                   [obs("O1", n=22), obs("O2", n=24, datasets=("vidor",))])
        self.assertGreater(workplan.claim_strength(two["claims"][0], two),
                           workplan.claim_strength(one["claims"][0], one))

    def test_a_second_run_on_the_same_corpus_does_not(self):
        one = plan([claim(evidence=["O1"])], [obs("O1", n=22)])
        again = plan([claim(evidence=["O1", "O2"])],
                     [obs("O1", n=22), obs("O2", n=22)])
        self.assertAlmostEqual(workplan.claim_strength(again["claims"][0], again),
                               workplan.claim_strength(one["claims"][0], one),
                               places=6)

    def test_two_corpora_can_license_a_stronger_tier(self):
        granted = claim(claim_type="comparative", evidence=["O1", "O2"])
        granted["approved_by"] = "author"
        two = plan([granted],
                   [obs("O1", n=22), obs("O2", n=24, datasets=("vidor",))])
        e = workplan.claim_strength(two["claims"][0], two)
        self.assertGreaterEqual(e, 0.6)
        self.assertEqual(workplan.wording_tier(two["claims"][0], e),
                         "comparative")

    def test_the_sentence_names_every_corpus(self):
        two = plan([claim(evidence=["O1", "O2"])],
                   [obs("O1", n=22), obs("O2", n=24, datasets=("vidor",))])
        text = workplan.claim_sentence(two["claims"][0], two)
        self.assertIn("vidor", text.lower())
        self.assertIn("vidvrd", text.lower())
        self.assertIn("2 datasets", text)

    def test_no_evidence_is_zero_and_not_an_error(self):
        empty = plan([claim(evidence=[])], [])
        self.assertEqual(workplan.claim_strength(empty["claims"][0], empty), 0.0)


class TestApprovalGate(unittest.TestCase):
    """Evidence licenses a tier. Only the author grants it.

    `DESIGN_WORKPLAN.md` 4.4: any tier above `scoped` needs explicit sign-off.
    Without that, enough evidence would silently promote a sentence in a
    document the author had not read — which is the same failure as declaring a
    task complete, one level down.
    """

    def test_strong_evidence_alone_does_not_promote(self):
        c = claim(claim_type="existential", evidence=["O1", "O2"])
        p = plan([c], [obs("O1", n=22), obs("O2", n=24, datasets=("vidor",))])
        e = workplan.claim_strength(c, p)
        self.assertGreaterEqual(e, 0.5)
        self.assertEqual(workplan.wording_tier(c, e), "scoped")

    def test_sign_off_grants_the_licensed_tier(self):
        c = claim(claim_type="existential", evidence=["O1", "O2"])
        c["approved_by"] = "author"
        p = plan([c], [obs("O1", n=22), obs("O2", n=24, datasets=("vidor",))])
        self.assertEqual(workplan.wording_tier(c, workplan.claim_strength(c, p)),
                         "existential")

    def test_sign_off_cannot_exceed_the_evidence(self):
        """Approval is permission, not evidence."""
        c = claim(claim_type="universal", evidence=["O1"])
        c["approved_by"] = "author"
        p = plan([c], [obs("O1", n=22)])
        self.assertEqual(workplan.wording_tier(c, workplan.claim_strength(c, p)),
                         "scoped")

    def test_a_claim_awaiting_sign_off_is_reported(self):
        c = claim(claim_type="existential", evidence=["O1", "O2"],
                  appears_in=[])
        p = plan([c], [obs("O1", n=22), obs("O2", n=24, datasets=("vidor",))])
        self.assertTrue(any("awaits" in x for x in workplan.check_claims(p)))
