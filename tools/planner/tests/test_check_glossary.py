"""The glossary gate. A term used in a special sense is defined once.

Two terms entered this project with no definition. "At window" appeared in a
report before it meant anything to a reader, and "corpus" was introduced as a
synonym for dataset and never declared.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, os.pardir))

from tools import check_glossary


class TestTerms(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "GLOSSARY.md")
        with open(self.path, "w") as handle:
            handle.write("# Glossary\n\n| term | definition |\n|---|---|\n")
            handle.write("| **dataset** | a published collection |\n")
            handle.write("| **window** | k consecutive frames |\n")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_it_reads_every_term(self):
        found = check_glossary.terms(self.path)
        self.assertIn("dataset", found)
        self.assertIn("window", found)
        self.assertEqual(found["dataset"], "a published collection")

    def test_the_header_row_is_not_a_term(self):
        self.assertNotIn("term", check_glossary.terms(self.path))

    def test_a_missing_glossary_gives_no_terms_rather_than_raising(self):
        self.assertEqual(check_glossary.terms(os.path.join(self.dir, "no.md")),
                         {})

    def ban_corpus(self):
        """Append the refusal table the real glossary carries."""
        with open(self.path, "a") as handle:
            handle.write("\n## Terms deliberately not used\n\n")
            handle.write("| avoid | use instead | why |\n|---|---|---|\n")
            handle.write("| corpus | dataset | it arrived undefined |\n")

    def test_the_refusal_table_is_not_read_as_definitions(self):
        """The bug that made this check pass vacuously on its first real run.

        The glossary lists the words this project refuses in a table of their
        own, in order to refuse them. Reading those rows as definitions
        registered `corpus` as a defined term.

        Every fixture here stopped at the definition table, so removing the
        heading guard from `terms` left all eleven tests in this file green.
        """
        self.ban_corpus()
        found = check_glossary.terms(self.path)
        self.assertIn("dataset", found)
        self.assertNotIn("corpus", found)

    def test_the_refusal_table_is_found_even_when_its_heading_is_renamed(self):
        """The heading was the single point of failure.

        `STOP_HEADING` matches one phrase. Rewording that heading -- an
        ordinary edit, on a document outside version control, so it leaves no
        diff -- put the refusal rows back into the definition set and
        readmitted every word the table exists to ban. The table's own column
        names are structure rather than prose, so they are checked too.
        """
        with open(self.path, "a") as handle:
            handle.write("\n## Words we replaced\n\n")
            handle.write("| avoid | use instead | why |\n|---|---|---|\n")
            handle.write("| corpus | dataset | it arrived undefined |\n")
        found = check_glossary.terms(self.path)
        self.assertIn("dataset", found)
        self.assertNotIn("corpus", found)
        doc = os.path.join(self.dir, "b.md")
        with open(doc, "w") as handle:
            handle.write("The corpus loads cleanly.\n")
        hits = check_glossary.undefined_terms([doc], found)
        self.assertEqual([h["term"] for h in hits], ["corpus"])

    def test_a_word_the_refusal_table_names_is_still_caught_in_a_document(self):
        """The consequence, stated where a reader will see the cost.

        With the refusal row read as a definition, every use of the word in
        every document was skipped and a planted violation went undetected.
        """
        self.ban_corpus()
        doc = os.path.join(self.dir, "a.md")
        with open(doc, "w") as handle:
            handle.write("The corpus loads cleanly.\n")
        hits = check_glossary.undefined_terms(
            [doc], check_glossary.terms(self.path))
        self.assertEqual([h["term"] for h in hits], ["corpus"])


class TestUndefined(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w") as handle:
            handle.write(text)
        return path

    def test_a_banned_word_is_reported(self):
        """The case that happened: corpus, used and never declared."""
        doc = self.write("a.md", "The corpus loads cleanly.\n")
        hits = check_glossary.undefined_terms([doc], {"dataset": "x"})
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["term"], "corpus")
        self.assertEqual(hits[0]["line"], 1)

    def test_the_replacement_is_named_in_the_hit(self):
        doc = self.write("a.md", "The corpus loads cleanly.\n")
        hits = check_glossary.undefined_terms([doc], {"dataset": "x"})
        self.assertEqual(hits[0]["use_instead"], "dataset")

    def test_an_approved_word_passes(self):
        doc = self.write("a.md", "The dataset loads cleanly.\n")
        self.assertEqual(
            check_glossary.undefined_terms([doc], {"dataset": "x"}), [])

    def test_the_glossary_itself_is_not_flagged(self):
        """A glossary naming a banned word is not a document using it."""
        doc = self.write("GLOSSARY.md", "| corpus | use dataset |\n")
        self.assertEqual(
            check_glossary.undefined_terms([doc], {"dataset": "x"}), [])

    def test_a_word_inside_a_longer_word_is_not_a_hit(self):
        """`corpora` is the plural and is caught; `incorporate` is not."""
        doc = self.write("a.md", "We incorporate the boxes.\n")
        self.assertEqual(
            check_glossary.undefined_terms([doc], {"dataset": "x"}), [])

    def test_the_plural_is_caught_too(self):
        doc = self.write("a.md", "Both corpora load.\n")
        hits = check_glossary.undefined_terms([doc], {"dataset": "x"})
        self.assertEqual(len(hits), 1)

    def test_a_line_quoting_the_rule_is_exempt(self):
        """A document explaining the ban is not a document breaking it."""
        doc = self.write("a.md",
                         "Do not write corpus; the word is banned here.\n")
        self.assertEqual(
            check_glossary.undefined_terms([doc], {"dataset": "x"}), [])


class TestMain(unittest.TestCase):

    def test_a_missing_glossary_fails_rather_than_passing_vacuously(self):
        """A check that cannot find its input must not report success."""
        code = check_glossary.main(["--glossary", "no/such/file.md"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
