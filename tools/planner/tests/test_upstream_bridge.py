"""Where upstream latplan lives is a property of the machine, not the source.

`UPSTREAM_LATPLAN` was the literal `/home/panoslat/Dev/Thesis/FOSAE/latplan`
with no override, so every ama3 run on Sherlock failed on a path that exists
only on this workstation. `install_roswell.sh`, in the same directory, already
read `${UPSTREAM_LATPLAN:-...}` -- so two files that must agree about one
directory disagreed about whether it could be configured at all.
"""

import importlib
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, os.pardir))
sys.path.insert(0, ROOT)

from tools.planner.ama3 import upstream_bridge


class TestWhereUpstreamIs(unittest.TestCase):

    def setUp(self):
        self._was = os.environ.get("UPSTREAM_LATPLAN")

    def tearDown(self):
        if self._was is None:
            os.environ.pop("UPSTREAM_LATPLAN", None)
        else:
            os.environ["UPSTREAM_LATPLAN"] = self._was
        importlib.reload(upstream_bridge)

    def _reload_with(self, value):
        if value is None:
            os.environ.pop("UPSTREAM_LATPLAN", None)
        else:
            os.environ["UPSTREAM_LATPLAN"] = value
        return importlib.reload(upstream_bridge)

    def test_the_environment_decides(self):
        """The fix, stated as the failure it removes: ama3 on the cluster."""
        mod = self._reload_with("/scratch/somewhere/latplan")
        self.assertEqual(mod.UPSTREAM_LATPLAN, "/scratch/somewhere/latplan")

    def test_upstream_dir_follows_the_override(self):
        mod = self._reload_with("/scratch/somewhere/latplan")
        self.assertEqual(mod.upstream_dir("lisp"),
                         os.path.join("/scratch/somewhere/latplan", "lisp"))

    def test_the_path_insert_follows_the_override(self):
        mod = self._reload_with("/scratch/somewhere/latplan")
        try:
            mod.ensure_upstream_on_path()
            self.assertIn("/scratch/somewhere/latplan", sys.path)
        finally:
            while "/scratch/somewhere/latplan" in sys.path:
                sys.path.remove("/scratch/somewhere/latplan")

    def test_without_the_variable_it_looks_beside_this_repository(self):
        """A relative default, so a fresh clone anywhere still resolves.

        The two repositories are siblings on this workstation and on the
        cluster. Deriving the default from this file's own location makes that
        arrangement work without anyone setting a variable, where the absolute
        literal worked on exactly one machine.
        """
        mod = self._reload_with(None)
        self.assertTrue(mod.UPSTREAM_LATPLAN.endswith("latplan"))
        self.assertEqual(os.path.dirname(mod.UPSTREAM_LATPLAN),
                         os.path.dirname(ROOT))

    def test_no_home_directory_is_written_into_the_source(self):
        with open(upstream_bridge.__file__, encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn("/home/panoslat", source)


if __name__ == "__main__":
    unittest.main()
