"""A cache key must name everything that changes the shape of what it holds.

`latplan/util/cache.py` says so in its own docstring: "A cache keyed on
category and fps alone returns patch-32 tensors to a caller that asked for
patch 8, silently and with no shape error, because the model reads the patch
dim off the data."

`puzzle_vidvrd.py` follows that. `actiongenome.py` did not: it keyed on
category and fps while `build_dataset` also took `num_objs` and `patch_size`.
Re-running Action Genome at a different patch size returned the previous
tensors, with no error anywhere, and the model would have read the wrong patch
dimension off them.

The loaders cannot be imported normally here, because `latplan/__init__.py`
pulls in TensorFlow and this environment has none. They are loaded from file
with the pieces they need stubbed, the same approach `oracle.load_canvas_scaler`
takes and for the same reason.
"""

import importlib.util
import os
import sys
import types
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    os.pardir, os.pardir, os.pardir)
ROOT = os.path.abspath(ROOT)


# Everything `_stub_latplan` puts into `sys.modules`. The stubs have to come
# back out after every test: `oracle.load_canvas_scaler` tries the package
# import first and falls back to loading the file only on ImportError, so a
# stub left behind makes that import succeed and hands the real oracle tests a
# `_scale_bbox_to_canvas` that returns None. That is what happened when this
# file was first written -- eleven tests in three other files went red.
_STUBBED = ("latplan", "latplan.util", "latplan.puzzles", "latplan.domains",
            "latplan.domains.video", "latplan.util.paths",
            "latplan.puzzles.puzzle_labeled_objects", "latplan.util.cache",
            "latplan.domains.video.actiongenome")


class StubbedCase(unittest.TestCase):
    """Installs the stubs for one test and takes them out again."""

    def setUp(self):
        self._saved = dict((k, sys.modules.get(k)) for k in _STUBBED)
        self.cache = _stub_latplan()

    def tearDown(self):
        for name, was in self._saved.items():
            if was is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = was


def _from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stub_latplan():
    """The minimum of `latplan` these two loaders touch, without TensorFlow."""
    for name in ("latplan", "latplan.util", "latplan.puzzles",
                 "latplan.domains", "latplan.domains.video"):
        sys.modules.setdefault(name, types.ModuleType(name))

    paths = types.ModuleType("latplan.util.paths")
    paths.DATA_DIR = os.path.join(ROOT, "data")
    sys.modules["latplan.util.paths"] = paths

    plo = types.ModuleType("latplan.puzzles.puzzle_labeled_objects")
    plo.PATCH_SIZE = 32
    plo.MAX_OBJECTS = 10
    plo.CANVAS_H, plo.CANVAS_W = 200, 300
    plo.PICSIZE = (200, 300)
    plo._crop_object = lambda *a, **k: None
    plo._scale_bbox_to_canvas = lambda *a, **k: None
    sys.modules["latplan.puzzles.puzzle_labeled_objects"] = plo

    return _from_file("latplan.util.cache",
                      os.path.join(ROOT, "latplan", "util", "cache.py"))


class TestTheKeyItself(StubbedCase):
    """`npz_cache_path` was always correct. It is the callers that differed."""

    def test_two_patch_sizes_are_two_files(self):
        a = self.cache.npz_cache_path("video", "d", "cat", 30, patch_size=8)
        b = self.cache.npz_cache_path("video", "d", "cat", 30, patch_size=32)
        self.assertNotEqual(a, b)

    def test_two_object_counts_are_two_files(self):
        a = self.cache.npz_cache_path("video", "d", "cat", 30, num_objs=3)
        b = self.cache.npz_cache_path("video", "d", "cat", 30, num_objs=5)
        self.assertNotEqual(a, b)

    def test_omitting_them_keeps_the_old_short_name(self):
        """Stated in the docstring, so a pre-existing cache is still found."""
        self.assertTrue(self.cache.npz_cache_path(
            "video", "d", "cat", 30).endswith("cat-30fps.npz"))


class TestWhatTheLoadersAskFor(StubbedCase):
    """The defect was here: one loader passed them, the other did not."""

    def setUp(self):
        StubbedCase.setUp(self)
        self.asked = []

    def _watch(self, module):
        """Record the key each loader composes, without touching the disk."""
        real = module.npz_cache_path

        def spy(*args, **kwargs):
            self.asked.append(kwargs)
            return real(*args, **kwargs)

        module.npz_cache_path = spy

    def _build(self, module, **kwargs):
        """Call `build_dataset` far enough to compose the key.

        The annotations do not exist, so it raises or returns empty once past
        the cache block. Either is fine: the key has been composed by then.
        """
        self.asked = []
        try:
            module.build_dataset(annotations_dir=os.path.join(ROOT, "nope"),
                                 frames_dir=os.path.join(ROOT, "nope"),
                                 category_filter="cat", **kwargs)
        except Exception:
            pass
        return self.asked

    def _actiongenome(self):
        return _from_file(
            "latplan.domains.video.actiongenome",
            os.path.join(ROOT, "latplan", "domains", "video",
                         "actiongenome.py"))

    def test_action_genome_names_the_patch_size_and_the_object_count(self):
        """The bug, stated as the behaviour it broke.

        Without these in the key, a bake at patch 8 reads back the patch-32
        arrays written by an earlier bake.
        """
        ag = self._actiongenome()
        self._watch(ag)
        asked = self._build(ag, patch_size=8, num_objs=3)
        self.assertTrue(asked, "build_dataset never composed a cache key")
        self.assertEqual(asked[0].get("patch_size"), 8)
        self.assertEqual(asked[0].get("num_objs"), 3)

    def test_action_genome_gets_a_different_file_for_a_different_patch(self):
        ag = self._actiongenome()
        self._watch(ag)
        small = self._build(ag, patch_size=8, num_objs=3)
        large = self._build(ag, patch_size=32, num_objs=3)
        cache = sys.modules["latplan.util.cache"]
        self.assertNotEqual(
            cache.npz_cache_path("video", "actiongenome", "cat", "native",
                                 **small[0]),
            cache.npz_cache_path("video", "actiongenome", "cat", "native",
                                 **large[0]))


if __name__ == "__main__":
    unittest.main()
