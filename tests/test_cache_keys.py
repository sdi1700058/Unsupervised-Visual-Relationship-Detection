"""A cache key must name everything that changes the shape of what it holds.

`model/loaders/cache.py` says so in its own docstring: "A cache keyed on
category and fps alone returns patch-32 tensors to a caller that asked for
patch 8, silently and with no shape error, because the model reads the patch
dim off the data."

`model/loaders/vidvrd.py` follows that. `actiongenome.py` did not: it keyed on
category and fps while `build_dataset` also took `num_objs` and `patch_size`.
Re-running Action Genome at a different patch size returned the previous
tensors, with no error anywhere, and the model would have read the wrong patch
dimension off them.

The loaders are not imported through the package here, because
`model/loaders/__init__.py` imports `mnist`, which pulls in Keras and
TensorFlow through `base`. They are loaded from file with the pieces they need
stubbed, the same approach `metrics.box_geometry.load_canvas_scaler` takes and
for the same reason.
"""

import importlib.util
import os
import sys
import types
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    os.pardir)
ROOT = os.path.abspath(ROOT)


# Everything `_stub_model` puts into `sys.modules`. The stubs have to come
# back out after every test: `load_canvas_scaler` tries the package
# import first and falls back to loading the file only on ImportError, so a
# stub left behind makes that import succeed and hands the real oracle tests a
# `_scale_bbox_to_canvas` that returns None. That is what happened when this
# file was first written -- eleven tests in three other files went red.
_STUBBED = ("model", "model.loaders", "model.paths",
            "model.loaders.labeled_objects", "model.loaders.cache")


class StubbedCase(unittest.TestCase):
    """Installs the stubs for one test and takes them out again."""

    def setUp(self):
        self._saved = dict((k, sys.modules.get(k)) for k in _STUBBED)
        self.cache = _stub_model()

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


def _stub_model():
    """The minimum of the `model` package that the loaders touch, without TensorFlow."""
    for name in ("model", "model.loaders"):
        sys.modules.setdefault(name, types.ModuleType(name))

    paths = types.ModuleType("model.paths")
    paths.DATA_DIR = os.path.join(ROOT, "data")
    sys.modules["model.paths"] = paths

    plo = types.ModuleType("model.loaders.labeled_objects")
    plo.PATCH_SIZE = 32
    plo.MAX_OBJECTS = 10
    plo.CANVAS_H, plo.CANVAS_W = 200, 300
    plo.PICSIZE = (200, 300)
    plo._crop_object = lambda *a, **k: None
    plo._scale_bbox_to_canvas = lambda *a, **k: None
    sys.modules["model.loaders.labeled_objects"] = plo

    return _from_file("model.loaders.cache",
                      os.path.join(ROOT, "model", "loaders", "cache.py"))


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


if __name__ == "__main__":
    unittest.main()
