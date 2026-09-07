"""The canvas must not restate an object's shape in terms of its source file.

`_scale_bbox_to_canvas` is the only thing that turns an annotation into the
model's position signal: the four one-hot runs in every feature vector are
`quantise`d straight off its output. Until 2026-09-07 it scaled x and y
**independently** onto the 300x200 canvas, which is an anisotropic map whenever
the source frame is not itself 3:2.

The corpus is nowhere near 3:2. Counted on this machine, 2026-09-07::

    python - <<'EOF'
    import json, glob, collections
    c = collections.Counter()
    for p in glob.glob('data/video/vidvrd/annotations/*/*.json'):
        d = json.load(open(p))
        c[(d.get('width'), d.get('height'))] += 1
    print(sum(c.values()), 'clips'); print(c.most_common())
    EOF

1000 clips, **45 distinct resolutions**, aspect ratios from 0.564 (406x720,
portrait) to 2.353 (1920x816, scope). The five commonest are 1280x720 (574),
480x360 (136), 640x360 (82), 1920x1080 (63) and 320x240 (16).

Under the old map an object's canvas width:height ratio came out as its true
ratio times `1.5 / source_aspect`. Run against the old code on 2026-09-07, a
square object came out **63x100 canvas px in a 1920x816 clip and 150x56 in a
406x720 one** -- ratios 0.63 and 2.68, a factor of 4.25 between two clips of
the same corpus, decided by nothing but which camera shot it. Every relation
FOSAE could learn about shape, containment or overlap was being taught that
identity depends on the file it came from.

These tests fix the property rather than the numbers: a square stays square,
and two clips of **different aspect** showing the same object can produce the
**same** canvas box. Neither was true before, and neither can quietly stop
being true again.
"""

import importlib.util
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, os.pardir))
sys.path.insert(0, ROOT)


def _has_pillow():
    """The loader imports PIL. Bare python3 usually has none; use .venv-local."""
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


def _loader():
    """The real `puzzle_labeled_objects`, read from file so TensorFlow stays out.

    Importing through the package runs `latplan/__init__.py`, which pulls in
    the training stack. `oracle.load_canvas_scaler` does the same dance; the
    last test here checks the two land on the same function object, because
    SPEC V5 allows the canvas geometry exactly one definition.
    """
    path = os.path.join(ROOT, "latplan", "puzzles", "puzzle_labeled_objects.py")
    spec = importlib.util.spec_from_file_location("_plo_for_aspect", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Every resolution below appears in VidVRD; the counts are from the census in
# the module docstring. The last four are the tails that make the point loudest.
RESOLUTIONS = [
    (1280, 720), (480, 360), (640, 360), (1920, 1080), (320, 240),
    (640, 480), (534, 360), (540, 360),
    (1280, 576), (270, 360), (406, 720), (1920, 816),
]


@unittest.skipUnless(_has_pillow(), "needs pillow (present in .venv-local)")
class TestShapeSurvivesTheCanvas(unittest.TestCase):

    def setUp(self):
        self.plo = _loader()
        self.scale = self.plo._scale_bbox_to_canvas
        self.W, self.H = self.plo.CANVAS_W, self.plo.CANVAS_H

    def _square(self, w, h):
        """A square source box, centred-ish, sized to fit any frame shape."""
        side = min(w, h) // 2
        x1, y1 = w // 4, h // 4
        return [x1, y1, x1 + side, y1 + side]

    def test_a_square_object_stays_square_at_every_source_resolution(self):
        """The headline defect. A square is square in the world; the canvas is
        the model's only view of the world, so it must be square there too.

        One pixel of slack, and no more: the map rounds to whole canvas pixels,
        so the two sides can land on either side of a boundary.
        """
        for w, h in RESOLUTIONS:
            x1, y1, x2, y2 = self.scale(self._square(w, h), w, h)
            self.assertLessEqual(
                abs((x2 - x1) - (y2 - y1)), 1,
                "%dx%d source: square object came out %dx%d on the canvas"
                % (w, h, x2 - x1, y2 - y1))

    def test_two_sources_of_different_aspect_give_one_canvas_box(self):
        """The same object, filmed 16:9 and 4:3, lands on the identical box.

        A 256x256 object at (256, 40) in a 1280x720 frame and a 108x108 object
        at (78, 45) in a 480x360 frame occupy the same place in the two
        pictures' shared content area. The old map sent them to (60, 11, 120,
        82) and (48, 25, 116, 85) -- different sizes, different shapes, no
        relation between them. There has to be one answer.

        The constants are not free: they are the source boxes whose letterbox
        images are whole canvas pixels in both frames, so the assertion tests
        the map and not the rounding.
        """
        a = self.scale([256, 40, 512, 296], 1280, 720)
        b = self.scale([78, 45, 186, 153], 480, 360)
        self.assertEqual(tuple(a), tuple(b))
        # And it is the square it should be, not merely a shared answer.
        self.assertEqual(a[2] - a[0], a[3] - a[1])

    def test_the_canvas_keeps_the_aspect_ratio_of_the_box(self):
        """A 2:1 object reads as 2:1 whatever shot it. Large boxes only, so
        that whole-pixel rounding cannot account for the tolerance."""
        for w, h in RESOLUTIONS:
            side = min(w, h) // 2
            box = [w // 8, h // 8, w // 8 + 2 * side, h // 8 + side]
            if box[2] > w or box[3] > h:
                continue
            x1, y1, x2, y2 = self.scale(box, w, h)
            self.assertGreater(y2 - y1, 0, "%dx%d" % (w, h))
            self.assertAlmostEqual(
                (x2 - x1) / float(y2 - y1), 2.0, delta=0.1,
                msg="%dx%d source: a 2:1 object came out %d:%d"
                    % (w, h, x2 - x1, y2 - y1))

    def test_the_whole_frame_fits_inside_the_canvas(self):
        """Contain, not cover. Filling the canvas would need a crop, and a crop
        pushes real objects off the edge where they decode as absent rather
        than as elsewhere -- silent deletion, which is the worse failure."""
        for w, h in RESOLUTIONS:
            x1, y1, x2, y2 = self.scale([0, 0, w, h], w, h)
            self.assertGreaterEqual(x1, 0, "%dx%d" % (w, h))
            self.assertGreaterEqual(y1, 0, "%dx%d" % (w, h))
            self.assertLessEqual(x2, self.W - 1, "%dx%d" % (w, h))
            self.assertLessEqual(y2, self.H - 1, "%dx%d" % (w, h))
            # One axis is filled: the fit is tight, not an arbitrary shrink.
            fills_x = x2 >= self.W - 1 and x1 == 0
            fills_y = y2 >= self.H - 1 and y1 == 0
            self.assertTrue(fills_x or fills_y,
                            "%dx%d source: frame maps to (%d,%d,%d,%d), which "
                            "touches neither pair of canvas edges"
                            % (w, h, x1, y1, x2, y2))

    def test_the_unused_margin_is_split_evenly(self):
        """Centred bars, so a frame's centre is the canvas centre. Off-centre
        padding would add a per-resolution translation on top of the scale, and
        the model would have to learn the offset before it could learn a
        relation.

        The far bar is `W - x2`, not `W - 1 - x2`: `x2` is the exclusive right
        edge of the content, and the canvas is W pixels wide. Two pixels of
        slack in the first draft of this test came from that off-by-one and not
        from the map. One pixel remains, and is real: `x2` is clamped to
        `W - 1` so a frame filling the width reports a 1-pixel right bar, and
        half-up rounding can lift both edges of a `.5` margin together.
        """
        for w, h in RESOLUTIONS:
            x1, y1, x2, y2 = self.scale([0, 0, w, h], w, h)
            self.assertLessEqual(abs(x1 - (self.W - x2)), 1,
                                 "%dx%d horizontal bars differ" % (w, h))
            self.assertLessEqual(abs(y1 - (self.H - y2)), 1,
                                 "%dx%d vertical bars differ" % (w, h))

    def test_a_box_running_off_the_frame_stays_where_it_was(self):
        """Annotations do run past the frame edge, and the result must still be
        near that edge rather than somewhere else on the canvas.

        The first draft of this test asserted `(0, 0)` for a box starting at
        (-40, -40) in a 1280x720 frame. That was the stretch map's answer, not
        a requirement: under the letterbox the frame sits inside a 15.6-pixel
        vertical bar, so 40 pixels above the frame is 6.25 pixels down the
        canvas -- representable, and clamping it to 0 would have thrown away
        information the bars exist to hold. Only x, whose bar is empty at this
        aspect, is clamped.
        """
        x1, y1, x2, y2 = self.scale([-40, -40, 1400, 800], 1280, 720)
        self.assertEqual(x1, 0)
        self.assertEqual(y1, 6)
        self.assertEqual(x2, self.W - 1)
        self.assertEqual(y2, self.H - 1)

    def test_the_oracle_shares_this_function_rather_than_copying_it(self):
        """SPEC V5: one definition of the canvas geometry. `oracle.py` exports
        planner inputs and would otherwise drift from what the loader trains
        on, which is a disagreement no test downstream of either could see."""
        from tools.planner.box_geometry import load_canvas_scaler

        scale, width, height = load_canvas_scaler()
        self.assertEqual((width, height), (self.W, self.H))
        self.assertEqual(scale([256, 40, 512, 296], 1280, 720),
                         self.scale([256, 40, 512, 296], 1280, 720))


if __name__ == "__main__":
    unittest.main()
