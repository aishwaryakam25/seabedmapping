"""Tests for pixel-based acoustic-shadow measurement."""
import unittest

import numpy as np

from shadow_physics import default_fish_altitude, measure_shadow, annotate_detections


def _frame(w=400, h=200, bg=120):
    """Uniform seabed return."""
    return np.full((h, w), bg, dtype=np.uint8)


def _add_object(img, x, y, ow, oh, bright=240):
    img[y:y + oh, x:x + ow] = bright


def _add_shadow(img, x, y, sw, sh, dark=15):
    img[y:y + sh, x:x + sw] = dark


class TestMeasureShadow(unittest.TestCase):
    def test_measures_shadow_outward_from_nadir(self):
        # Object on the starboard (right) side → shadow falls further right (outward).
        img = _frame()
        _add_object(img, 260, 90, 20, 20)
        _add_shadow(img, 280, 90, 40, 20)  # 40px shadow just past the object
        res = measure_shadow(img, [260, 90, 280, 110], range_scale=150)
        self.assertTrue(res["measured"])
        self.assertEqual(res["shadow_dir"], 1)            # rightward / outward
        self.assertAlmostEqual(res["shadow_len_px"], 40, delta=4)

    def test_shadow_falls_left_when_object_is_port_side(self):
        img = _frame()
        _add_object(img, 120, 90, 20, 20)
        _add_shadow(img, 80, 90, 40, 20)   # shadow to the LEFT (outward, port side)
        res = measure_shadow(img, [120, 90, 140, 110], range_scale=150)
        self.assertEqual(res["shadow_dir"], -1)
        self.assertAlmostEqual(res["shadow_len_px"], 40, delta=4)

    def test_longer_shadow_means_taller_object(self):
        short = _frame(); _add_object(short, 260, 90, 20, 20); _add_shadow(short, 280, 90, 20, 20)
        tall = _frame();  _add_object(tall, 260, 90, 20, 20);  _add_shadow(tall, 280, 90, 60, 20)
        h_short = measure_shadow(short, [260, 90, 280, 110], range_scale=150)["height_m"]
        h_tall = measure_shadow(tall, [260, 90, 280, 110], range_scale=150)["height_m"]
        self.assertGreater(h_tall, h_short)

    def test_height_bounded_by_fish_altitude(self):
        img = _frame()
        _add_object(img, 220, 90, 20, 20)
        _add_shadow(img, 240, 90, 150, 20)  # absurdly long shadow
        res = measure_shadow(img, [220, 90, 240, 110], range_scale=150, fish_altitude_m=10.0)
        self.assertLessEqual(res["height_m"], 10.0)       # h = H·L/(R+L) < H always

    def test_no_shadow_reads_as_buried(self):
        img = _frame()
        _add_object(img, 260, 90, 20, 20)   # bright object, NO shadow behind it
        res = measure_shadow(img, [260, 90, 280, 110], range_scale=150)
        self.assertFalse(res["measured"])
        self.assertEqual(res["burial"], "buried")
        self.assertEqual(res["height_m"], 0.0)

    def test_shadow_box_within_image_and_on_shadow_side(self):
        img = _frame()
        _add_object(img, 260, 90, 20, 20)
        _add_shadow(img, 280, 90, 30, 20)
        res = measure_shadow(img, [260, 90, 280, 110], range_scale=150)
        sb = res["shadow_box"]
        self.assertGreaterEqual(sb[0], 280 - 2)           # starts at object far edge
        self.assertLessEqual(sb[2], img.shape[1])         # stays in-frame
        self.assertEqual(sb[1], 90)
        self.assertEqual(sb[3], 110)

    def test_metres_scale_with_range(self):
        img = _frame()
        _add_object(img, 260, 90, 20, 20)
        _add_shadow(img, 280, 90, 40, 20)
        near = measure_shadow(img, [260, 90, 280, 110], range_scale=75)["shadow_len_m"]
        far = measure_shadow(img, [260, 90, 280, 110], range_scale=150)["shadow_len_m"]
        self.assertAlmostEqual(far, near * 2, delta=0.5)  # double range → double metres/px

    def test_default_altitude_scales_with_range(self):
        self.assertGreater(default_fish_altitude(300), default_fish_altitude(150))
        self.assertGreaterEqual(default_fish_altitude(10), 4.0)  # floor


class TestAnnotateDetections(unittest.TestCase):
    def test_attaches_shadow_block(self):
        img = _frame()
        _add_object(img, 260, 90, 20, 20)
        _add_shadow(img, 280, 90, 40, 20)
        dets = [{"cls": "wreck", "conf": 0.9, "box": [260, 90, 280, 110]}]
        out = annotate_detections(img, dets, range_scale=150)
        self.assertIn("shadow", out[0])
        self.assertTrue(out[0]["shadow"]["measured"])

    def test_skips_detection_without_box(self):
        dets = [{"cls": "wreck", "conf": 0.9}]
        out = annotate_detections(_frame(), dets, range_scale=150)
        self.assertNotIn("shadow", out[0])


if __name__ == "__main__":
    unittest.main()
