"""Detektorvertrag und synthetische Störfälle; ohne externe Testpakete."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mvsr_kl.localizer import (  # noqa: E402
    Detection, _suppress_overlaps, draw_detections, find_sign_candidates,
    intersection_over_union,
)
from mvsr_kl.preprocess import segment_colors  # noqa: E402


def circle_scene(center=(100, 90), radius=24, color=(0, 0, 220)):
    image = np.full((200, 280, 3), 160, np.uint8)
    cv2.circle(image, center, radius, color, -1)
    cv2.circle(image, center, radius - max(3, radius // 4), (240, 240, 240), -1)
    x, y = center
    return image, (x - radius, y - radius, x + radius + 1, y + radius + 1)


class DetectorContractTests(unittest.TestCase):
    def assert_found(self, image, box, **kwargs):
        detections = find_sign_candidates(image, **kwargs)
        self.assertTrue(any(intersection_over_union(
            (d.x1, d.y1, d.x2, d.y2), box) >= .5 for d in detections),
            msg=f"Box {box} not found: {detections}")
        return detections

    def test_empty_inputs(self):
        for image in (None, np.empty((0, 0, 3), np.uint8)):
            with self.subTest(image=image), self.assertRaises(ValueError):
                find_sign_candidates(image)

    def test_invalid_image_contract(self):
        for image in (np.zeros((50, 50), np.uint8),
                      np.zeros((50, 50, 4), np.uint8),
                      np.zeros((50, 50, 3), np.float32)):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                find_sign_candidates(image)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            find_sign_candidates(np.zeros((30, 30, 3), np.uint8), "invalid")

    def test_small_images(self):
        for size in (1, 2, 8):
            self.assertEqual(find_sign_candidates(
                np.full((size, size, 3), 100, np.uint8)), [])

    def test_red_circle_across_positions_and_sizes(self):
        for center, radius in (((30, 30), 10), ((130, 50), 17), ((210, 130), 30)):
            with self.subTest(center=center, radius=radius):
                self.assert_found(*circle_scene(center, radius))

    def test_dark_and_blurred_circle(self):
        image, box = circle_scene()
        self.assert_found((image.astype(np.float32) * .3).astype(np.uint8), box)
        self.assert_found(cv2.GaussianBlur(image, (5, 5), 1), box)

    def test_multiple_signs(self):
        image, first = circle_scene((55, 80), 20)
        other, second = circle_scene((200, 110), 27, (210, 0, 0))
        cv2.circle(other, (200, 110), 27, (210, 0, 0), -1)
        cv2.arrowedLine(other, (200, 127), (200, 93), (245, 245, 245), 5)
        image[:, 140:] = other[:, 140:]
        detections = self.assert_found(image, first)
        self.assert_found(image, second)
        self.assertEqual(len(detections), 2)

    def test_no_entry_split_color_regions(self):
        image = np.full((160, 200, 3), 100, np.uint8)
        cv2.circle(image, (70, 70), 10, (20, 20, 230), -1)
        cv2.rectangle(image, (61, 68), (79, 72), (245, 245, 245), -1)
        self.assert_found(image, (60, 60, 81, 81))

    def test_blue_rectangle(self):
        image = np.full((180, 220, 3), 120, np.uint8)
        cv2.rectangle(image, (70, 50), (110, 100), (220, 40, 20), -1)
        cv2.arrowedLine(image, (90, 90), (90, 60), (245, 245, 245), 5)
        self.assert_found(image, (70, 50, 111, 101))

    def test_triangle(self):
        image = np.full((180, 220, 3), 120, np.uint8)
        cv2.fillPoly(image, [np.array([(90, 45), (60, 100), (120, 100)])],
                     (20, 20, 220))
        cv2.fillPoly(image, [np.array([(90, 59), (72, 93), (108, 93)])],
                     (240, 240, 240))
        self.assert_found(image, (60, 45, 121, 101))

    def test_yellow_priority_diamond(self):
        image = np.full((180, 220, 3), 90, np.uint8)
        cv2.fillPoly(image, [np.array([(100, 40), (130, 70),
                                      (100, 100), (70, 70)])], (245, 245, 245))
        cv2.fillPoly(image, [np.array([(100, 53), (117, 70),
                                      (100, 87), (83, 70)])], (0, 180, 230))
        self.assert_found(image, (70, 40, 131, 101))

    def test_octagonal_sign(self):
        image = np.full((180, 220, 3), 100, np.uint8)
        points = np.array([(78, 50), (102, 50), (120, 68), (120, 92),
                           (102, 110), (78, 110), (60, 92), (60, 68)])
        cv2.fillPoly(image, [points], (0, 0, 210))
        cv2.putText(image, "STOP", (65, 86), 0, .55, (245, 245, 245), 2)
        self.assert_found(image, (60, 50, 121, 111))

    def test_short_occlusion_in_border(self):
        image, box = circle_scene()
        image[88:92, 118:129] = 160
        self.assert_found(image, box)

    def test_solid_red_distractor(self):
        image = np.full((180, 220, 3), 100, np.uint8)
        cv2.circle(image, (90, 90), 24, (0, 0, 255), -1)
        self.assertEqual(find_sign_candidates(image), [])

    def test_saturated_tail_light(self):
        image = np.full((180, 220, 3), 50, np.uint8)
        cv2.circle(image, (90, 90), 24, (0, 0, 255), -1)
        cv2.circle(image, (90, 90), 15, (0, 230, 255), -1)
        self.assertEqual(find_sign_candidates(image), [])

    def test_no_mutation_and_serializable_contract(self):
        image, box = circle_scene()
        original = image.copy()
        detections = self.assert_found(image, box)
        for d in detections:
            self.assertIsNone(d.class_id)
            self.assertIsInstance(d.x1, int)
            self.assertTrue(0 <= d.confidence <= 1)
            self.assertTrue(0 <= d.x1 < d.x2 <= image.shape[1])
            self.assertTrue(0 <= d.y1 < d.y2 <= image.shape[0])
            self.assertEqual(replace(d, class_id=14).class_id, 14)
            self.assertEqual(set(d.to_dict()), {
                "x1", "y1", "x2", "y2", "color", "shape", "confidence", "class_id"})
        drawn = draw_detections(image, detections)
        self.assertTrue(np.array_equal(image, original))
        self.assertFalse(np.array_equal(image, drawn))

    def test_nms_duplicate_and_neighbor(self):
        a = Detection(0, 0, 20, 20, "red", "circle", .9)
        duplicate = replace(a, x1=1, confidence=.7, color="blue")
        neighbor = replace(a, x1=15, x2=35, confidence=.8)
        self.assertEqual(_suppress_overlaps([duplicate, neighbor, a]), [a, neighbor])

    def test_iou_partial_and_degenerate(self):
        self.assertAlmostEqual(intersection_over_union(
            (0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)
        self.assertEqual(intersection_over_union((0, 0, 0, 10), (0, 0, 10, 10)), 0)

    def test_hue_wrap_and_dark_noise(self):
        hsv = np.array([[[0, 200, 160], [179, 200, 160],
                         [160, 130, 60], [0, 255, 8]]], np.uint8)
        masks = segment_colors(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        self.assertTrue(np.all(masks["red"][0, :3] == 255))
        self.assertEqual(int(masks["red"][0, 3]), 0)


if __name__ == "__main__":
    unittest.main()
