"""Bewertung von Mehrfachzeichen, Duplikaten und unvollständigen Läufen."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_detection import evaluate, evaluation_scope, match_detections


def box(x1=0, y1=0, x2=10, y2=10):
    return dict(x1=x1, y1=y1, x2=x2, y2=y2)


class EvaluationTests(unittest.TestCase):
    def test_duplicate_is_false_positive(self):
        result = evaluate({"a": [box(), box()]}, {"a": [box()]}, .5)
        self.assertEqual((result["true_positive"], result["false_positive"],
                          result["false_negative"]), (1, 1, 0))

    def test_multiple_ground_and_missing_image(self):
        result = evaluate({"a": [box()]}, {"a": [box(), box(20, 0, 30, 10)],
                                          "b": [box()]}, .5)
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_negative"], 2)

    def test_negative_image_false_positive(self):
        result = evaluate({"negative": [box()]}, {}, .5)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["detection_recall"], 0)

    def test_iou_threshold_is_inclusive(self):
        self.assertEqual(len(match_detections([box(0, 0, 20, 10)], [box()], .5)), 1)
        self.assertEqual(len(match_detections([box(0, 0, 21, 10)], [box()], .5)), 0)

    def test_bad_iou_and_box(self):
        for threshold in (0, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                evaluate({}, {}, threshold)
        with self.assertRaises(ValueError):
            match_detections([box(x2=0)], [box()])

    def test_partial_evaluation_requires_explicit_option(self):
        with self.assertRaises(ValueError):
            evaluation_scope({"a": []}, {}, {"a", "b"})
        self.assertEqual(evaluation_scope({"a": []}, {}, {"a", "b"}, True),
                         {"a": []})

    def test_unknown_or_empty_predictions_rejected(self):
        with self.assertRaises(ValueError):
            evaluation_scope({"typo": []}, {}, {"a"}, True)
        with self.assertRaises(ValueError):
            evaluation_scope({}, {}, {"a"}, True)


if __name__ == "__main__":
    unittest.main()
