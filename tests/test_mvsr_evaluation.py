"""Hand-calculated system metrics, including background, rejection and misses."""
import unittest

from src.mvsr_evaluation import evaluate_system


def box(c, x=0):
    return {"x1": x, "y1": 0, "x2": x + 10, "y2": 10, "class_id": c}


class SystemEvaluationTests(unittest.TestCase):
    def test_wrong_class_rejection_miss_duplicate_and_negative_image(self):
        predictions = {
            "street": [box(1), box(3, 20), box(-1, 40), box(1)],
            "negative": [box(2), box(-1, 20)],
        }
        labels = {"street": [box(1), box(2, 20), box(4, 40), box(5, 60)], "negative": []}
        result = evaluate_system(predictions, labels)
        detection = result["detection"]
        self.assertEqual((detection["true_positive"], detection["false_positive"],
                          detection["false_negative"]), (3, 3, 1))
        self.assertAlmostEqual(result["classification_on_matched_candidates"]["accuracy_including_unknown"], 1/3)
        joint = result["end_to_end"]
        self.assertEqual((joint["TP"], joint["FP"], joint["FN"]), (1, 3, 3))
        self.assertEqual((joint["precision"], joint["recall"]), (.25, .25))
        cm = result["confusion"]["counts"]
        self.assertEqual((cm[1][1], cm[2][3], cm[4][43], cm[5][44]), (1, 1, 1, 1))
        self.assertEqual((cm[43][1], cm[43][2], cm[43][43]), (1, 1, 1))
        self.assertEqual(sum(map(sum, cm)), 7)
        self.assertEqual(result["per_class"]["1"]["precision"], .5)
        self.assertEqual(result["per_class"]["2"]["FP"], 1)
        self.assertEqual(result["per_class"]["2"]["FN"], 1)
        self.assertEqual(len(result["per_class"]), 43)

    def test_absent_class_and_no_predictions_are_not_fabricated_accuracy(self):
        result = evaluate_system({"a": []}, {"a": [box(42)]})
        self.assertIsNone(result["classification_on_matched_candidates"]["accuracy_including_unknown"])
        self.assertIsNone(result["end_to_end"]["precision"])
        self.assertEqual(result["end_to_end"]["recall"], 0)
        self.assertIsNone(result["per_class"]["0"]["recall"])
        self.assertEqual(result["confusion"]["counts"][42][44], 1)

    def test_partial_or_extra_images_rejected(self):
        for predictions, labels in (({}, {}), ({"a": []}, {"a": [], "b": []}),
                                     ({"a": [], "b": []}, {"a": []})):
            with self.assertRaises(ValueError):
                evaluate_system(predictions, labels)

    def test_invalid_class_and_iou_rejected(self):
        for bad in (43, None, 1.5, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                evaluate_system({"a": [box(bad)]}, {"a": [box(1)]})
        with self.assertRaises(ValueError):
            evaluate_system({"a": []}, {"a": [box(-1)]})
        with self.assertRaises(ValueError):
            evaluate_system({"a": []}, {"a": []}, 0)
