"""Integration contracts exercised with real detection and real template fitting."""
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.classification import TemplateMatchingClassifier
from src.detection import detect_signs
from src.mvsr_kl.localizer import Detection
from src.pipeline import detect_and_classify, validate_classifier


def two_signs():
    image = np.full((200, 280, 3), 160, np.uint8)
    cv2.circle(image, (55, 80), 20, (0, 0, 220), -1)
    cv2.circle(image, (55, 80), 15, (240, 240, 240), -1)
    cv2.circle(image, (200, 110), 27, (210, 0, 0), -1)
    cv2.arrowedLine(image, (200, 127), (200, 93), (245, 245, 245), 5)
    return image


class PipelineTests(unittest.TestCase):
    def test_exclusive_xywh_conversion_and_image_edges(self):
        image = np.zeros((20, 30, 3), np.uint8)
        detections = [Detection(2, 3, 30, 20, "red", "circle", .9)]
        with patch("src.detection.find_sign_candidates", return_value=detections):
            self.assertEqual(detect_signs(image), [(2, 3, 28, 17)])

    def test_bgr_crop_pixels_and_original_survive_classifier_mutation(self):
        image = np.arange(6 * 7 * 3, dtype=np.uint8).reshape(6, 7, 3)
        original = image.copy()
        seen = []
        classifier = Mock()
        def predict(crop):
            seen.append(crop.copy())
            crop[:] = 0
            return 42, .7
        classifier.predict.side_effect = predict
        with patch("src.pipeline.detect_signs", return_value=[(1, 2, 6, 4)]):
            result = detect_and_classify(image, classifier)
        np.testing.assert_array_equal(seen[0], original[2:6, 1:7])
        np.testing.assert_array_equal(image, original)
        self.assertEqual(result, [dict(x1=1, y1=2, x2=7, y2=6, class_id=42,
                                      classification_confidence=.7)])

    def test_blank_image_never_calls_classifier(self):
        classifier = Mock()
        self.assertEqual(detect_and_classify(np.full((100, 100, 3), 128, np.uint8), classifier), [])
        classifier.predict.assert_not_called()

    def test_real_detector_and_template_classifier_multiple_signs(self):
        image = two_signs()
        classifier = TemplateMatchingClassifier()
        classifier.fit([image[60:101, 35:76], image[83:138, 173:228]], [1, 38])
        result = detect_and_classify(image, classifier)
        self.assertEqual(len(result), 2)
        self.assertEqual({b["class_id"] for b in result}, {1, 38})

    def test_unknown_rejection_is_preserved(self):
        classifier = Mock()
        classifier.predict.return_value = (-1, .2)
        with patch("src.pipeline.detect_signs", return_value=[(0, 0, 2, 2)]):
            result = detect_and_classify(np.zeros((2, 2, 3), np.uint8), classifier)
        self.assertEqual(result[0]["class_id"], -1)
        self.assertEqual(result[0]["classification_confidence"], .2)

    def test_invalid_predictions_fail(self):
        classifier = Mock()
        for prediction in ((43, .9), (1.5, .9), (True, .9), (1, float("nan"))):
            classifier.predict.return_value = prediction
            with self.subTest(prediction=prediction), patch(
                    "src.pipeline.detect_signs", return_value=[(0, 0, 2, 2)]):
                with self.assertRaises(ValueError):
                    detect_and_classify(np.zeros((2, 2, 3), np.uint8), classifier)

    def test_partial_class_coverage_requires_explicit_choice(self):
        classifier = Mock(classes_=np.array([1, 2, 4, 9, 11, 12, 13, 14, 17, 25, 35, 38]))
        with self.assertRaises(ValueError):
            validate_classifier(classifier)
        with self.assertWarns(UserWarning):
            self.assertEqual(len(validate_classifier(classifier, True)), 31)

    def test_invalid_class_coverage(self):
        for values in (None, [], [0, 43], [0, 0], [1.2], [True]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_classifier(Mock(classes_=values))

    def test_template_can_fit_all_43_nonremapped_class_ids(self):
        rng = np.random.default_rng(42)
        images = [rng.integers(0, 256, (48, 48, 3), dtype=np.uint8) for _ in range(43)]
        classifier = TemplateMatchingClassifier()
        classifier.fit(images, np.arange(43))
        self.assertEqual(validate_classifier(classifier), [])
        actual, _ = classifier.predict_batch(images)
        np.testing.assert_array_equal(actual, np.arange(43))
