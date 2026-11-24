"""Tests for the handling of rare classes (augmentation + CV folds).

Run with:  python -m unittest discover -s tests -v
Uses small synthetic sign images, so no dataset is needed.
"""

import unittest

import cv2
import numpy as np

from src import config
from src.augmentation import augment_image, oversample_rare_classes
from src.classification import FeatureClassifier


def synthetic_sign(shape: str, color, rng) -> np.ndarray:
    """Draw a simple 'sign' (circle/triangle/square) with random noise."""
    image = np.full((48, 48, 3), 128, np.uint8)
    if shape == "circle":
        cv2.circle(image, (24, 24), 18, color, -1)
    elif shape == "triangle":
        points = np.array([(24, 5), (43, 42), (5, 42)], np.int32)
        cv2.fillPoly(image, [points], color)
    else:
        cv2.rectangle(image, (8, 8), (40, 40), color, -1)
    noise = rng.integers(-15, 16, image.shape)
    return np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)


def synthetic_dataset(rare_examples: int = 1):
    """Three frequent classes (5 groups x 3 images) and one rare class."""
    rng = np.random.default_rng(0)
    kinds = {0: ("circle", (0, 0, 220)), 1: ("circle", (220, 0, 0)),
             2: ("triangle", (0, 0, 220)), 3: ("square", (220, 0, 0))}
    images, labels, groups = [], [], []
    for class_id in (0, 1, 2):
        for group in range(5):
            for _ in range(3):
                images.append(synthetic_sign(*kinds[class_id], rng))
                labels.append(class_id)
                groups.append(f"c{class_id}_g{group}")
    for number in range(rare_examples):
        images.append(synthetic_sign(*kinds[3], rng))
        labels.append(3)
        groups.append(f"rare_source_{number}")
    return images, np.array(labels), np.array(groups)


class AugmentImageTest(unittest.TestCase):
    def test_output_size_type_and_reproducibility(self):
        image = synthetic_sign("circle", (0, 0, 220),
                               np.random.default_rng(1))
        first = augment_image(image, np.random.default_rng(5))
        second = augment_image(image, np.random.default_rng(5))
        self.assertEqual(first.shape, (config.IMAGE_SIZE,
                                       config.IMAGE_SIZE, 3))
        self.assertEqual(first.dtype, np.uint8)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, image))

    def test_accepts_any_input_size(self):
        image = np.full((31, 57, 3), 90, np.uint8)
        result = augment_image(image, np.random.default_rng(0))
        self.assertEqual(result.shape[:2],
                         (config.IMAGE_SIZE, config.IMAGE_SIZE))


class OversampleTest(unittest.TestCase):
    def test_no_rare_class_leaves_data_unchanged(self):
        images, labels, groups = synthetic_dataset(rare_examples=0)
        out_images, out_labels, out_groups, rare = oversample_rare_classes(
            images, labels, groups)
        self.assertEqual(rare, [])
        self.assertIs(out_images, images)
        np.testing.assert_array_equal(out_labels, labels)
        self.assertIs(out_groups, groups)

    def test_no_rare_class_without_groups_keeps_none(self):
        images, labels, _ = synthetic_dataset(rare_examples=0)
        _, _, out_groups, rare = oversample_rare_classes(images, labels)
        self.assertEqual(rare, [])
        self.assertIsNone(out_groups)

    def test_rare_class_is_filled_up_with_own_groups(self):
        images, labels, groups = synthetic_dataset(rare_examples=1)
        out_images, out_labels, out_groups, rare = oversample_rare_classes(
            images, labels, groups)
        self.assertEqual(rare, [3])
        rare_mask = out_labels == 3
        self.assertEqual(rare_mask.sum(), config.MIN_SAMPLES_PER_CLASS)
        self.assertEqual(len(set(out_groups[rare_mask])),
                         config.MIN_SAMPLES_PER_CLASS)
        self.assertEqual(len(out_images), len(out_labels))
        # Frequent classes keep their samples and their groups.
        frequent = labels != 3
        np.testing.assert_array_equal(out_groups[:len(labels)][frequent],
                                      groups[frequent])
        self.assertEqual((out_labels != 3).sum(), frequent.sum())


class FeatureClassifierRareClassTest(unittest.TestCase):
    def test_svm_learns_class_with_single_example(self):
        images, labels, groups = synthetic_dataset(rare_examples=1)
        classifier = FeatureClassifier("svm")
        classifier.fit(images, labels, groups=groups)
        self.assertEqual(list(classifier.classes_), [0, 1, 2, 3])
        self.assertEqual(classifier.rare_classes_, [3])
        # A new, unseen square must be recognised as the rare class.
        new_square = synthetic_sign("square", (220, 0, 0),
                                    np.random.default_rng(99))
        class_ids, _ = classifier.predict_batch([new_square],
                                                apply_threshold=False)
        self.assertEqual(class_ids[0], 3)

    def test_check_folds_reports_missing_class(self):
        labels = np.array([0, 0, 1, 1, 2])
        # The only example of class 2 is in the validation part.
        splits = [(np.array([0, 1, 2, 3]), np.array([4]))]
        with self.assertRaises(ValueError):
            FeatureClassifier._check_folds(labels, splits)


if __name__ == "__main__":
    unittest.main()
