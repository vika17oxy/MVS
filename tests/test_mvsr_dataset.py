"""TRAIN-only data format, split provenance and class mapping checks."""
import csv
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from src.mvsr_dataset import (development_path, development_split, load_training,
                              split_manifest, validate_manifest)
from src.mvsr_kl.files import save_image
from train_mvsr import prepare_classifier


class MvsrDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "train/images"
        self.images.mkdir(parents=True)
        shutil.copyfile(Path(__file__).resolve().parents[1] / "src/mvsr_classes.csv",
                        self.root / "classes.csv")
        for i in range(10):
            image = np.full((20, 30, 3), 10 * i, np.uint8)
            save_image(self.images / f"{i}.png", image)
        self.rows = [[f"{i}.png", 0, 0, 30, 20, 1] for i in range(8)]
        self.rows.append(["8.png", 0, 0, 30, 20, 19])
        self.write_labels()

    def write_labels(self):
        with (self.root / "train/labels.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["filename", "x1", "y1", "x2", "y2", "class_id"])
            writer.writerows(self.rows)

    def test_negative_images_and_exclusive_crops(self):
        data = load_training(self.root)
        self.assertEqual(data.labels["9.png"], [])
        crops, labels, _ = data.crops(["0.png", "8.png"])
        self.assertEqual(crops[0].shape, (20, 30, 3))
        self.assertEqual(labels.tolist(), [1, 19])
        self.assertEqual(set(data.classes), set(range(43)))

    def test_split_is_repeatable_disjoint_and_keeps_rare_class_in_train(self):
        # A second encoding of the same image is still grouped by decoded pixels.
        shutil.copyfile(self.images / "0.png", self.images / "copy.png")
        data = load_training(self.root)
        train, dev = development_split(data)
        self.assertEqual((train, dev), development_split(data))
        self.assertIn("8.png", train)
        self.assertEqual("0.png" in train, "copy.png" in train)
        self.assertFalse(set(train) & set(dev))
        self.assertFalse({data.groups[n] for n in train} & {data.groups[n] for n in dev})
        manifest = split_manifest(data, train, dev, 42, .2)
        self.assertEqual(validate_manifest(data, manifest), (train, dev))
        manifest["label_sha256"] = "changed"
        with self.assertRaises(ValueError):
            validate_manifest(data, manifest)

    def test_manifest_cannot_evaluate_training_images(self):
        data = load_training(self.root)
        train, dev = development_split(data)
        manifest = split_manifest(data, train, dev, 42, .2)
        manifest["dev_images"] = dev + train[:1]
        with self.assertRaises(ValueError):
            validate_manifest(data, manifest)

    def test_heldout_path_blocked_before_read(self):
        for name in ("heldout/train/images", "HeLdOuT/labels.csv"):
            with self.assertRaises(ValueError):
                development_path(self.root / name)
        with self.assertRaises(ValueError):
            load_training(self.root / "heldout")

    def test_unknown_filename_out_of_bounds_and_duplicate_annotations(self):
        variants = [
            ["missing.png", 0, 0, 30, 20, 1],
            ["0.png", 0, 0, 31, 20, 1],
            ["0.png", 0, 0, 30, 20, 43],
            ["0.png", 0, 0, 0, 20, 1],
            self.rows[0],
        ]
        for row in variants:
            with self.subTest(row=row):
                self.rows.append(row)
                self.write_labels()
                with self.assertRaises(ValueError):
                    load_training(self.root)
                self.rows.pop()

    def test_swapped_class_names_are_rejected(self):
        path = self.root / "classes.csv"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("Speed limit (20km/h)", "WRONG"), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_training(self.root)

    def test_full_svm_rejects_insufficient_class_groups_before_training(self):
        images = [np.zeros((20, 20, 3), np.uint8) for _ in range(43)]
        with self.assertRaisesRegex(ValueError, "independent training groups"):
            prepare_classifier("svm", images, np.arange(43), np.arange(43).astype(str))

    def test_partial_template_training_requires_explicit_choice(self):
        images = [np.zeros((20, 20, 3), np.uint8)]
        with self.assertRaises(ValueError):
            prepare_classifier("template", images, np.array([1]), np.array(["image"]))
