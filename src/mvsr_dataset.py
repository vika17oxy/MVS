"""Strict MVSR TRAIN loader and reproducible image-group development split.

No GTSRB defaults are changed. Only train/images, train/labels.csv and
classes.csv are read. Pixel-identical full images stay in one partition.
"""
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from src.mvsr_kl.files import load_image, read_labels

CLASS_IDS = tuple(range(43))
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def development_path(path):
    path = Path(path)
    if any(part.lower() == "heldout"
           for candidate in (path.absolute(), path.resolve())
           for part in candidate.parts):
        raise ValueError("heldout is forbidden in development commands.")
    return path.resolve()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_classes(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    classes = {int(r["class_id"]): r["name_en"].strip() for r in rows}
    if len(rows) != 43 or set(classes) != set(CLASS_IDS):
        raise ValueError("classes.csv must contain each ID 0..42 exactly once.")
    return classes


@dataclass
class TrainingData:
    root: Path
    paths: dict
    labels: dict
    classes: dict
    groups: dict
    hashes: dict
    label_hash: str
    class_hash: str

    def crops(self, names):
        images, targets, groups = [], [], []
        for name in names:
            image = load_image(self.paths[name])
            for box in self.labels[name]:
                images.append(image[box["y1"]:box["y2"],
                                    box["x1"]:box["x2"]].copy())
                targets.append(box["class_id"])
                groups.append(self.groups[name])
        return images, np.asarray(targets, dtype=int), np.asarray(groups)


def load_training(root):
    root = development_path(root)
    images_dir = development_path(root / "train/images")
    labels_path = development_path(root / "train/labels.csv")
    classes_path = development_path(root / "classes.csv")
    classes = read_classes(classes_path)
    expected = read_classes(Path(__file__).with_name("mvsr_classes.csv"))
    if classes != expected:
        raise ValueError("Class IDs/names differ from the verified MVSR/GTSRB mapping.")
    paths = {p.name: p for p in sorted(images_dir.iterdir())
             if p.is_file() and p.suffix.lower() in EXTENSIONS}
    if not paths:
        raise ValueError("No TRAIN images found.")
    labels = read_labels(labels_path)
    if set(labels) - set(paths):
        raise ValueError("Annotation filenames must exactly match TRAIN image basenames.")
    groups, hashes = {}, {}
    for name, path in paths.items():
        resolved = development_path(path)
        if resolved.parent != images_dir:
            raise ValueError(f"Image escapes TRAIN directory: {name}")
        image = load_image(resolved)
        h, w = image.shape[:2]
        seen = set()
        for b in labels.get(name, []):
            key = tuple(b[k] for k in ("x1", "y1", "x2", "y2", "class_id"))
            if (not 0 <= b["x1"] < b["x2"] <= w
                    or not 0 <= b["y1"] < b["y2"] <= h
                    or b["class_id"] not in CLASS_IDS):
                raise ValueError(f"Invalid exclusive MVSR box/class in {name}: {b}")
            if key in seen:
                raise ValueError(f"Duplicate annotation in {name}: {b}")
            seen.add(key)
        # Shape is included to distinguish different layouts of the same bytes.
        digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        groups[name] = digest
        hashes[name] = sha256(resolved)
        labels.setdefault(name, [])  # Negative street images are part of evaluation.
    return TrainingData(root, paths, labels, classes, groups, hashes,
                        sha256(labels_path), sha256(classes_path))


def development_split(data, fraction=0.2, seed=42, min_train_groups=3):
    """Random whole-image groups, reserving up to 3 groups/class for training.

    Rare classes with <=3 groups stay in training. This rule is fixed before
    any model scores are measured, not chosen by validation performance.
    Other frames of the same physical sign cannot be identified without track
    metadata; only identical full-image pixels and same-image ROIs are grouped.
    """
    if not 0 < fraction < 1:
        raise ValueError("Development fraction must be between 0 and 1.")
    if min_train_groups < 1:
        raise ValueError("At least one training group per class is required.")
    group_classes = defaultdict(set)
    for name in data.paths:
        group_classes[data.groups[name]].update(b["class_id"] for b in data.labels[name])
    remaining = Counter(c for values in group_classes.values() for c in values)
    reserve = {c: min(min_train_groups, remaining[c]) for c in CLASS_IDS}
    group_ids = sorted(group_classes)
    rng = np.random.default_rng(seed)
    target = max(1, round(len(group_ids) * fraction))
    dev_groups = set()
    for index in rng.permutation(len(group_ids)):
        group = group_ids[index]
        if all(remaining[c] - 1 >= reserve[c] for c in group_classes[group]):
            dev_groups.add(group)
            remaining.subtract(group_classes[group])
            if len(dev_groups) == target:
                break
    train = sorted(n for n in data.paths if data.groups[n] not in dev_groups)
    dev = sorted(n for n in data.paths if data.groups[n] in dev_groups)
    if not train or not dev or not any(data.labels[n] for n in dev):
        raise ValueError("Insufficient independent images for a useful development split.")
    return train, dev


def split_manifest(data, train, dev, seed, fraction):
    return {
        "schema_version": 1, "dataset_split": "train", "seed": seed,
        "requested_dev_fraction": fraction, "min_train_groups_per_class": 3,
        "grouping": "sha256_of_decoded_full_image_shape_and_pixels",
        "label_sha256": data.label_hash, "classes_sha256": data.class_hash,
        "image_sha256": data.hashes, "image_groups": data.groups,
        "train_images": train, "dev_images": dev,
        "note": "Classifier development split only; detector previously tuned on all TRAIN.",
    }


def validate_manifest(data, manifest):
    train, dev = manifest["train_images"], manifest["dev_images"]
    if (len(set(train)) != len(train) or len(set(dev)) != len(dev)
            or set(train) & set(dev) or set(train) | set(dev) != set(data.paths)
            or not train or not dev):
        raise ValueError("Invalid or incomplete train/dev image partition.")
    if ({data.groups[n] for n in train} & {data.groups[n] for n in dev}):
        raise ValueError("Train/dev share identical source images.")
    for key, expected in (("label_sha256", data.label_hash),
                          ("classes_sha256", data.class_hash),
                          ("image_sha256", data.hashes),
                          ("image_groups", data.groups)):
        if manifest.get(key) != expected:
            raise ValueError(f"Dataset differs from training manifest ({key}).")
    return train, dev
