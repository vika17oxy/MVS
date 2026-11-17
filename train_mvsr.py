"""Train Viktoriia's unchanged classifiers on MVSR TRAIN ground-truth crops."""
import argparse
from collections import Counter
import importlib.metadata
from pathlib import Path
import platform
import time

import numpy as np

from src import config
from src.classification import FeatureClassifier, TemplateMatchingClassifier
from src.mvsr_dataset import (CLASS_IDS, development_path, development_split,
                              load_training, sha256, split_manifest)
from src.mvsr_kl.files import write_json
from src.pipeline import validate_classifier


def environment():
    packages = ("numpy", "opencv-python", "scikit-learn", "scikit-image", "joblib")
    return {"python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in packages}}


def source_hashes():
    root = Path(__file__).resolve().parent
    paths = sorted((root / "src").rglob("*.py")) + [
        root / "train_mvsr.py", root / "evaluate_mvsr.py", root / "evaluate_detection.py"]
    return {p.relative_to(root).as_posix(): sha256(p) for p in paths}


def prepare_classifier(method, images, labels, groups, allow_partial=False):
    """Keep existing fit(), CV, HOG and rejection methods unchanged."""
    if method == "template":
        classifier = TemplateMatchingClassifier()
    else:
        classifier = FeatureClassifier(method)
        counts = {c: len(set(groups[labels == c])) for c in CLASS_IDS}
        insufficient = [c for c in CLASS_IDS if counts[c] < config.CV_FOLDS]
        if insufficient and not allow_partial:
            raise ValueError(
                f"Cannot train {config.CV_FOLDS}-fold {method} for all 43 classes: "
                f"fewer than {config.CV_FOLDS} independent training groups for {insufficient}. "
                "Use template for 43-class integration, or --allow-partial-classes "
                "for an explicitly incomplete feature-classifier experiment.")
        if insufficient:
            keep = np.array([c not in insufficient for c in labels])
            images = [img for img, selected in zip(images, keep) if selected]
            labels, groups = labels[keep], groups[keep]
        if len(set(labels)) < 2:
            raise ValueError("Feature classifiers need at least two supported classes.")
        splits = classifier._cv_splits(labels, groups)
        for train_idx, validation_idx in splits:
            if set(groups[train_idx]) & set(groups[validation_idx]):
                raise ValueError("Cross-validation group leakage.")
            if set(labels[train_idx]) != set(labels):
                raise ValueError("A CV training fold loses a class. More independent data required.")
            if method == "knn" and len(train_idx) < 9:
                raise ValueError("Too few samples for the original k-NN grid (up to 9 neighbours).")
    if not allow_partial and set(map(int, labels)) != set(CLASS_IDS):
        raise ValueError("Training crops do not cover all 43 classes.")
    return classifier, images, labels, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--method", choices=("template", "svm", "knn", "rf"), default="template")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--allow-partial-classes", action="store_true")
    args = parser.parse_args()
    output = development_path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty. Choose a new model directory.")
    try:
        data = load_training(args.dataset)
        train, dev = development_split(data, args.dev_fraction, args.seed)
        images, labels, groups = data.crops(train)
        classifier, images, labels, groups = prepare_classifier(
            args.method, images, labels, groups, args.allow_partial_classes)
    except ValueError as error:
        parser.error(str(error))
    print(f"Training {args.method}: {len(images)} crops, {len(set(labels))} classes, "
          f"{len(train)} training / {len(dev)} development images", flush=True)
    start = time.perf_counter()
    classifier.fit(images, labels, groups=groups)
    elapsed = time.perf_counter() - start
    missing = validate_classifier(classifier, args.allow_partial_classes)
    output.mkdir(parents=True, exist_ok=True)
    classifier.save(output / "model.joblib")
    manifest = split_manifest(data, train, dev, args.seed, args.dev_fraction)
    manifest.update({
        "method": args.method, "trained_classes": list(map(int, classifier.classes_)),
        "missing_classes": missing, "model_sha256": sha256(output / "model.joblib"),
        "training_class_counts": {str(c): int(sum(labels == c)) for c in CLASS_IDS},
        "development_class_counts": dict(sorted(Counter(
            str(b["class_id"]) for n in dev for b in data.labels[n]).items())),
        "environment": environment(), "source_sha256": source_hashes(),
        "threshold": classifier.threshold, "train_seconds": elapsed,
    })
    write_json(output / "manifest.json", manifest)
    if isinstance(classifier, FeatureClassifier):
        write_json(output / "cv_results.json", classifier.cv_results_)
    print(f"Saved {output / 'model.joblib'}; missing classes: {missing}", flush=True)


if __name__ == "__main__":
    main()
