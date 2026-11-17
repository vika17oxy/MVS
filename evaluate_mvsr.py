"""Evaluate a saved MVSR model on its recorded TRAIN development images."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from src.classification import SignClassifier
from src.mvsr_dataset import (development_path, load_training, sha256, validate_manifest)
from src.mvsr_evaluation import evaluate_system
from src.mvsr_kl.files import load_image, write_json
from src.pipeline import detect_and_classify, validate_classifier
from train_mvsr import environment, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial-classes", action="store_true")
    args = parser.parse_args()
    try:
        model_dir = development_path(args.model_dir)
        output = development_path(args.output)
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is not empty. Choose a new results directory.")
        data = load_training(args.dataset)
        manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        _, dev = validate_manifest(data, manifest)
        if sha256(model_dir / "model.joblib") != manifest["model_sha256"]:
            raise ValueError("Model file does not match its training manifest.")
        classifier = SignClassifier.load(model_dir / "model.joblib")
        missing = validate_classifier(classifier, args.allow_partial_classes)
        if list(map(int, classifier.classes_)) != manifest["trained_classes"]:
            raise ValueError("Model class coverage differs from its training manifest.")
    except ValueError as error:
        parser.error(str(error))
    predictions = {}
    start = time.perf_counter()
    for i, name in enumerate(dev, 1):
        predictions[name] = detect_and_classify(load_image(data.paths[name]), classifier)
        if i % 50 == 0:
            print(f"Evaluated {i}/{len(dev)} development images", flush=True)
    elapsed = time.perf_counter() - start
    metrics = evaluate_system(predictions, {n: data.labels[n] for n in dev})
    images, actual, _ = data.crops(dev)
    predicted, _ = classifier.predict_batch(images)
    best, _ = classifier.predict_batch(images, apply_threshold=False)
    metrics["classification_on_ground_truth_crops"] = {
        "count": len(images), "correct": int(np.sum(predicted == actual)),
        "accuracy_including_unknown": float(np.mean(predicted == actual)),
        "accuracy_without_threshold": float(np.mean(best == actual)),
        "note": "Oracle GT boxes; not full street-image system accuracy.",
    }
    metrics["scope"] = {
        "dataset_split": "train", "purpose": "classifier_development_validation",
        "images": len(dev), "method": manifest["method"],
        "trained_classes": manifest["trained_classes"], "missing_classes": missing,
        "detector_note": "Detector was previously selected on all TRAIN; not independent system test.",
        "grouping_limit": "Same image and identical pixels grouped; no physical-sign track IDs available.",
        "model_sha256": manifest["model_sha256"],
    }
    metrics["timing_seconds_including_image_io"] = elapsed
    metrics["environment"] = environment()
    metrics["source_sha256"] = source_hashes()
    write_json(output / "metrics.json", metrics)
    write_json(output / "predictions.json", predictions)
    write_json(output / "split_manifest.json", manifest)
    with (output / "per_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["class_id", "name"] + list(metrics["per_class"]["0"]))
        writer.writeheader()
        for c, row in metrics["per_class"].items():
            writer.writerow({"class_id": c, "name": data.classes[int(c)], **row})
    confusion = metrics["confusion"]
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true / predicted"] + confusion["columns"])
        writer.writerows([label] + row for label, row in zip(confusion["rows"], confusion["counts"]))
    print(json.dumps({k: metrics[k] for k in (
        "detection", "classification_on_matched_candidates",
        "classification_on_ground_truth_crops", "end_to_end")}, indent=2))


if __name__ == "__main__":
    main()
