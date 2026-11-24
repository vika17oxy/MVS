"""Evaluate all trained classifiers on the test split and write ``results/``.

Requires the models created by ``python train.py``. Uses the same
reproducible track-based split as training, so the test images were never
seen during training or hyper-parameter search.

Outputs (in ``results/``):
    comparison.csv                    one row per classifier: metrics + timing
    metrics.json                      all metrics incl. per-class values
    per_class_<model>.csv             precision / recall / F1 per class
    confusion_matrix_<model>.png      row-normalised confusion matrix
    examples_correct_<model>.png      correctly classified test images
    examples_wrong_<model>.png        misclassified test images
    f1_per_class_comparison.png       baseline vs. HOG+SVM per class

Usage:
    python evaluate.py
"""

import csv
import json

import numpy as np

from src import config
from src.classification import UNKNOWN, SignClassifier
from src.dataset import load_gtsrb, split_by_track
from src.evaluation import (compute_metrics, measure_inference_time,
                            plot_confusion_matrix, plot_examples,
                            plot_f1_comparison)

# Model file names (as written by train.py) -> name shown in reports.
MODELS = {
    "template_matching": "Template Matching",
    "hog_svm": "HOG + SVM",
    "hog_knn": "HOG + k-NN",
    "hog_rf": "HOG + Random Forest",
}
# The two methods required by the task, compared class by class.
BASELINE, MAIN_METHOD = "template_matching", "hog_svm"


def save_examples(name: str, title: str, images, y_true, y_pred, confidences):
    """Save one grid of correct and one grid of wrong predictions.

    Images are picked at random (fixed seed) instead of taking the first
    ones, which would all show the same few tracks of the first class.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    for kind, mask in (("correct", y_pred == y_true),
                       ("wrong", y_pred != y_true)):
        indices = np.flatnonzero(mask)
        chosen = rng.choice(indices, size=min(12, len(indices)), replace=False)
        plot_examples(images[chosen], y_true[chosen], y_pred[chosen],
                      confidences[chosen], f"{title}: {kind} predictions",
                      config.RESULTS_DIR / f"examples_{kind}_{name}.png")


def write_per_class_csv(name: str, metrics: dict) -> None:
    """Write the per-class metrics of one classifier as a CSV table."""
    path = config.RESULTS_DIR / f"per_class_{name}.csv"
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class_id", "name", "precision", "recall", "f1",
                         "support"])
        for class_id, row in metrics["per_class"].items():
            writer.writerow([class_id, row["name"], f"{row['precision']:.4f}",
                             f"{row['recall']:.4f}", f"{row['f1']:.4f}",
                             row["support"]])


def main() -> None:
    """Evaluate every model on the test split and save all results."""
    print("Loading dataset ...")
    images, labels, tracks = load_gtsrb()
    _, test_idx = split_by_track(labels, tracks)
    test_images, test_labels = images[test_idx], labels[test_idx]
    print(f"{len(test_idx)} test images")
    config.RESULTS_DIR.mkdir(exist_ok=True)

    all_metrics, comparison = {}, []
    for name, title in MODELS.items():
        model_path = config.MODELS_DIR / f"{name}.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{model_path} missing - run 'python train.py' first.")
        classifier = SignClassifier.load(model_path)

        print(f"Evaluating {title} ...")
        # Predict once without threshold, then apply it here. This gives two
        # views from one (slow) prediction run: the pure classifier quality
        # (best class always taken) and the behaviour of the full system
        # with "unknown" rejection, as it is used in the pipeline.
        y_best, confidences = classifier.predict_batch(test_images,
                                                       apply_threshold=False)
        y_pred = np.where(confidences >= classifier.threshold, y_best, UNKNOWN)
        metrics = compute_metrics(test_labels, y_pred, config.CLASS_IDS)
        metrics["accuracy_without_threshold"] = float(
            np.mean(y_best == test_labels))
        metrics["inference_ms_per_image"] = measure_inference_time(
            classifier, test_images)
        all_metrics[name] = metrics

        comparison.append({
            "method": title,
            "accuracy_without_threshold":
                metrics["accuracy_without_threshold"],
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "unknown_rate": metrics["unknown_rate"],
            "inference_ms_per_image": metrics["inference_ms_per_image"],
        })
        print(f"  accuracy {metrics['accuracy']:.4f} (without threshold "
              f"{metrics['accuracy_without_threshold']:.4f}), macro F1 "
              f"{metrics['macro_f1']:.4f}, "
              f"{metrics['inference_ms_per_image']:.2f} ms/image")

        write_per_class_csv(name, metrics)
        plot_confusion_matrix(
            test_labels, y_pred, config.CLASS_IDS, title,
            config.RESULTS_DIR / f"confusion_matrix_{name}.png")
        save_examples(name, title, test_images, test_labels, y_pred,
                      confidences)

    plot_f1_comparison(all_metrics[BASELINE], all_metrics[MAIN_METHOD],
                       [MODELS[BASELINE], MODELS[MAIN_METHOD]],
                       config.RESULTS_DIR / "f1_per_class_comparison.png")

    with open(config.RESULTS_DIR / "comparison.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=comparison[0].keys())
        writer.writeheader()
        # Round for readability; four decimals are more than the test set
        # size (~4000 images) can resolve anyway.
        writer.writerows(
            {key: round(value, 4) if isinstance(value, float) else value
             for key, value in row.items()}
            for row in comparison)
    with open(config.RESULTS_DIR / "metrics.json", "w") as file:
        json.dump(all_metrics, file, indent=2)

    # Print the comparison table so the result is visible without opening
    # any file.
    print(f"\n{'method':<22}{'acc (no thr.)':>14}{'accuracy':>10}"
          f"{'macro F1':>10}{'unknown':>9}{'ms/img':>8}")
    for row in comparison:
        print(f"{row['method']:<22}{row['accuracy_without_threshold']:>14.4f}"
              f"{row['accuracy']:>10.4f}{row['macro_f1']:>10.4f}"
              f"{row['unknown_rate']:>9.3f}"
              f"{row['inference_ms_per_image']:>8.2f}")
    print(f"\nAll results written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
