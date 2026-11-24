"""Train all classifiers on the GTSRB training split and save them.

Steps:
    1. load the configured GTSRB classes (see src/config.py),
    2. split them by track into training and test part (test part unused here),
    3. train the template-matching baseline and the HOG feature classifiers
       (SVM as main method, k-NN and Random Forest for comparison),
    4. save the models to ``models/`` and a training report to ``results/``.

Usage:
    python train.py
"""

import csv
import json
import time

from src import config
from src.classification import FeatureClassifier, TemplateMatchingClassifier
from src.dataset import load_gtsrb, split_by_track

# Name used for files and reports -> untrained classifier object.
CLASSIFIERS = {
    "template_matching": TemplateMatchingClassifier(),
    "hog_svm": FeatureClassifier("svm"),
    "hog_knn": FeatureClassifier("knn"),
    "hog_rf": FeatureClassifier("rf"),
}


def main() -> None:
    """Train, save and report every classifier in ``CLASSIFIERS``."""
    print("Loading dataset ...")
    images, labels, tracks = load_gtsrb()
    train_idx, _ = split_by_track(labels, tracks)
    print(f"{len(train_idx)} training images, {len(config.CLASS_IDS)} classes")

    config.MODELS_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)

    summary, cv_rows = {}, []
    for name, classifier in CLASSIFIERS.items():
        print(f"Training {name} ...")
        start = time.perf_counter()
        classifier.fit(images[train_idx], labels[train_idx],
                       groups=tracks[train_idx])
        train_seconds = time.perf_counter() - start
        classifier.save(config.MODELS_DIR / f"{name}.joblib")

        summary[name] = {"train_time_s": round(train_seconds, 1)}
        # Only the feature classifiers run a grid search; the baseline has
        # no hyper-parameters to tune.
        if isinstance(classifier, FeatureClassifier):
            best = max(classifier.cv_results_,
                       key=lambda row: row["cv_accuracy_mean"])
            summary[name].update(best)
            cv_rows.extend(classifier.cv_results_)
            print(f"  best CV accuracy {best['cv_accuracy_mean']:.4f} with "
                  f"{best['features']}, {best['params']}")
        print(f"  done in {train_seconds:.1f} s")

    # Training report: best setting per model and the full grid-search log,
    # so the choice of hyper-parameters can be shown in the presentation.
    with open(config.RESULTS_DIR / "training_summary.json", "w") as file:
        json.dump(summary, file, indent=2)
    with open(config.RESULTS_DIR / "cv_results.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=cv_rows[0].keys())
        writer.writeheader()
        writer.writerows(cv_rows)
    print(f"Models saved to {config.MODELS_DIR}, "
          f"reports to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
