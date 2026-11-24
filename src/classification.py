"""Traffic sign classifiers behind one common interface.

* :class:`TemplateMatchingClassifier` – baseline: compares the sign with one
  averaged template image per class.
* :class:`FeatureClassifier` – main method: HOG (+ colour) features and a
  classical machine-learning model (SVM; k-NN and Random Forest for
  comparison) whose hyper-parameters are chosen by cross-validation.

Both implement :class:`SignClassifier`, so evaluation and pipeline code can
use either one without knowing which it is (integration contract:
``fit(images, labels)`` and ``predict(roi_bgr) -> (class_id, confidence)``).
"""

from abc import ABC, abstractmethod

import cv2
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config
from src.augmentation import oversample_rare_classes
from src.dataset import resize_roi
from src.features import extract_features

# Class id returned when no class is confident enough.
UNKNOWN = -1


class SignClassifier(ABC):
    """Common base class of all classifiers.

    Subclasses only implement :meth:`fit` and :meth:`_scores` (one score per
    class and image). Choosing the best class, applying the "unknown"
    threshold and saving/loading are identical for all of them and therefore
    implemented here once.
    """

    def __init__(self, threshold: float):
        # Minimum confidence for a valid prediction (see config.py).
        self.threshold = threshold
        # Class ids in the column order of _scores(); set by fit().
        self.classes_ = None

    @abstractmethod
    def fit(self, images, labels, groups=None) -> None:
        """Train on sign images (BGR, any size) and their class ids.

        ``groups`` (track ids) is optional; classifiers that run a
        cross-validation use it to keep all frames of one track in the same
        fold (same reason as for the train/test split, see dataset.py).
        """

    @abstractmethod
    def _scores(self, images) -> np.ndarray:
        """Return an ``(N, n_classes)`` array of class scores.

        A higher score means a better match.
        """

    def predict_batch(self, images, apply_threshold: bool = True):
        """Classify many images at once.

        Args:
            images: list/array of BGR sign images of any size.
            apply_threshold: if False, always return the best class, even
                when its confidence is low (used to evaluate the classifier
                separately from the choice of threshold).

        Returns:
            class_ids: ``(N,)`` int array, ``UNKNOWN`` where the confidence
                is below the threshold.
            confidences: ``(N,)`` float array, score of the best class.
        """
        scores = self._scores(images)
        best = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(best)), best]
        class_ids = self.classes_[best].copy()
        if apply_threshold:
            class_ids[confidences < self.threshold] = UNKNOWN
        return class_ids, confidences

    def predict(self, roi_bgr):
        """Classify one cropped sign (BGR, any size).

        Returns:
            ``(class_id, confidence)``; ``class_id`` is ``UNKNOWN`` when the
            confidence is below the threshold.
        """
        class_ids, confidences = self.predict_batch([roi_bgr])
        return int(class_ids[0]), float(confidences[0])

    def save(self, path) -> None:
        """Store the trained classifier in a file.

        joblib is pickle with efficient storage of large NumPy arrays.
        """
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "SignClassifier":
        """Load a classifier previously stored with :meth:`save`."""
        return joblib.load(path)


class TemplateMatchingClassifier(SignClassifier):
    """Baseline: nearest averaged template by normalised cross-correlation.

    Training builds one template per class: the mean of all training images
    of that class. Each image is standardised (zero mean, unit variance)
    before averaging, so dark and bright photos contribute equally.

    Classification uses ``cv2.matchTemplate`` with ``TM_CCOEFF_NORMED``: the
    template is slid over the image, and at every position the correlation
    coefficient between template and image patch is computed,

        R = sum((T - mean T) * (I - mean I)) / (std T * std I * n),

    which is 1 for a perfect match and independent of brightness and
    contrast. The template is the centre of the averaged image (a margin is
    cut off), so it can slide a few pixels in each direction; this tolerates
    signs that are not perfectly centred in their bounding box.
    """

    def __init__(self, size: int = config.IMAGE_SIZE, margin: int = 4,
                 threshold: float = config.TM_UNKNOWN_THRESHOLD):
        super().__init__(threshold)
        self.size = size
        # Pixels cut from each side of the template = maximum shift tested.
        self.margin = margin
        # (n_classes, size - 2*margin, size - 2*margin, 3) float32 array.
        self.templates_ = None

    @staticmethod
    def _standardize(image: np.ndarray) -> np.ndarray:
        """Scale an image to zero mean and unit standard deviation."""
        image = image.astype(np.float32)
        # The small constant avoids division by zero for flat images.
        return (image - image.mean()) / (image.std() + 1e-6)

    def fit(self, images, labels, groups=None) -> None:
        """Build one averaged colour template per class (``groups`` unused)."""
        labels = np.asarray(labels)
        self.classes_ = np.unique(labels)
        m = self.margin
        templates = []
        for class_id in self.classes_:
            class_images = [self._standardize(resize_roi(image, self.size))
                            for image, label in zip(images, labels)
                            if label == class_id]
            mean_image = np.mean(class_images, axis=0)
            # Colour is kept (3 channels) instead of converting to grey-scale,
            # because colour separates e.g. red from blue signs of the same
            # round shape; matchTemplate sums the correlation over channels.
            templates.append(mean_image[m:-m, m:-m])
        self.templates_ = np.array(templates, dtype=np.float32)

    def _scores(self, images) -> np.ndarray:
        """Best correlation coefficient of every template with every image."""
        scores = np.empty((len(images), len(self.templates_)))
        for i, image in enumerate(images):
            image = resize_roi(image, self.size).astype(np.float32)
            for j, template in enumerate(self.templates_):
                # The result map holds the coefficient for every template
                # position; its maximum is the best alignment.
                result = cv2.matchTemplate(image, template,
                                           cv2.TM_CCOEFF_NORMED)
                scores[i, j] = result.max()
        return scores


# Candidate models and their hyper-parameter grids for the grid search.
# Every model is preceded by a StandardScaler (zero mean, unit variance per
# feature), because SVM and k-NN use distances, which would otherwise be
# dominated by features with large value ranges.
MODELS = {
    # SVM: C trades a wide margin (small C) against fewer training errors
    # (large C). gamma sets the width of the RBF kernel: large gamma = very
    # local decision boundary (risk of overfitting).
    "svm": (SVC(), [
        {"svc__kernel": ["linear"], "svc__C": [0.01, 0.1, 1]},
        {"svc__kernel": ["rbf"], "svc__C": [1, 10, 100],
         "svc__gamma": ["scale", 1e-3, 1e-4]},
    ]),
    # k-NN: number of neighbours that vote; distance weighting gives closer
    # neighbours more influence.
    "knn": (KNeighborsClassifier(), {
        "kneighborsclassifier__n_neighbors": [1, 3, 5, 9],
        "kneighborsclassifier__weights": ["uniform", "distance"],
    }),
    # Random Forest: many decision trees on random subsets of data and
    # features; the majority vote reduces the variance of a single tree.
    "rf": (RandomForestClassifier(random_state=config.RANDOM_SEED), {
        "randomforestclassifier__n_estimators": [100, 300],
        "randomforestclassifier__max_depth": [None, 20],
    }),
}


class FeatureClassifier(SignClassifier):
    """HOG (+ optional colour histogram) features + a scikit-learn model.

    ``fit`` runs a grid search with cross-validation on the training data
    only, for both feature variants (HOG alone and HOG + colour histogram),
    and keeps the best combination. The test set is never touched here.
    Classes with too few examples for the cross-validation are filled up by
    augmentation first (see src/augmentation.py).
    """

    def __init__(self, model: str = "svm",
                 threshold: float = config.ML_UNKNOWN_THRESHOLD):
        super().__init__(threshold)
        self.model = model
        # Set by fit(): chosen feature variant, trained model, search log
        # and the classes that had to be augmented (for the reports).
        self.use_color_ = None
        self.estimator_ = None
        self.cv_results_ = []
        self.rare_classes_ = []

    def _cv_splits(self, labels, groups):
        """Fixed list of cross-validation folds, reused by every search.

        With track ids, frames of one track never end up in both the
        training and validation part of a fold; otherwise the CV score would
        be too optimistic and favour overfitting hyper-parameters.
        """
        dummy_x = np.zeros(len(labels))
        if groups is None:
            splitter = StratifiedKFold(config.CV_FOLDS, shuffle=True,
                                       random_state=config.RANDOM_SEED)
            return list(splitter.split(dummy_x, labels))
        splitter = StratifiedGroupKFold(config.CV_FOLDS, shuffle=True,
                                        random_state=config.RANDOM_SEED)
        return list(splitter.split(dummy_x, labels, groups))

    @staticmethod
    def _check_folds(labels, splits) -> None:
        """Stop with a clear message if a CV training fold lacks a class.

        Without this check, scikit-learn would only warn, the SVM would
        silently never learn that class in the fold, and the calibrated
        probabilities of the class would be meaningless.
        """
        all_classes = set(np.unique(labels))
        for train_idx, _ in splits:
            missing = all_classes - set(np.unique(labels[train_idx]))
            if missing:
                raise ValueError(
                    f"Classes {sorted(map(int, missing))} are missing from a "
                    "CV training fold. Increase MIN_GROUPS_PER_CLASS or "
                    "MIN_SAMPLES_PER_CLASS in src/config.py.")

    def fit(self, images, labels, groups=None) -> None:
        """Choose features and hyper-parameters by CV, then train the model."""
        # Rare classes get augmented copies and ungrouped samples, so that
        # every class occurs in every CV fold (unchanged if none is rare).
        images, labels, groups, self.rare_classes_ = oversample_rare_classes(
            images, labels, groups)
        splits = self._cv_splits(labels, groups)
        self._check_folds(labels, splits)
        estimator, param_grid = MODELS[self.model]

        best_search, best_features = None, None
        for use_color in (False, True):
            features = extract_features(images, use_color)
            search = GridSearchCV(make_pipeline(StandardScaler(), estimator),
                                  param_grid, cv=splits, scoring="accuracy",
                                  n_jobs=-1)
            search.fit(features, labels)
            self._log_search(search, use_color)
            if (best_search is None
                    or search.best_score_ > best_search.best_score_):
                best_search, best_features = search, features
                self.use_color_ = use_color

        # GridSearchCV has already re-trained the best pipeline on the whole
        # training set (refit=True).
        self.estimator_ = best_search.best_estimator_
        if self.model == "svm":
            # An SVM outputs distances to the decision boundary, not
            # probabilities. CalibratedClassifierCV learns a mapping from
            # these distances to probabilities (Platt scaling = sigmoid fit)
            # on cross-validated predictions, so the confidence can be
            # compared with a probability threshold like for k-NN and RF.
            self.estimator_ = CalibratedClassifierCV(
                self.estimator_, method="sigmoid", cv=splits, ensemble=False)
            self.estimator_.fit(best_features, labels)
        self.classes_ = self.estimator_.classes_

    def _log_search(self, search: GridSearchCV, use_color: bool) -> None:
        """Remember mean/std CV accuracy of every tested parameter set."""
        results = search.cv_results_
        for params, mean, std in zip(results["params"],
                                     results["mean_test_score"],
                                     results["std_test_score"]):
            self.cv_results_.append({
                "model": self.model,
                "features": "HOG + color" if use_color else "HOG",
                # Drop the "svc__" style prefixes for readable reports.
                "params": {key.split("__")[1]: value
                           for key, value in params.items()},
                "cv_accuracy_mean": float(mean),
                "cv_accuracy_std": float(std),
            })

    def _scores(self, images) -> np.ndarray:
        """Class probabilities for every image."""
        features = extract_features(images, self.use_color_)
        return self.estimator_.predict_proba(features)
