"""Street image -> automatic boxes -> unmodified BGR crops -> classifier."""
from numbers import Integral
import warnings

import numpy as np

from src.detection import detect_signs

CLASS_IDS = tuple(range(43))
UNKNOWN = -1


def validate_classifier(classifier, allow_partial=False):
    """Check fitted class coverage, without consulting any image labels."""
    classes = getattr(classifier, "classes_", None)
    if classes is None or not len(classes):
        raise ValueError("Classifier is not fitted (classes_ is empty).")
    if any(not isinstance(c, Integral) or isinstance(c, (bool, np.bool_))
           or int(c) not in CLASS_IDS for c in classes):
        raise ValueError("Classifier classes must be integer MVSR IDs 0..42.")
    if len(set(classes)) != len(classes):
        raise ValueError("Classifier classes_ contains duplicate IDs.")
    missing = sorted(set(CLASS_IDS) - set(map(int, classes)))
    if missing and not allow_partial:
        raise ValueError(f"Classifier does not cover all 43 MVSR classes; missing {missing}. "
                         "Explicitly allow partial coverage for a diagnostic run.")
    if missing:
        warnings.warn(f"Partial classifier: missing MVSR classes {missing}",
                      UserWarning, stacklevel=2)
    return missing


def detect_and_classify(image_bgr, classifier):
    """Return one record per candidate, including UNKNOWN=-1 rejections.

    Scores belong to the classifier (NCC for templates, probability for ML).
    The detector's heuristic score is never substituted for class confidence.
    The caller loads/trains and validates the classifier once before a run.
    """
    boxes = detect_signs(image_bgr)
    results = []
    for x, y, width, height in boxes:
        # A copy protects the original image even if a classifier mutates its ROI.
        roi = image_bgr[y:y + height, x:x + width].copy()
        class_id, confidence = classifier.predict(roi)
        if (not isinstance(class_id, Integral)
                or isinstance(class_id, (bool, np.bool_))
                or int(class_id) not in (*CLASS_IDS, UNKNOWN)):
            raise ValueError("Classifier returned an invalid class ID.")
        if not np.isfinite(confidence):
            raise ValueError("Classifier returned a non-finite confidence.")
        results.append({
            "x1": x, "y1": y, "x2": x + width, "y2": y + height,
            "class_id": int(class_id),
            "classification_confidence": float(confidence),
        })
    return results
