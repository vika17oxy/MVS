"""Data augmentation for classes with too few training examples.

Problem: in the course dataset some classes (e.g. 19, 24, 27, 31, 37) occur
on only one or two training images. The feature classifiers choose their
hyper-parameters by a *grouped* 3-fold cross-validation (all crops of one
source image / track stay in the same fold, see classification.py). A class
with fewer groups than folds is then missing from at least one training
fold: the SVM cannot learn it there, and the probability calibration of the
SVM (CalibratedClassifierCV) has no usable scores for it.

Solution used here (both options from the project plan combined):

1. **Augmentation**: every rare class is filled up with randomly
   transformed copies of its few real examples until it has
   ``config.MIN_SAMPLES_PER_CLASS`` samples. The transformations imitate
   variations that also occur in real recordings: slightly tilted signs,
   imprecise detector boxes, other lighting, blur.
2. **No groups for rare classes**: every sample of a rare class (original
   and copy) gets its own group id, so the cross-validation may spread them
   over all folds.

Known limitation (must be mentioned in the report): copies of the same
original image can end up in a training *and* a validation fold, so the CV
accuracy of the rare classes is optimistic. With only one real example
there is no leak-free alternative. Rare classes are a handful of samples
among thousands, so this hardly influences the choice of hyper-parameters.
Classes with enough groups are not changed at all.
"""

import cv2
import numpy as np

from src import config
from src.dataset import resize_roi

# Maximum rotation in degrees. Real signs are only slightly tilted (pole,
# camera roll); larger angles would create views that never occur.
MAX_ROTATION_DEG = 10
# Random zoom and shift (fraction of the image size). They imitate detector
# boxes that are a little too large, too small or not centred.
SCALE_RANGE = (0.9, 1.1)
MAX_SHIFT = 0.08
# Contrast factor and brightness offset (grey values 0..255) for different
# lighting, e.g. shadow or low sun.
CONTRAST_RANGE = (0.7, 1.3)
MAX_BRIGHTNESS_SHIFT = 30
# Half of the copies are slightly blurred (distant, upscaled or moving
# signs). Sigma in pixels of the 48 x 48 image.
BLUR_PROBABILITY = 0.5
BLUR_SIGMA_RANGE = (0.5, 1.2)
# Mirroring is deliberately NOT used: it changes the meaning of many signs
# (keep left <-> keep right, curve left <-> curve right) and mirrors digits.


def augment_image(image_bgr: np.ndarray, rng: np.random.Generator,
                  size: int = config.IMAGE_SIZE) -> np.ndarray:
    """Return one randomly transformed ``size`` x ``size`` copy of a sign.

    Args:
        image_bgr: sign crop (BGR, any size).
        rng: NumPy random generator; passing a seeded generator makes the
            copies reproducible.
        size: side length of the returned image (same as for features).
    """
    image = resize_roi(image_bgr, size)

    # Rotation, zoom and shift are combined into one affine transformation,
    # so the image is interpolated only once.
    angle = rng.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG)
    scale = rng.uniform(*SCALE_RANGE)
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
    matrix[:, 2] += rng.uniform(-MAX_SHIFT, MAX_SHIFT, size=2) * size
    # BORDER_REPLICATE continues the edge pixels instead of adding black
    # corners, which would be an artefact that real crops never have.
    image = cv2.warpAffine(image, matrix, (size, size),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

    # new = contrast * old + brightness, clipped to 0..255.
    image = cv2.convertScaleAbs(
        image, alpha=rng.uniform(*CONTRAST_RANGE),
        beta=rng.uniform(-MAX_BRIGHTNESS_SHIFT, MAX_BRIGHTNESS_SHIFT))

    if rng.random() < BLUR_PROBABILITY:
        # Kernel size (0, 0) lets OpenCV derive it from sigma.
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(*BLUR_SIGMA_RANGE))
    return image


def oversample_rare_classes(images, labels, groups=None,
                            min_groups: int = config.MIN_GROUPS_PER_CLASS,
                            min_samples: int = config.MIN_SAMPLES_PER_CLASS,
                            seed: int = config.RANDOM_SEED):
    """Fill up classes with too few groups by augmented copies.

    Args:
        images: sequence of BGR sign crops (any size).
        labels: class id per image.
        groups: group id per image (track / source image) or ``None``, in
            which case every image counts as its own group.
        min_groups: a class with fewer distinct groups counts as rare.
        min_samples: number of samples a rare class has afterwards.
        seed: seed of the random transformations (reproducible training).

    Returns:
        images, labels, groups, rare_classes. If no class is rare, the
        inputs are returned unchanged (labels as array, groups as given),
        so results on well-filled datasets such as GTSRB stay identical.
    """
    labels = np.asarray(labels)
    # Without groups every image is independent, i.e. its own group.
    # Strings for all ids, so original and new group ids can be mixed and
    # sorted (StratifiedGroupKFold calls np.unique on them).
    group_ids = np.asarray(
        np.arange(len(labels)) if groups is None else groups).astype(str)

    rare_classes = [int(c) for c in np.unique(labels)
                    if len(np.unique(group_ids[labels == c])) < min_groups]
    if not rare_classes:
        return images, labels, groups, rare_classes
    groups = group_ids

    rng = np.random.default_rng(seed)
    images, labels, groups = list(images), list(labels), list(groups)
    for class_id in rare_classes:
        originals = [i for i, label in enumerate(labels) if label == class_id]
        # Each real example of the class becomes its own group ...
        for number, index in enumerate(originals):
            groups[index] = f"rare_{class_id}_{number}"
        # ... and the copies are made from the originals in turn, so that
        # every real example is used equally often.
        for number in range(len(originals), min_samples):
            source = originals[number % len(originals)]
            images.append(augment_image(images[source], rng))
            labels.append(class_id)
            groups.append(f"rare_{class_id}_{number}")
    return images, np.asarray(labels), np.asarray(groups), rare_classes
