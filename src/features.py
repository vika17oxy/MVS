"""Feature extraction for the classical-ML classifiers.

Two complementary descriptors are combined:

* **HOG** (Histogram of Oriented Gradients, Dalal & Triggs, CVPR 2005)
  describes the *shape*: the image is divided into small cells, and for every
  cell a histogram of edge directions (gradient orientations, weighted by
  gradient magnitude) is computed. Neighbouring cells are grouped into
  overlapping blocks and each block is normalised, which makes the descriptor
  robust against brightness and contrast changes – important for GTSRB,
  where many images are very dark or over-exposed.
* **Colour histogram** in HSV describes the *colour* (red, blue, yellow...),
  which HOG ignores because it works on a grey-scale image.
"""

import cv2
import numpy as np
from skimage.feature import hog

from src import config
from src.dataset import resize_roi

# HOG parameters. With 48x48 images, 8x8-pixel cells give a 6x6 cell grid;
# 2x2-cell blocks moved by one cell give 5x5 blocks. Each block contributes
# 2*2 cells * 9 orientation bins = 36 values -> 5*5*36 = 900 features.
# 9 unsigned orientation bins (0-180 degrees) is the setting recommended by
# Dalal & Triggs.
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# Colour histogram: joint 2-D histogram of hue (colour tone) and saturation
# (colour purity). Brightness (V) is left out on purpose so that the
# histogram does not depend on lighting. 8 hue x 4 saturation = 32 bins.
HUE_BINS = 8
SATURATION_BINS = 4


def hog_features(image_bgr: np.ndarray) -> np.ndarray:
    """Compute the HOG descriptor of one (already resized) sign image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # 'L2-Hys' = L2 normalisation, clipping of large values at 0.2 and
    # re-normalisation; it limits the influence of single very strong edges.
    return hog(gray,
               orientations=HOG_ORIENTATIONS,
               pixels_per_cell=HOG_PIXELS_PER_CELL,
               cells_per_block=HOG_CELLS_PER_BLOCK,
               block_norm="L2-Hys")


def color_histogram(image_bgr: np.ndarray) -> np.ndarray:
    """Compute a normalised hue/saturation histogram of one sign image."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # In OpenCV, hue ranges from 0 to 179 and saturation from 0 to 255.
    hist = cv2.calcHist([hsv], [0, 1], None, [HUE_BINS, SATURATION_BINS],
                        [0, 180, 0, 256])
    # Divide by the number of pixels so that the values are fractions of the
    # image area and therefore independent of the image size.
    return hist.flatten() / hist.sum()


def extract_features(images, use_color: bool = True,
                     size: int = config.IMAGE_SIZE) -> np.ndarray:
    """Turn a list of sign images (any size) into a feature matrix.

    Args:
        images: iterable of BGR images.
        use_color: append the colour histogram to the HOG descriptor.
        size: images are resized to ``size`` x ``size`` first, because HOG
            only yields vectors of equal length for equally sized images.

    Returns:
        ``(N, D)`` float array, one feature vector per image.
    """
    rows = []
    for image in images:
        image = resize_roi(image, size)
        parts = [hog_features(image)]
        if use_color:
            parts.append(color_histogram(image))
        rows.append(np.concatenate(parts))
    return np.array(rows)
