"""Traffic sign detection (localization) – STUB.

This module belongs to the partner's part of the project (colour
segmentation in HSV, contour and shape analysis). Until the real detector is
integrated, the stub below lets the end-to-end pipeline run: it treats the
whole input image as one sign candidate, which is correct for GTSRB images
because they are already cropped around a single sign.

Integration contract (must stay unchanged when the stub is replaced):
    * input:  one image as a BGR ``uint8`` NumPy array (as from ``cv2.imread``)
    * output: list of boxes ``(x, y, w, h)`` in pixels, integers, origin at
      the top-left corner, fully inside the image; empty list = no sign found
"""

import numpy as np


def detect_signs(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return bounding boxes ``(x, y, w, h)`` of candidate traffic signs.

    Stub behaviour: one box covering the entire image.
    """
    height, width = image_bgr.shape[:2]
    return [(0, 0, width, height)]
