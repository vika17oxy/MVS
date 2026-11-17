"""Ali's classical detector, preserving Viktoriia's BGR -> xywh contract."""
import numpy as np

from src.mvsr_kl.localizer import find_sign_candidates


def detect_signs(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return all strict-mode candidates as (x, y, width, height).

    Ali's x2/y2 are exclusive: width=x2-x1, height=y2-y1; no +1.
    Blank images return []; invalid inputs raise ValueError.
    """
    return [(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1)
            for d in find_sign_candidates(image_bgr)]
