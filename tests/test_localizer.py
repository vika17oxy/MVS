"""Ausführbare Grundtests ohne die großen Originaldaten."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mvsr_kl.localizer import intersection_over_union, find_sign_candidates
from mvsr_kl.preprocess import segment_colors


def test_iou_identical():
    assert intersection_over_union((1, 2, 11, 12), (1, 2, 11, 12)) == 1.0


def test_iou_disjoint():
    assert intersection_over_union((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_colors_detect_red_circle():
    image = np.full((200, 250, 3), 255, np.uint8)
    cv2.circle(image, (100, 100), 25, (0, 0, 255), thickness=7)
    assert np.count_nonzero(segment_colors(image)["red"]) > 0
    assert len(find_sign_candidates(image)) >= 1


def test_blank_image():
    image = np.full((180, 250, 3), 245, dtype=np.uint8)
    assert find_sign_candidates(image) == []
