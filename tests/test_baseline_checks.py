"""Die vier ursprünglichen Funktionstests auch über unittest ausführen."""
import unittest
from test_localizer import (
    test_blank_image, test_colors_detect_red_circle, test_iou_disjoint,
    test_iou_identical,
)


class OriginalTests(unittest.TestCase):
    test_blank = staticmethod(test_blank_image)
    test_red_circle = staticmethod(test_colors_detect_red_circle)
    test_disjoint = staticmethod(test_iou_disjoint)
    test_identical = staticmethod(test_iou_identical)
