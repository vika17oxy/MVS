"""Deterministische Bildvorverarbeitung ohne neuronale Modelle."""
from __future__ import annotations

import cv2
import numpy as np


def improve_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE nur im Helligkeitskanal des LAB-Farbraums anwenden.

    Lokale Histogramme (8 x 8 Kacheln), begrenzt gegen Rauschverstärkung.
    LAB-Rückwandlung kann Hue/Sättigung verschieben; deshalb verwendet der
    Lokalisierer zusätzlich die unveränderten Originalfarben.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    corrected = cv2.merge((clahe.apply(lightness), channel_a, channel_b))
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)


def segment_colors(image: np.ndarray) -> dict[str, np.ndarray]:
    """HSV-Masken erzeugen (OpenCV-Hue 0..179, S/V 0..255).

    Rot umfasst beide Enden des Hue-Kreises. Nur bei niedriger Helligkeit
    wird der Bereich Richtung Violett erweitert (blauer Farbstich im Schatten).
    Nahezu schwarze Pixel bleiben ausgeschlossen, da ihr Hue instabil ist.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_a = cv2.inRange(hsv, (0, 65, 25), (12, 255, 255))
    red_b = cv2.inRange(hsv, (165, 60, 25), (179, 255, 255))
    shadow_red = cv2.inRange(hsv, (153, 55, 20), (179, 255, 110))
    return {
        "red": red_a | red_b | shadow_red,
        "blue": cv2.inRange(hsv, (95, 65, 25), (130, 255, 255)),
        "yellow": cv2.inRange(hsv, (12, 90, 50), (38, 255, 255)),
    }
