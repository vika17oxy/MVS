"""Pixelbasierte Verkehrszeichenlokalisierung mit klassischen Formvorlagen."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import cv2
import numpy as np

from .preprocess import improve_contrast, segment_colors


@dataclass(frozen=True)
class Detection:
    """Box (x1, y1, x2, y2); rechte/untere Grenze exklusiv, BGR-Pixel."""

    x1: int
    y1: int
    x2: int
    y2: int
    color: str
    shape: str
    confidence: float  # Heuristischer Qualitätswert, keine Wahrscheinlichkeit.
    class_id: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def intersection_over_union(a: tuple[int, int, int, int],
                            b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    if overlap <= 0:
        return 0.0
    size_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    size_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return overlap / (size_a + size_b - overlap)


@lru_cache(maxsize=1)
def _shape_templates() -> tuple[tuple[str, np.ndarray], ...]:
    """Analytische Silhouetten, keine aus Datensatz-Crops gelernten Vorlagen."""
    templates = []
    circle = np.zeros((32, 32), np.uint8)
    cv2.ellipse(circle, (31, 31), (31, 31), 0, 0, 360, 1, -1, shift=1)
    templates.append(("circle", circle.astype(bool)))
    polygons = [
        ("triangle", [(16, 0), (31, 31), (0, 31)]),
        ("triangle", [(0, 0), (31, 0), (16, 31)]),
        ("quadrilateral", [(16, 0), (31, 16), (16, 31), (0, 16)]),
        ("rectangle", [(0, 0), (31, 0), (31, 31), (0, 31)]),
        ("octagon", [(9, 0), (22, 0), (31, 9), (31, 22),
                     (22, 31), (9, 31), (0, 22), (0, 9)]),
    ]
    for name, points in polygons:
        mask = np.zeros((32, 32), np.uint8)
        cv2.fillPoly(mask, [np.array(points, np.int32)], 1)
        templates.append((name, mask.astype(bool)))
    return tuple(templates)


def _match_shape(contour: np.ndarray) -> tuple[str, float]:
    """Konvexe Hülle toleriert kleine Lücken und teilweise verdeckte Ränder."""
    x, y, w, h = cv2.boundingRect(contour)
    points = contour.astype(np.float32) - (x, y)
    points *= (31 / max(1, w - 1), 31 / max(1, h - 1))
    mask = np.zeros((32, 32), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points.astype(np.int32)), 1)
    mask = mask.astype(bool)
    scores = [(name, np.count_nonzero(mask & template)
               / max(1, np.count_nonzero(mask | template)))
              for name, template in _shape_templates()]
    return max(scores, key=lambda pair: pair[1])


def _color_contrast(mask: np.ndarray, x: int, y: int,
                    w: int, h: int) -> tuple[float, float]:
    height, width = mask.shape
    foreground = np.count_nonzero(mask[y:y + h, x:x + w])
    px, py = max(2, round(.25 * w)), max(2, round(.25 * h))
    a, b = max(0, x - px), max(0, y - py)
    c, d = min(width, x + w + px), min(height, y + h + py)
    outside = np.count_nonzero(mask[b:d, a:c]) - foreground
    fraction = foreground / (w * h)
    contrast = fraction - outside / max(1, (c - a) * (d - b) - w * h)
    return float(fraction), float(contrast)


def _light_interior(hsv: np.ndarray, mask: np.ndarray,
                    box: tuple[int, int, int, int]) -> float:
    """Helle, wenig gesättigte Zeichenfläche relativ zum farbigen Rand."""
    x, y, w, h = box
    crop = hsv[y:y + h, x:x + w]
    colored = mask[y:y + h, x:x + w] > 0
    value = crop[:, :, 2]
    reference = np.median(value[colored]) if np.any(colored) else 255
    light = (crop[:, :, 1] < 125) & ~colored & (
        (value > reference + 15)
        | ((crop[:, :, 1] < 60) & (value >= .85 * reference)))
    dx, dy = max(1, w // 5), max(1, h // 5)
    return float(np.mean(light[dy:h - dy, dx:w - dx]))


def _border_support(mask: np.ndarray, contour: np.ndarray) -> float:
    """Zielfarbe muss über den äußeren Rand verteilt sein, nicht nur innen."""
    x, y, w, h = cv2.boundingRect(contour)
    points = contour.astype(np.float32) - (x, y)
    points *= (31 / max(1, w - 1), 31 / max(1, h - 1))
    silhouette = np.zeros((32, 32), np.uint8)
    cv2.fillConvexPoly(silhouette, cv2.convexHull(points.astype(np.int32)), 1)
    inner = cv2.erode(silhouette, np.ones((9, 9), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
    ring = (silhouette > 0) & (inner == 0)
    color = cv2.resize(mask[y:y + h, x:x + w], (32, 32)) > 127
    return float(np.count_nonzero(ring & color) / max(1, np.count_nonzero(ring)))


def _mask_proposals(raw: np.ndarray,
                    corrected: np.ndarray) -> list[tuple[np.ndarray, int]]:
    """Zwei Lückenskalen plus farbtreuer, kontrastierter und entstörter Zweig.

    Opening ist nur ein zusätzlicher Vorschlag: dünne Ränder bleiben im
    Originalzweig erhalten. Closing mit 7 Pixeln verbindet unterbrochene Ränder.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    return [(raw, 3), (raw, 7), (corrected, 3), (opened, 3)]


def _suppress_overlaps(items: list[Detection],
                       threshold: float = .25) -> list[Detection]:
    """Farben-/Zweigübergreifende NMS; Nachbarzeichen werden nicht vereinigt."""
    results: list[Detection] = []
    for item in sorted(items, key=lambda d: d.confidence, reverse=True):
        box = (item.x1, item.y1, item.x2, item.y2)
        if all(intersection_over_union(box, (d.x1, d.y1, d.x2, d.y2))
               <= threshold for d in results):
            results.append(item)
    return results


def find_sign_candidates(image: np.ndarray,
                         mode: str = "strict") -> list[Detection]:
    """Automatische BGR-Inferenz ohne Dateinamen, Labels oder feste Positionen.

    ``strict`` prüft Form und Farbverteilung strenger. ``sensitive`` liefert
    zusätzliche Vorschläge für einen nachgeschalteten Klassifikator. Beide
    Modi geben ausschließlich geometrische und photometrische Heuristiken aus.
    """
    if image is None or image.size == 0:
        raise ValueError("Das Eingabebild ist leer oder konnte nicht gelesen werden.")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Erwartet wird ein BGR-Bild mit drei uint8-Kanälen.")
    if mode not in ("strict", "sensitive"):
        raise ValueError("mode muss 'strict' oder 'sensitive' sein")

    return _find_candidates(image, mode)


def _find_candidates(image: np.ndarray, mode: str) -> list[Detection]:
    height, width = image.shape[:2]
    masks = segment_colors(image)
    corrected = improve_contrast(image)
    corrected_masks = segment_colors(corrected)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    candidates = []
    for color, mask in masks.items():
        for source, kernel_size in _mask_proposals(mask, corrected_masks[color]):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (kernel_size, kernel_size))
            closed = cv2.morphologyEx(source, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if min(w, h) < 13 or max(w, h) > .5 * min(width, height):
                    continue
                if not .60 <= w / h <= 1.55:
                    continue
                area = cv2.contourArea(contour)
                hull_area = cv2.contourArea(cv2.convexHull(contour))
                if area < 30 or area / max(1, hull_area) < .65:
                    continue
                shape, fit = _match_shape(contour)
                min_fit = .73 if shape == "triangle" else .84
                if color == "yellow":
                    min_fit = .78
                if fit < (min_fit if mode == "strict" else .70):
                    continue
                fraction, contrast = _color_contrast(source, x, y, w, h)
                if fraction < .16 or contrast < .08:
                    continue
                border = _border_support(source, contour)
                if border < (.45 if mode == "strict" else .3):
                    continue
                interior = _light_interior(hsv, source, (x, y, w, h))
                if color == "yellow":
                    if shape != "quadrilateral" or fraction < .35:
                        continue
                else:
                    if interior < (.12 if mode == "strict" else .04):
                        continue
                    if color == "red" and shape in ("rectangle", "quadrilateral"):
                        continue
                    if color == "blue" and shape == "triangle":
                        continue
                    if color == "blue" and fraction < .4:
                        continue
                padding = round((.4 if color == "yellow" else .03) * max(w, h))
                score = .55 * fit + .25 * min(1., interior * 3) + .20 * contrast
                candidates.append(Detection(
                    max(0, x - padding), max(0, y - padding),
                    min(width, x + w + padding), min(height, y + h + padding),
                    color, shape, round(float(score), 4),
                ))
    return _suppress_overlaps(candidates)


def draw_detections(image: np.ndarray,
                    detections: list[Detection]) -> np.ndarray:
    annotated = image.copy()
    for d in detections:
        cv2.rectangle(annotated, (d.x1, d.y1), (d.x2 - 1, d.y2 - 1),
                      (0, 255, 0), 2)
        name = f"{d.color}/{d.shape} {d.confidence:.2f}"
        if d.class_id is not None:
            name += f" class={d.class_id}"
        cv2.putText(annotated, name, (d.x1, max(12, d.y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .40, (0, 230, 0), 1, cv2.LINE_AA)
    return annotated
