"""Ein- und Ausgabe von Bildern, CSV-Dateien und JSON."""
from __future__ import annotations
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def load_image(path: Path) -> np.ndarray:
    """Unicode-kompatibles Einlesen (auch Windows-Pfade mit Umlauten)."""
    pixels = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(pixels, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Bild konnte nicht geöffnet werden: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise OSError(f"Bild konnte nicht gespeichert werden: {path}")
    encoded.tofile(str(path))


def read_labels(path: Path) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"filename", "x1", "y1", "x2", "y2", "class_id"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Fehlende CSV-Spalten in {path}: {required}")
        for row in reader:
            record = {name: int(row[name]) for name in ("x1", "y1", "x2", "y2", "class_id")}
            records.setdefault(row["filename"], []).append(record)
    return records


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
