"""Separate Qualitätsmessung der Lokalisierung (ohne Klassifikation)."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))
from mvsr_kl.files import read_labels, write_json  # noqa: E402
from mvsr_kl.localizer import intersection_over_union  # noqa: E402


def _bbox(record: dict) -> tuple:
    box = tuple(record[k] for k in ("x1", "y1", "x2", "y2"))
    if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
            or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError(f"Ungültige Bounding Box: {box}")
    return box


def match_detections(boxes: list[dict], ground: list[dict],
                     min_iou: float = .5) -> list[tuple[float, int, int]]:
    """Wie die Baseline: greedy nach absteigender IoU, jede Box höchstens einmal."""
    if not 0 < min_iou <= 1:
        raise ValueError("IoU-Schwelle muss im Intervall (0, 1] liegen.")
    predicted_boxes = [_bbox(b) for b in boxes]
    ground_boxes = [_bbox(b) for b in ground]
    pairs = []
    for i, predicted in enumerate(predicted_boxes):
        for j, target in enumerate(ground_boxes):
            overlap = intersection_over_union(predicted, target)
            if overlap >= min_iou:
                pairs.append((overlap, i, j))
    matched_predictions, matched_ground = set(), set()
    matches = []
    for overlap, i, j in sorted(pairs, reverse=True):
        if i not in matched_predictions and j not in matched_ground:
            matched_predictions.add(i)
            matched_ground.add(j)
            matches.append((overlap, i, j))
    return matches


def evaluation_scope(predictions: dict, labels: dict, filenames: set[str],
                     allow_partial: bool = False) -> dict:
    """Unvollständige Läufe dürfen die Messmenge nicht unbemerkt verkleinern."""
    unknown = set(predictions) - filenames
    missing = filenames - set(predictions)
    if unknown:
        raise ValueError(f"Vorhersagen für unbekannte Bilder: {sorted(unknown)[:5]}")
    if missing and not allow_partial:
        raise ValueError(f"Vorhersagen für {len(missing)} Bilder fehlen. "
                         "Für bewusst begrenzte Läufe --allow-partial angeben.")
    if not predictions:
        raise ValueError("Die Vorhersagedatei enthält keine Bilder.")
    return {name: labels.get(name, []) for name in predictions}


def evaluate(predictions: dict, labels: dict, min_iou: float) -> dict:
    if not 0 < min_iou <= 1:
        raise ValueError("IoU-Schwelle muss im Intervall (0, 1] liegen.")
    true_positive = false_positive = false_negative = 0
    detail: dict[str, dict[str, int]] = {}
    for filename in sorted(set(predictions) | set(labels)):
        boxes = predictions.get(filename, [])
        ground = labels.get(filename, [])
        tp = len(match_detections(boxes, ground, min_iou))
        fp, fn = len(boxes) - tp, len(ground) - tp
        true_positive += tp
        false_positive += fp
        false_negative += fn
        detail[filename] = {"TP": tp, "FP": fp, "FN": fn}

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {"iou_threshold": min_iou,
            "true_positive": true_positive, "false_positive": false_positive,
            "false_negative": false_negative, "detection_precision": round(precision, 5),
            "detection_recall": round(recall, 5), "detection_f1": round(f1, 5),
            "image_count": len(detail),
            "ground_truth_count": true_positive + false_negative,
            "matching": "greedy_descending_iou_one_to_one",
            "per_image": detail,
            "hinweis": "Nur Lokalisierungsmetriken; class_id wird ignoriert. "
                       "Klassenmetriken und Confusion Matrix liefert evaluate_mvsr.py."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluierung der Verkehrszeichen-Lokalisierung")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--predictions", type=Path, default=PROJECT / "artifacts/inference/detections.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/inference/detection_metrics.json")
    parser.add_argument("--iou", type=float, default=.5)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Bewusst nur die Bilder der Vorhersagedatei bewerten")
    args = parser.parse_args()
    from src.mvsr_dataset import development_path
    development_path(args.dataset)
    development_path(args.dataset / args.split / "images")
    development_path(args.output)
    labels_path = development_path(args.dataset / args.split / "labels.csv")
    if not labels_path.is_file():
        parser.error(f"Ground-Truth-Datei nicht gefunden: {labels_path}")
    if not args.predictions.is_file():
        parser.error("Keine Vorhersagen gefunden. Zuerst 'python run.py' ausführen.")
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    labels = read_labels(labels_path)
    filenames = {p.name for p in (args.dataset / args.split / "images").iterdir()
                 if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    try:
        labels = evaluation_scope(predictions, labels, filenames, args.allow_partial)
    except ValueError as error:
        parser.error(str(error))
    metrics = evaluate(predictions, labels, args.iou)
    metrics["available_images"] = len(filenames)
    metrics["partial_evaluation"] = len(predictions) != len(filenames)
    write_json(args.output, metrics)
    print(f"IoU >= {args.iou}: TP={metrics['true_positive']}, FP={metrics['false_positive']}, FN={metrics['false_negative']}")
    print(f"Detection Precision={metrics['detection_precision']:.3f}, Detection Recall={metrics['detection_recall']:.3f}")
    print(metrics["hinweis"])


if __name__ == "__main__":
    main()
