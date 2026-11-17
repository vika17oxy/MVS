"""TRAIN-Fehleranalyse gespeicherter Vorhersagen, unabhängig von der Inferenz."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))
from evaluate_detection import match_detections  # noqa: E402
from mvsr_kl.files import read_labels, write_json  # noqa: E402
from mvsr_kl.localizer import intersection_over_union  # noqa: E402


def breakdown(predictions: dict, labels: dict) -> dict:
    sizes, classes, misses = {}, {}, []
    negative_images = negative_fp = 0
    for name, boxes in sorted(predictions.items()):
        ground = labels.get(name, [])
        matches = match_detections(boxes, ground)
        matched_gt = {j for _, _, j in matches}
        if not ground:
            negative_images += 1
            negative_fp += len(boxes)
        for j, target in enumerate(ground):
            box = tuple(target[k] for k in ("x1", "y1", "x2", "y2"))
            size = min(box[2] - box[0], box[3] - box[1])
            band = "<24px" if size < 24 else "24-47px" if size < 48 else ">=48px"
            for groups, key in ((sizes, band), (classes, str(target["class_id"]))):
                group = groups.setdefault(key, {"ground_truth": 0, "TP": 0, "FN": 0})
                group["ground_truth"] += 1
                group["TP" if j in matched_gt else "FN"] += 1
            if j not in matched_gt:
                overlap = max((intersection_over_union(box, tuple(
                    p[k] for k in ("x1", "y1", "x2", "y2"))) for p in boxes), default=0.)
                misses.append({"filename": name, "ground_truth": target,
                               "best_prediction_iou": round(overlap, 5)})
    for groups in (sizes, classes):
        for group in groups.values():
            group["localization_recall"] = round(group["TP"] / group["ground_truth"], 5)
    return {"by_minimum_gt_side": sizes, "by_ground_truth_class": classes,
            "negative_images": negative_images,
            "false_positives_on_negative_images": negative_fp,
            "misses": misses,
            "note": "Referenzklassen gruppieren nur den Lokalisierungs-Recall; "
                    "keine vorhergesagten Klassen, keine Klassen-Accuracy."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT / "sample_dataset")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = args.dataset / "train"
    if "heldout" in {part.lower() for part in train.resolve().parts}:
        parser.error("Dieser Entwicklungsbericht darf heldout nicht verwenden.")
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    result = breakdown(predictions, read_labels(train / "labels.csv"))
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "misses"}, indent=2))


if __name__ == "__main__":
    main()
