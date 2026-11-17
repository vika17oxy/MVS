"""Reproduzierbare TRAIN-Messung, Ablationen und TP/FP/FN-Visualisierung.

Es wird ausschließlich dataset/train geöffnet. Labels fließen nur in Bewertung
und Visualisierung ein, nie in die Funktion find_sign_candidates.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib
import platform
from pathlib import Path
import sys
import time
import types
from unittest.mock import patch

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))
from mvsr_kl import localizer  # noqa: E402
from mvsr_kl.files import load_image, read_labels, save_image, write_json  # noqa: E402
from evaluate_detection import evaluate, match_detections  # noqa: E402


def load_snapshot(source: Path):
    """Archivierte Baseline als separates Paket laden, Arbeitscode bleibt gleich."""
    package = types.ModuleType("benchmark_baseline")
    package.__path__ = [str(source.resolve())]
    sys.modules[package.__name__] = package
    return importlib.import_module(package.__name__ + ".localizer")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transformed(image: np.ndarray, variant: str) -> np.ndarray:
    if variant == "dark_060":
        return np.rint(image.astype(np.float32) * .6).astype(np.uint8)
    if variant == "blur_sigma1":
        return cv2.GaussianBlur(image, (5, 5), 1)
    if variant == "resize_075":
        return cv2.resize(image, None, fx=.75, fy=.75,
                          interpolation=cv2.INTER_AREA)
    return image


def annotation_view(image: np.ndarray, boxes: list[dict],
                    ground: list[dict]) -> tuple[np.ndarray, list]:
    """Grün=TP, Rot=FP, Orange=FN; blaue dünne Rahmen zeigen zugeordnete GT."""
    matches = match_detections(boxes, ground)
    matched_pred = {i for _, i, _ in matches}
    matched_gt = {j for _, _, j in matches}
    view = image.copy()
    records = []
    for j, box in enumerate(ground):
        color = (255, 130, 0) if j in matched_gt else (0, 165, 255)
        cv2.rectangle(view, (box["x1"], box["y1"]),
                      (box["x2"] - 1, box["y2"] - 1), color, 1)
        if j not in matched_gt:
            records.append(("FN", box, color))
    for i, box in enumerate(boxes):
        kind = "TP" if i in matched_pred else "FP"
        color = (0, 200, 0) if kind == "TP" else (0, 0, 255)
        cv2.rectangle(view, (box["x1"], box["y1"]),
                      (box["x2"] - 1, box["y2"] - 1), color, 2)
        records.append((kind, box, color))
    return view, records


def save_examples(paths: list[Path], predictions: dict,
                  labels: dict, output: Path) -> None:
    """Je Kategorie die ersten 12 Fälle in Dateireihenfolge, keine Schönwahl."""
    tiles: dict[str, list] = {"TP": [], "FP": [], "FN": []}
    example_records = []
    for index, path in enumerate(paths):
        image = load_image(path)
        view, records = annotation_view(image, predictions[path.name],
                                        labels.get(path.name, []))
        if index < 10:
            save_image(output / "annotated" / f"{path.stem}.jpg", view)
        for kind, box, color in records:
            if len(tiles[kind]) >= 12:
                continue
            x, y, u, v = [box[k] for k in ("x1", "y1", "x2", "y2")]
            pad = max(12, (max(u - x, v - y)) // 3)
            crop = view[max(0, y - pad):v + pad, max(0, x - pad):u + pad]
            factor = min(220 / crop.shape[1], 170 / crop.shape[0])
            crop = cv2.resize(crop, None, fx=factor, fy=factor,
                              interpolation=cv2.INTER_NEAREST)
            tile = np.full((210, 240, 3), 245, np.uint8)
            tile[35:35 + crop.shape[0], 5:5 + crop.shape[1]] = crop
            cv2.putText(tile, f"{kind} {path.name}", (5, 16), 0, .4, color, 1)
            cv2.putText(tile, f"({x},{y})-({u},{v})", (5, 31), 0, .35, (0, 0, 0), 1)
            tiles[kind].append(tile)
            example_records.append({"filename": path.name, "kind": kind, "box": box})
        if index >= 9 and all(len(items) == 12 for items in tiles.values()):
            break
    for kind, items in tiles.items():
        if not items:
            continue
        sheet = np.full((((len(items) + 3) // 4) * 210, 960, 3), 255, np.uint8)
        for i, tile in enumerate(items):
            sheet[i // 4 * 210:(i // 4 + 1) * 210,
                  i % 4 * 240:(i % 4 + 1) * 240] = tile
        save_image(output / f"examples_{kind}.jpg", sheet)
    write_json(output / "examples.json", example_records)


def run_variant(paths, labels, detector, mode, transform, output, visualize):
    predictions, adjusted_labels, timings = {}, {}, []
    for index, path in enumerate(paths, 1):
        original = load_image(path)
        image = transformed(original, transform)
        start = time.perf_counter()
        predictions[path.name] = [d.to_dict() for d in detector(image, mode=mode)]
        timings.append(time.perf_counter() - start)
        sx, sy = image.shape[1] / original.shape[1], image.shape[0] / original.shape[0]
        adjusted_labels[path.name] = [dict(
            b, x1=round(b["x1"] * sx), x2=round(b["x2"] * sx),
            y1=round(b["y1"] * sy), y2=round(b["y2"] * sy),
        ) for b in labels.get(path.name, [])]
        if index % 100 == 0:
            print(f"{output.name}: {index}/{len(paths)}", flush=True)
    metrics = evaluate(predictions, adjusted_labels, .5)
    metrics["timing_seconds"] = {
        "inference_total": round(sum(timings), 4),
        "median_per_image": round(float(np.median(timings)), 5),
        "p95_per_image": round(float(np.percentile(timings, 95)), 5),
    }
    metrics["mode"] = mode
    metrics["transform"] = transform
    write_json(output / "detections.json", predictions)
    write_json(output / "detection_metrics.json", metrics)
    if visualize and transform == "original":
        save_examples(paths, predictions, labels, output)
    print(f"{output.name}: TP={metrics['true_positive']} "
          f"FP={metrics['false_positive']} FN={metrics['false_negative']} "
          f"P={metrics['detection_precision']:.5f} "
          f"R={metrics['detection_recall']:.5f}", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT / "sample_dataset")
    parser.add_argument("--output", type=Path, default=PROJECT / "results/benchmark")
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--robustness", action="store_true")
    args = parser.parse_args()
    train = args.dataset / "train"
    if not (train / "labels.csv").is_file() or not (train / "images").is_dir():
        parser.error("Erwartet werden train/images und train/labels.csv.")
    if "heldout" in {part.lower() for part in train.resolve().parts}:
        parser.error("Dieser Entwicklungsbenchmark darf heldout nicht verwenden.")
    paths = sorted(p for p in (train / "images").iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not paths:
        parser.error("Keine Trainingsbilder gefunden.")
    labels = read_labels(train / "labels.csv")
    metadata = {
        "python": platform.python_version(), "opencv": cv2.__version__,
        "numpy": np.__version__, "platform": platform.platform(),
        "dataset_train": str(train.resolve()), "iou_threshold": .5,
        "opencv_threads": cv2.getNumThreads(),
        "label_sha256": file_hash(train / "labels.csv"),
        "images_sha256": {p.name: file_hash(p) for p in paths},
        "source_sha256": {p.name: file_hash(p) for p in (
            PROJECT / "src/mvsr_kl/localizer.py", PROJECT / "src/mvsr_kl/preprocess.py",
            PROJECT / "evaluate_detection.py", Path(__file__),
        )},
    }
    models = [("final", localizer)]
    if args.baseline_source:
        models.insert(0, ("baseline", load_snapshot(args.baseline_source)))
        metadata["baseline_source"] = {
            p.name: file_hash(p) for p in args.baseline_source.glob("*.py")}
    write_json(args.output / "manifest.json", metadata)
    summary = {}
    for name, module in models:
        variants = ["original"]
        if args.robustness:
            variants += ["dark_060", "blur_sigma1", "resize_075"]
        for variant in variants:
            key = name if variant == "original" else f"{name}_{variant}"
            metrics = run_variant(paths, labels, module.find_sign_candidates,
                                  "strict", variant, args.output / key, True)
            summary[key] = {k: v for k, v in metrics.items() if k != "per_image"}
    if args.ablations:
        proposals = localizer._mask_proposals
        nms = localizer._suppress_overlaps
        experiments = [
            ("sensitive", "sensitive", nullcontext()),
            ("without_clahe", "strict", patch.object(
                localizer, "improve_contrast", lambda image: image)),
            ("without_large_closing", "strict", patch.object(
                localizer, "_mask_proposals", lambda a, b: [
                    item for item in proposals(a, b) if item[1] != 7])),
            ("without_opening", "strict", patch.object(
                localizer, "_mask_proposals", lambda a, b: proposals(a, b)[:3])),
            ("without_border_check", "strict", patch.object(
                localizer, "_border_support", lambda mask, contour: 1.)),
            ("nms_045", "strict", patch.object(
                localizer, "_suppress_overlaps", lambda items: nms(items, .45))),
        ]
        for name, mode, context in experiments:
            with context:
                metrics = run_variant(paths, labels, localizer.find_sign_candidates,
                                      mode, "original", args.output / name, False)
            summary[name] = {k: v for k, v in metrics.items() if k != "per_image"}
    write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
