"""Run localization or the complete MVSR pipeline on TRAIN street images."""
import argparse
from pathlib import Path

import cv2

from src.mvsr_dataset import development_path, EXTENSIONS
from src.mvsr_kl.files import load_image, save_image, write_json
from src.mvsr_kl.localizer import draw_detections, find_sign_candidates
from src.pipeline import detect_and_classify, validate_classifier

PROJECT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--mode", choices=("strict", "sensitive"), default="strict")
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/inference")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", type=Path, help="Trusted local SignClassifier joblib file")
    parser.add_argument("--allow-partial-classes", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0.")
    if args.model and args.mode != "strict":
        parser.error("Integrated detect_signs() uses the fixed strict detector.")
    try:
        image_dir = development_path(args.dataset / args.split / "images")
        output = development_path(args.output)
        classifier = None
        if args.model:
            from src.classification import SignClassifier
            classifier = SignClassifier.load(development_path(args.model))
            validate_classifier(classifier, args.allow_partial_classes)
    except ValueError as error:
        parser.error(str(error))
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if args.limit:
        images = images[:args.limit]
    if not images:
        parser.error("No images found.")
    predictions = {}
    for index, path in enumerate(images, 1):
        image = load_image(development_path(path))
        if classifier is None:
            detections = find_sign_candidates(image, mode=args.mode)
            records = [d.to_dict() for d in detections]
            annotated = draw_detections(image, detections)
        else:
            records = detect_and_classify(image, classifier)
            annotated = image.copy()
            for b in records:
                color = (0, 180, 0) if b["class_id"] != -1 else (0, 165, 255)
                cv2.rectangle(annotated, (b["x1"], b["y1"]), (b["x2"] - 1, b["y2"] - 1), color, 2)
                text = f"class={b['class_id']} score={b['classification_confidence']:.3f}"
                cv2.putText(annotated, text, (b["x1"], max(12, b["y1"] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        predictions[path.name] = records
        save_image(output / "annotated" / f"{path.stem}.jpg", annotated)
        print(f"[{index}/{len(images)}] {path.name}: {len(records)} candidates", flush=True)
    write_json(output / "detections.json", predictions)
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
