"""Separate localization, matched-crop accuracy and joint system metrics."""
from collections import Counter

from evaluate_detection import evaluate, match_detections
from src.pipeline import CLASS_IDS, UNKNOWN


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate_system(predictions, labels, min_iou=0.5):
    """One fixed class-blind greedy IoU assignment, then class evaluation.

    All raw candidates, including rejections, participate in the assignment.
    Wrong labels count as FP of the predicted class and FN of the true class.
    Rejections count as FN but not as an accepted prediction/FP.
    """
    if not predictions or set(predictions) != set(labels):
        raise ValueError("Predictions and labels must cover exactly the same nonempty image set.")
    for records in labels.values():
        if any(type(r["class_id"]) is not int or r["class_id"] not in CLASS_IDS for r in records):
            raise ValueError("Ground truth requires integer class IDs 0..42.")
    for records in predictions.values():
        if any(type(r["class_id"]) is not int or r["class_id"] not in (*CLASS_IDS, UNKNOWN)
               for r in records):
            raise ValueError("Predictions require class IDs 0..42 or UNKNOWN=-1.")
    detection = evaluate(predictions, labels, min_iou)
    detection.pop("hinweis", None)
    detection.pop("per_image", None)
    # Rows: true 0..42 plus background. Columns: predicted 0..42, unknown, missed.
    matrix = [[0] * 45 for _ in range(44)]
    support, predicted, correct = Counter(), Counter(), Counter()
    matches_count = correct_count = accepted_matched = rejected = 0
    for name, boxes in predictions.items():
        ground = labels[name]
        support.update(b["class_id"] for b in ground)
        predicted.update(b["class_id"] for b in boxes if b["class_id"] != UNKNOWN)
        rejected += sum(b["class_id"] == UNKNOWN for b in boxes)
        matches = match_detections(boxes, ground, min_iou)
        matched_pred = {i for _, i, _ in matches}
        matched_gt = {j for _, _, j in matches}
        matches_count += len(matches)
        for _, i, j in matches:
            actual, guess = ground[j]["class_id"], boxes[i]["class_id"]
            matrix[actual][43 if guess == UNKNOWN else guess] += 1
            accepted_matched += guess != UNKNOWN
            if actual == guess:
                correct[actual] += 1
                correct_count += 1
        for i, b in enumerate(boxes):
            if i not in matched_pred:
                guess = b["class_id"]
                matrix[43][43 if guess == UNKNOWN else guess] += 1
        for j, b in enumerate(ground):
            if j not in matched_gt:
                matrix[b["class_id"]][44] += 1
    accepted = sum(predicted.values())
    total_gt = sum(support.values())
    per_class = {}
    for c in CLASS_IDS:
        tp, fp, fn = correct[c], predicted[c] - correct[c], support[c] - correct[c]
        per_class[str(c)] = {
            "support": support[c], "accepted_predictions": predicted[c],
            "TP": tp, "FP": fp, "FN": fn,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
        }
    return {
        "detection": detection,
        "classification_on_matched_candidates": {
            "matched": matches_count, "correct": correct_count,
            "accuracy_including_unknown": ratio(correct_count, matches_count),
            "accepted_matched": accepted_matched,
            "accuracy_on_accepted_only": ratio(correct_count, accepted_matched),
        },
        "end_to_end": {
            "TP": correct_count, "FP": accepted - correct_count, "FN": total_gt - correct_count,
            "precision": ratio(correct_count, accepted), "recall": ratio(correct_count, total_gt),
            "f1": ratio(2 * correct_count, accepted + total_gt),
            "accepted_predictions": accepted, "rejected_candidates": rejected,
        },
        "per_class": per_class,
        "confusion": {
            "rows": list(CLASS_IDS) + ["background"],
            "columns": list(CLASS_IDS) + ["unknown", "missed"],
            "counts": matrix,
        },
        "matching": "fixed_class_blind_greedy_descending_iou_one_to_one_before_rejection",
        "undefined_ratios": "null (zero denominator)",
    }
