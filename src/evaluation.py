"""Metrics, plots and timing for comparing the classifiers.

Metrics (all computed on the held-out test set with ``sklearn.metrics``):

* **Accuracy**  = correct predictions / all predictions.
* **Precision** of class c = TP / (TP + FP): of all images *predicted* as c,
  how many really are c? (low precision = class c is "over-predicted").
* **Recall** of class c = TP / (TP + FN): of all images that *really are* c,
  how many were found? (low recall = class c is often missed).
* **F1** = harmonic mean 2*P*R / (P + R); high only if both P and R are high.
* **Macro average**: mean over classes, every class counts equally.
  **Weighted average**: mean weighted by the number of test images per class.

"Unknown" predictions (confidence below threshold) count as errors: they
lower the accuracy and the recall of the true class, but they are not a
false positive for any real class, so they do not lower precision.
"""

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.classification import UNKNOWN
from src.dataset import CLASS_NAMES

# Non-interactive backend: plots are only written to PNG files, so no window
# (and no GUI toolkit) is needed, e.g. when running on a server.
plt.switch_backend("Agg")

# Colours for the plots: a light chart surface, dark text, a single-hue
# blue ramp for magnitudes (confusion matrix) and two clearly different
# hues for the two compared methods.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
BLUE_RAMP = ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SERIES_COLORS = ["#2a78d6", "#eb6834"]


def class_label(class_id: int) -> str:
    """Readable name for a class id, including the "unknown" pseudo class."""
    return "unknown" if class_id == UNKNOWN else CLASS_NAMES[class_id]


def compute_metrics(y_true, y_pred, class_ids) -> dict:
    """Overall and per-class metrics of one classifier.

    Returns a dict with ``accuracy``, ``unknown_rate``, macro and weighted
    precision/recall/F1 and a ``per_class`` dict (class id -> metrics).
    """
    # labels=class_ids restricts the report to the real classes, so
    # "unknown" never appears as a class of its own (see module docstring).
    report = classification_report(y_true, y_pred, labels=class_ids,
                                   output_dict=True, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "unknown_rate": float(np.mean(np.asarray(y_pred) == UNKNOWN)),
    }
    for average in ("macro avg", "weighted avg"):
        prefix = average.split()[0]
        for metric in ("precision", "recall", "f1-score"):
            metrics[f"{prefix}_{metric.replace('-score', '')}"] = report[average][metric]
    metrics["per_class"] = {
        class_id: {
            "name": CLASS_NAMES[class_id],
            "precision": report[str(class_id)]["precision"],
            "recall": report[str(class_id)]["recall"],
            "f1": report[str(class_id)]["f1-score"],
            "support": int(report[str(class_id)]["support"]),
        }
        for class_id in class_ids
    }
    return metrics


def measure_inference_time(classifier, images, n_images: int = 200) -> float:
    """Average time in milliseconds to classify one ROI with ``predict``.

    Single-image calls are timed on purpose (not ``predict_batch``), because
    that is how the pipeline uses the classifier: one ROI at a time.
    The first call is excluded as warm-up (lazy initialisation, caches).
    """
    classifier.predict(images[0])
    start = time.perf_counter()
    for image in images[:n_images]:
        classifier.predict(image)
    return (time.perf_counter() - start) / min(n_images, len(images)) * 1000


def _style_axes(ax) -> None:
    """Recessive axes: no frame, muted tick labels, no tick marks."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=TEXT_MUTED, length=0)


def plot_confusion_matrix(y_true, y_pred, class_ids, title: str, path) -> None:
    """Save a row-normalised confusion matrix as PNG.

    Rows = true class, columns = predicted class (plus an "unknown" column).
    Each row is divided by its number of images, so the diagonal shows the
    recall per class and off-diagonal cells show which classes are confused.
    """
    columns = list(class_ids) + [UNKNOWN]
    counts = confusion_matrix(y_true, y_pred, labels=columns)
    # The last row would be "true class = unknown", which never occurs.
    counts = counts[:-1]
    rates = counts / counts.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    image = ax.imshow(rates, cmap=cmap, vmin=0, vmax=1)

    # Write the percentage into every non-empty cell; empty cells stay blank
    # so the few real confusions stand out. Rare confusions are shown as
    # "<1" instead of a rounded "0", which would look like no error at all.
    # White text on dark cells.
    for row in range(rates.shape[0]):
        for col in range(rates.shape[1]):
            if counts[row, col] > 0:
                value = rates[row, col]
                label = "<1" if value < 0.005 else f"{value * 100:.0f}"
                ax.text(col, row, label, ha="center", va="center", fontsize=8,
                        color="white" if value > 0.5 else TEXT)

    names = [class_label(c) for c in columns]
    ax.set_xticks(range(len(columns)), names, rotation=45, ha="right",
                  fontsize=8)
    ax.set_yticks(range(len(class_ids)), names[:-1], fontsize=8)
    ax.set_xlabel("Predicted class", color=TEXT_MUTED)
    ax.set_ylabel("True class", color=TEXT_MUTED)
    ax.set_title(f"{title} – confusion matrix (% of true class)",
                 color=TEXT, loc="left")
    _style_axes(ax)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=TEXT_MUTED, length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_f1_comparison(metrics_a: dict, metrics_b: dict, names, path) -> None:
    """Save a horizontal grouped bar chart of per-class F1 for two methods."""
    class_ids = list(metrics_a["per_class"])
    positions = np.arange(len(class_ids))
    bar_height = 0.38

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for offset, metrics, name, color in zip((-1, 1), (metrics_a, metrics_b),
                                            names, SERIES_COLORS):
        values = [metrics["per_class"][c]["f1"] for c in class_ids]
        # The white edge leaves a thin gap between the two bars of a class.
        ax.barh(positions + offset * bar_height / 2, values, height=bar_height,
                color=color, edgecolor=SURFACE, linewidth=1.5, label=name)

    ax.set_yticks(positions, [CLASS_NAMES[c] for c in class_ids], fontsize=9)
    # Best class at the top reads more naturally than ascending ids.
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score on test set", color=TEXT_MUTED)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Per-class F1: baseline vs. main method", color=TEXT,
                 loc="left", pad=28)
    _style_axes(ax)
    # Legend in one row between title and plot, so it never covers bars.
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=2, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_examples(images, y_true, y_pred, confidences, title: str, path,
                  n_images: int = 12) -> None:
    """Save a grid of example images with true and predicted class."""
    n_images = min(n_images, len(images))
    n_cols = 6
    n_rows = max(1, int(np.ceil(n_images / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.1, n_rows * 3.0),
                             facecolor=SURFACE, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, true, pred, conf in zip(axes.flat, images[:n_images],
                                           y_true, y_pred, confidences):
        # OpenCV stores BGR, matplotlib expects RGB.
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(f"true: {class_label(true)}\npred: {class_label(pred)}"
                     f"\nconf: {conf:.2f}", fontsize=7, color=TEXT)
    fig.suptitle(title, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
