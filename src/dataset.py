"""Loading and preprocessing of the GTSRB dataset.

Layout of the GTSRB training archive (one folder per class)::

    Images/00014/GT-00014.csv      annotation file of class 14
    Images/00014/00003_00029.ppm   image 29 of track 3 of class 14

Each CSV row describes one image: ``Filename;Width;Height;Roi.X1;Roi.Y1;
Roi.X2;Roi.Y2;ClassId``. The ROI is the bounding box of the sign inside the
image (the images contain a margin of background around the sign).

A *track* is a series of ~30 frames of the same physical sign, recorded while
the car approached it. Frames of one track are nearly identical, which matters
for the train/test split (see :func:`split_by_track`).
"""

import csv
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src import config

# Short English names of all 43 GTSRB classes (index = class id), used for
# plot labels and reports.
CLASS_NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End of limit 80", "Speed limit 100",
    "Speed limit 120", "No passing", "No passing >3.5t",
    "Right-of-way next junction", "Priority road", "Yield", "Stop",
    "No vehicles", "No vehicles >3.5t", "No entry", "General caution",
    "Curve left", "Curve right", "Double curve", "Bumpy road",
    "Slippery road", "Road narrows right", "Road work", "Traffic signals",
    "Pedestrians", "Children crossing", "Bicycles crossing", "Ice/snow",
    "Wild animals", "End of all limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Straight or right", "Straight or left",
    "Keep right", "Keep left", "Roundabout", "End of no passing",
    "End of no passing >3.5t",
]


def read_image(path: Path) -> np.ndarray:
    """Read an image file as a BGR ``uint8`` array.

    ``cv2.imread`` cannot open paths with non-ASCII characters on Windows,
    so the file is read as raw bytes and decoded by OpenCV instead.
    """
    raw_bytes = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Could not read image {path}")
    return image


def resize_roi(roi_bgr: np.ndarray, size: int = config.IMAGE_SIZE) -> np.ndarray:
    """Scale a sign crop of any size to ``size`` x ``size`` pixels.

    ``INTER_AREA`` averages pixels when shrinking, which avoids aliasing;
    for the (rarer) case of enlarging small crops it behaves like bilinear
    interpolation. The aspect ratio is not kept on purpose: GTSRB ROIs are
    almost square, and a fixed size is required for templates and HOG.
    """
    return cv2.resize(roi_bgr, (size, size), interpolation=cv2.INTER_AREA)


def load_class(class_id: int, root: Path = config.DATA_DIR,
               size: int = config.IMAGE_SIZE):
    """Load all images of one class, cropped to their ROI and resized.

    Returns:
        images: list of ``size`` x ``size`` x 3 BGR arrays.
        tracks: list of track ids (str), one per image, e.g. ``"14_00003"``.
    """
    class_dir = root / f"{class_id:05d}"
    csv_path = class_dir / f"GT-{class_id:05d}.csv"

    images, tracks = [], []
    with open(csv_path, newline="") as csv_file:
        for row in csv.DictReader(csv_file, delimiter=";"):
            image = read_image(class_dir / row["Filename"])

            # The ROI corners are inclusive pixel coordinates, hence the +1
            # in the slice end. Cropping removes the background margin so the
            # classifier sees the same kind of input the detector delivers.
            x1, y1 = int(row["Roi.X1"]), int(row["Roi.Y1"])
            x2, y2 = int(row["Roi.X2"]), int(row["Roi.Y2"])
            images.append(resize_roi(image[y1:y2 + 1, x1:x2 + 1], size))

            # The file name prefix ("00003" in "00003_00029.ppm") is the track
            # number. It is only unique within a class, so the class id is
            # added to get a dataset-wide unique group id.
            tracks.append(f"{class_id}_{row['Filename'].split('_')[0]}")
    return images, tracks


def load_gtsrb(class_ids=config.CLASS_IDS, root: Path = config.DATA_DIR,
               size: int = config.IMAGE_SIZE):
    """Load the chosen GTSRB classes.

    Args:
        class_ids: list of class ids to load, or ``None`` for all 43 classes.
        root: folder that contains one sub-folder per class.
        size: side length in pixels of the resized images.

    Returns:
        images: ``(N, size, size, 3)`` uint8 array of BGR images.
        labels: ``(N,)`` int array of class ids.
        tracks: ``(N,)`` str array of track ids (used as groups for splitting).
    """
    if class_ids is None:
        class_ids = range(len(CLASS_NAMES))

    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset not found in {root}. Run 'python download_data.py' first.")

    all_images, all_labels, all_tracks = [], [], []
    for class_id in class_ids:
        images, tracks = load_class(class_id, root, size)
        all_images.extend(images)
        all_tracks.extend(tracks)
        all_labels.extend([class_id] * len(images))

    return np.array(all_images), np.array(all_labels), np.array(all_tracks)


def split_by_track(labels: np.ndarray, tracks: np.ndarray,
                   test_fraction: float = config.TEST_FRACTION,
                   seed: int = config.RANDOM_SEED):
    """Create a reproducible train/test split in which no track is shared.

    Why not a plain random split: the ~30 frames of one track are almost
    identical. If some went to training and others to testing, the test set
    would contain near-copies of training images and the measured accuracy
    would be too optimistic. Therefore whole tracks are assigned to one side.

    ``StratifiedGroupKFold`` does exactly this (groups = tracks) while keeping
    the class proportions similar in both parts (stratified by label). With
    ``n_splits = 1 / test_fraction`` folds, one fold is used as the test set.

    Returns:
        train_idx, test_idx: index arrays into the loaded dataset.
    """
    n_splits = round(1 / test_fraction)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed)
    # Only the first fold is needed; the image data itself is irrelevant for
    # the split, so a dummy array of the right length is passed as X.
    train_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels,
                                              groups=tracks))
    return train_idx, test_idx
