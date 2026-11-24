"""Central configuration of the classification part of the project.

All scripts import their settings from here, so an experiment (other classes,
other image size, other seed) is changed in exactly one place.
"""

from pathlib import Path

# Project root = parent of the ``src`` folder; all other paths are built from
# it so the scripts work no matter from which directory they are started.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "GTSRB" / "Final_Training" / "Images"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Subset of GTSRB classes used for training and evaluation. They were chosen
# to cover all sign shapes and colours (round red/blue, triangles, octagon,
# diamond) and to include visually similar pairs (speed limits 30/50/70),
# which makes the comparison of the two classifiers meaningful.
# Set to ``None`` to use all 43 classes.
CLASS_IDS = [1, 2, 4, 9, 11, 12, 13, 14, 17, 25, 35, 38]

# Every sign is cropped to its ROI and resized to IMAGE_SIZE x IMAGE_SIZE px.
# 48 px keeps enough detail for the digits on speed-limit signs while keeping
# feature vectors small.
IMAGE_SIZE = 48

# Fraction of the data reserved for the final test set (never used for
# training or hyper-parameter tuning) and the seed that makes the split and
# all other random steps reproducible.
TEST_FRACTION = 0.2
RANDOM_SEED = 42

# Number of folds for the cross-validation used in hyper-parameter search.
# 3 folds keep training time short while still averaging over several splits.
CV_FOLDS = 3

# Rare classes (see src/augmentation.py). A class needs at least CV_FOLDS
# independent groups (tracks / source images); otherwise some CV training
# fold contains no example of it, the SVM never learns it there and the
# probability calibration fails. Such classes are filled up with augmented
# copies to MIN_SAMPLES_PER_CLASS samples, so every one of the 3 validation
# folds still gets 4 examples of the class and every training fold 8.
MIN_GROUPS_PER_CLASS = CV_FOLDS
MIN_SAMPLES_PER_CLASS = 4 * CV_FOLDS

# Below these confidence values a prediction is reported as "unknown".
# This is needed in the full pipeline, where the detector can also deliver
# regions that are not traffic signs at all.
# * Template matching: the confidence is a correlation coefficient (-1..1);
#   values under 0.3 are usually considered only a weak correlation.
# * Feature classifiers: the confidence is a class probability (0..1); 0.5
#   means "the winning class is more likely than all others together".
TM_UNKNOWN_THRESHOLD = 0.3
ML_UNKNOWN_THRESHOLD = 0.5
