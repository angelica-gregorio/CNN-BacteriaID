"""Shared constants for the BacteriaID-CNN pipeline.

Only values genuinely shared across preprocessing, training, and evaluation
modules live here. Single-algorithm hyperparameters (Canny sigma, adaptive
threshold block size, morphology kernel size, training hyperparameters) stay
local to the module/CLI that owns them.
"""

from pathlib import Path

IMAGE_SIZE = (128, 128)
RANDOM_SEED = 42
MIN_CONTOUR_AREA = 500  # px^2; shared by segmentation region filtering and RIA metric

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"
SEGMENTED_DIR = DATASET_DIR / "segmented"
AUGMENTED_DIR = DATASET_DIR / "augmented"
RESULTS_DIR = PROJECT_ROOT / "results"

VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
