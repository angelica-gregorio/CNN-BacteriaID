"""Canny edge detection with automatic (median-based) threshold selection.

Manually tuning Canny thresholds per image is impractical across 33
heterogeneous species folders with varying stain intensity, so thresholds are
derived from each image's own median intensity (Adrian Rosebrock's
"auto Canny" method).
"""

import cv2
import numpy as np


def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny edge detection with thresholds derived from the image median."""
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(gray, lower, upper)


def edges_to_mask(edges: np.ndarray, dilate_ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """Dilate thin, unclosed Canny edge lines into small closed regions.

    Raw Canny output is single-pixel-wide contour lines, not filled blobs;
    dilating slightly closes small gaps so downstream connected-component
    analysis sees continuous boundaries rather than broken segments.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
    return cv2.dilate(edges, kernel, iterations=iterations)
