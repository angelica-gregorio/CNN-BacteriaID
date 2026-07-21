"""Image thresholding: Otsu global thresholding and adaptive thresholding.

Both functions produce an inverted binary mask (foreground = 255, background
= 0), since stained bacteria are darker/more saturated than the surrounding
field under bright-field microscopy -- inverting keeps the bacteria as the
"foreground" that cv2.connectedComponentsWithStats will later isolate.
"""

import cv2
import numpy as np


def otsu_threshold(gray: np.ndarray, blur_ksize: int = 5) -> tuple[float, np.ndarray]:
    """Global Otsu threshold after Gaussian blur (reduces noise-driven mis-thresholding).

    Returns:
        (otsu_threshold_value, binary_mask)
    """
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    thresh_value, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return thresh_value, mask


def adaptive_threshold(gray: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
    """Adaptive Gaussian thresholding -- handles uneven microscope illumination
    that a single global Otsu threshold would miss.

    block_size must be odd and >= 3; it is corrected automatically if not.
    """
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        C,
    )


def combine_thresholds(mask_a: np.ndarray, mask_b: np.ndarray, mode: str = "or") -> np.ndarray:
    """Combine two binary masks.

    mode="or" (default): union -- Otsu catches strong global contrast while
    adaptive catches local illumination unevenness; union recovers regions
    either method alone would miss (extra noise gets cleaned up by morphology).
    mode="and": intersection -- stricter, keeps only pixels both methods agree on.
    """
    if mode == "or":
        return cv2.bitwise_or(mask_a, mask_b)
    if mode == "and":
        return cv2.bitwise_and(mask_a, mask_b)
    raise ValueError(f"mode must be 'or' or 'and', got {mode!r}")
