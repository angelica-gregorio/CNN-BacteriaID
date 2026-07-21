"""Morphological operations for cleaning up binary segmentation masks."""

import cv2
import numpy as np

_SHAPES = {
    "ellipse": cv2.MORPH_ELLIPSE,
    "rect": cv2.MORPH_RECT,
    "cross": cv2.MORPH_CROSS,
}


def get_kernel(shape: str = "ellipse", ksize: tuple[int, int] = (5, 5)) -> np.ndarray:
    if shape not in _SHAPES:
        raise ValueError(f"shape must be one of {list(_SHAPES)}, got {shape!r}")
    return cv2.getStructuringElement(_SHAPES[shape], ksize)


def close_mask(mask: np.ndarray, kernel: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Morphological closing: fills small internal holes and bridges gaps into solid blobs."""
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def open_mask(mask: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Morphological opening: strips small noise specks and stray pixels."""
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)


def clean_mask(
    mask: np.ndarray,
    kernel_size: tuple[int, int] = (5, 5),
    close_iterations: int = 2,
    open_iterations: int = 1,
) -> np.ndarray:
    """Close then open a binary mask.

    Close first to consolidate bacteria blobs into solid regions, then open to
    remove noise -- reversing the order would let opening strip thin
    bacterial shapes before they're consolidated.
    """
    kernel = get_kernel("ellipse", kernel_size)
    closed = close_mask(mask, kernel, iterations=close_iterations)
    return open_mask(closed, kernel, iterations=open_iterations)
