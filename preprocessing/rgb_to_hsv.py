"""RGB -> HSV conversion utilities.

For Gram-stained microscopy images, the crystal-violet/safranin stain makes
bacteria noticeably more saturated in color than the background field, so the
saturation channel tends to separate stain from background better than plain
grayscale luminance. Hue and value are exposed too since some species/stain
intensities respond better to one channel over another.
"""

import cv2
import numpy as np

_CHANNEL_INDEX = {"hue": 0, "saturation": 1, "value": 2}


def bgr_to_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image (as loaded by cv2.imread) to HSV color space."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)


def split_hsv_channels(hsv_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an HSV image into its (hue, saturation, value) channels."""
    h, s, v = cv2.split(hsv_image)
    return h, s, v


def get_segmentation_channel(hsv_image: np.ndarray, channel: str = "saturation") -> np.ndarray:
    """Return a single 2D channel from an HSV image for use as thresholding input.

    Args:
        hsv_image: HSV image as produced by bgr_to_hsv.
        channel: one of "hue", "saturation", "value".
    """
    if channel not in _CHANNEL_INDEX:
        raise ValueError(f"channel must be one of {list(_CHANNEL_INDEX)}, got {channel!r}")
    return hsv_image[:, :, _CHANNEL_INDEX[channel]]
