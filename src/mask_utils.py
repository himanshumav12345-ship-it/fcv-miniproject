"""
mask_utils.py - small helpers for binary masks (uint8 images that are 0 or 255).

A "mask" here is a picture the same size as the leaf photo where 255 means
"this pixel belongs to the thing we are looking for" and 0 means "it does not".
All the morphology (opening / closing / hole filling) lives here so every other
module can reuse the same, well-tested building blocks.
"""
import cv2
import numpy as np


def ellipse_kernel(size):
    """A round structuring element. Round kernels treat all directions equally,
    so they do not make blobs look square."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def open_mask(mask, kernel_size):
    """Opening = erode then dilate. Removes specks smaller than the kernel
    while leaving large shapes almost unchanged."""
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse_kernel(kernel_size))


def close_mask(mask, kernel_size):
    """Closing = dilate then erode. Joins fragments and fills gaps smaller than
    the kernel."""
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse_kernel(kernel_size))


def keep_largest_component(mask):
    """Keep only the biggest connected blob (the leaf), fill its holes, and
    drop everything else. Returns an all-zero mask if there is no blob."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(mask)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        # thickness=FILLED paints the inside of the outline, which also fills holes
        cv2.drawContours(result, [biggest], -1, 255, thickness=cv2.FILLED)
    return result


def remove_small_blobs(mask, min_area):
    """Delete connected blobs smaller than min_area pixels."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = np.zeros_like(mask)
    for label in range(1, count):          # label 0 is the background
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            result[labels == label] = 255
    return result


def area_in_pixels(mask):
    """Number of mask pixels that are on."""
    return int(np.count_nonzero(mask))
