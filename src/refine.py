"""
refine.py - clean the raw lesion mask, draw an edge map, and measure each lesion.

Input : the raw lesion mask (from segmentation.py) and the images.
Output: a cleaned lesion mask, a Canny edge map, and one `Lesion` record per blob.

Important honesty note about Canny: it finds EDGES (places where brightness changes
sharply). Here it is used as an extra view for the GUI and as one numeric feature
(edge density). It does NOT edit or correct the lesion mask.
"""
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from src import config, mask_utils


# -----------------------------------------------------------------------------
# Morphological clean-up
# -----------------------------------------------------------------------------
def clean_lesion_mask(raw_mask):
    """Opening -> closing -> drop tiny blobs.

    Opening removes speckles, closing glues together a lesion that thresholding
    cut into nearby fragments, and the area filter deletes whatever noise is left.
    """
    mask = mask_utils.open_mask(raw_mask, config.LESION_OPEN_KERNEL)
    mask = mask_utils.close_mask(mask, config.LESION_CLOSE_KERNEL)
    return mask_utils.remove_small_blobs(mask, config.MIN_LESION_AREA_PIXELS)


# -----------------------------------------------------------------------------
# Canny edge map
# -----------------------------------------------------------------------------
def compute_edge_map(enhanced_bgr, leaf_mask):
    """Canny edges of the leaf's grey image, restricted to the leaf core.

    Canny needs a low and a high threshold. Picking fixed numbers fails on dark or
    bright photos, so both are derived from the image's median brightness
    ("auto-Canny"): low = (1 - sigma) * median, high = (1 + sigma) * median.
    """
    grey = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    core = mask_utils.ellipse_kernel(config.LEAF_RIM_ERODE_KERNEL)
    core_mask = cv2.erode(leaf_mask, core)        # skip the leaf outline itself
    inside = grey[core_mask > 0]
    median = float(np.median(inside)) if inside.size else 128.0
    low = int(max(0, (1.0 - config.CANNY_SIGMA) * median))
    high = int(min(255, (1.0 + config.CANNY_SIGMA) * median))
    edges = cv2.Canny(grey, low, high)
    edges[core_mask == 0] = 0
    return edges


# -----------------------------------------------------------------------------
# Contour analysis
# -----------------------------------------------------------------------------
@dataclass
class Lesion:
    """Measurements of one lesion blob (all areas are in pixels)."""
    contour: np.ndarray
    area: float
    perimeter: float
    circularity: float      # round vs elongated indicator, see note in describe_contour
    solidity: float         # area / convex-hull area; low = ragged or branching
    aspect_ratio: float     # long side / short side of the bounding box


def describe_contour(contour):
    """Turn one OpenCV contour into a Lesion with its shape measurements."""
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    # circularity = 4*pi*A / P^2 ; exactly 1 for a perfect continuous circle. On a pixel grid
    # OpenCV traces the outline through boundary pixel centres, which under-counts the area and
    # over-counts the staircase perimeter, so a perfect digital circle scores only ~0.67 at
    # radius 3, ~0.83 at radius 10 and ~0.89 at radius 60. Treat circularity as a RELATIVE
    # round-vs-elongated measure (a thin streak scores far lower), not as an absolute 0-1 scale.
    circularity = 4.0 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = area / hull_area if hull_area > 0 else 0.0
    _, _, width, height = cv2.boundingRect(contour)
    aspect_ratio = max(width, height) / max(1, min(width, height))
    return Lesion(contour, area, perimeter, min(circularity, 1.0), solidity, aspect_ratio)


def find_lesions(lesion_mask) -> List[Lesion]:
    """Outline every lesion blob (outer contours only) and measure it."""
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [describe_contour(c) for c in contours]


# -----------------------------------------------------------------------------
# Drawing for the GUI and the saved stage images
# -----------------------------------------------------------------------------
def draw_lesion_overlay(original_bgr, leaf_mask, lesions):
    """Original image + leaf outline (green) + lesion outlines (red) + boxes (yellow)."""
    overlay = original_bgr.copy()
    leaf_contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, leaf_contours, -1, (0, 200, 0), 1)
    for lesion in lesions:
        cv2.drawContours(overlay, [lesion.contour], -1, (0, 0, 255), 1)
        x, y, w, h = cv2.boundingRect(lesion.contour)
        cv2.rectangle(overlay, (x - 1, y - 1), (x + w + 1, y + h + 1), (0, 220, 255), 1)
    return overlay
