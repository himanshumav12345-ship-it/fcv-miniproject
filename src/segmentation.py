"""
segmentation.py - find WHERE things are in the image.

Two questions are answered here:
  1. Which pixels are leaf, and which are background?   -> extract_leaf_mask()
  2. Which leaf pixels look diseased?                    -> lesion functions below

A "mask" is a uint8 image with 255 where the thing is and 0 elsewhere.
"""
import cv2
import numpy as np
from skimage.filters import apply_hysteresis_threshold

from src import config
from src import mask_utils
from src.errors import NoLeafFoundError


# =============================================================================
# 1. LEAF MASK
# =============================================================================
# Idea: the background of a leaf photo is one roughly uniform colour (grey/purple
# paper or table). So we (a) measure that colour from the photo's border, and
# (b) label every pixel by how far its COLOUR is from it.
#
# Two choices here came from looking at real PlantVillage images:
#  * We compare only the colour channels a and b of Lab space and ignore lightness.
#    A cast shadow is darker than the background but has the same colour, so it
#    disappears; a plain saturation threshold kept the shadows as "leaf".
#  * Hysteresis thresholding (explained below) recovers pale or dead leaf tissue
#    whose colour is close to the background.

def estimate_background_chroma(lab_image):
    """Median (a, b) colour of the border pixels = the background colour.

    The median is used instead of the mean so a leaf tip touching the border
    does not drag the estimate towards leaf colours.
    """
    width = config.BACKGROUND_BORDER_PX
    border_pixels = np.concatenate([
        lab_image[:width].reshape(-1, 3),        # top strip
        lab_image[-width:].reshape(-1, 3),       # bottom strip
        lab_image[:, :width].reshape(-1, 3),     # left strip
        lab_image[:, -width:].reshape(-1, 3),    # right strip
    ])
    return np.median(border_pixels[:, 1:], axis=0)   # keep only a and b


def chroma_distance_map(lab_image, background_chroma):
    """Per pixel: Euclidean distance in the (a, b) plane to the background colour."""
    delta_a = lab_image[:, :, 1] - background_chroma[0]
    delta_b = lab_image[:, :, 2] - background_chroma[1]
    return np.sqrt(delta_a ** 2 + delta_b ** 2)


def otsu_threshold(float_map):
    """Otsu's threshold for a float image.

    Otsu picks the cut that best separates the histogram into two groups
    (minimises the variance inside each group). OpenCV's version needs an 8-bit
    image, so we rescale to 0..255, run Otsu, and convert the threshold back.
    """
    maximum = float(float_map.max())
    if maximum <= 1e-6:
        return 0.0                                  # perfectly flat image
    scaled = np.clip(float_map * (255.0 / maximum), 0, 255).astype(np.uint8)
    threshold_8bit, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold_8bit * maximum / 255.0


def extract_leaf_mask(denoised_bgr):
    """Return the leaf mask for a preprocessed image, or raise NoLeafFoundError.

    Steps: colour distance from background -> Otsu "sure leaf" -> hysteresis grows
    it into weaker neighbours -> opening removes specks -> keep the biggest blob
    and fill its holes.
    """
    lab_image = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    background = estimate_background_chroma(lab_image)
    distance = chroma_distance_map(lab_image, background)

    high = otsu_threshold(distance)
    low = config.HYSTERESIS_LOW_FACTOR * high
    # Hysteresis: keep pixels above `low` ONLY if they are connected to a pixel
    # above `high`. Weak pixels next to the leaf (dead tissue) are kept; weak
    # pixels floating in the background (noise) are not. Canny uses the same idea.
    keep = apply_hysteresis_threshold(distance, low, high)

    mask = keep.astype(np.uint8) * 255
    mask = mask_utils.open_mask(mask, config.LEAF_OPEN_KERNEL)
    mask = mask_utils.keep_largest_component(mask)

    fraction = mask_utils.area_in_pixels(mask) / mask.size
    if fraction < config.MIN_LEAF_FRACTION:
        raise NoLeafFoundError(
            f"The largest object covers only {fraction:.1%} of the image, "
            "too small to be a leaf.")
    if fraction > config.MAX_LEAF_FRACTION:
        raise NoLeafFoundError(
            f"The 'leaf' fills {fraction:.1%} of the image, so the background "
            "could not be told apart from the leaf.")
    check_looks_like_leaf(denoised_bgr, mask)
    return mask


def plant_colour_fraction(denoised_bgr, region_mask):
    """Share of the region's pixels that have a plant colour (brown/yellow/green, not grey)."""
    hsv = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation = hsv[:, :, 0], hsv[:, :, 1]
    plant = ((hue >= config.PLANT_HUE_MIN) & (hue <= config.PLANT_HUE_MAX)
             & (saturation >= config.PLANT_MIN_SATURATION))
    inside = region_mask > 0
    return np.count_nonzero(plant & inside) / max(np.count_nonzero(inside), 1)


def mask_solidity(mask):
    """Area / convex-hull area of the biggest blob: 1 = solid blob, small = ragged or scattered."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = max(contours, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(biggest))
    return cv2.contourArea(biggest) / hull_area if hull_area > 0 else 0.0


def check_looks_like_leaf(denoised_bgr, leaf_mask):
    """Reject objects that are clearly not a leaf, so the classifier is not asked to label them.

    The classifier only knows six leaf classes; given a random picture it would still
    answer confidently. These two cheap checks (colour and compactness) stop the obvious cases.
    """
    core = leaf_core_mask(leaf_mask)
    colour_share = plant_colour_fraction(denoised_bgr, core)
    if colour_share < config.MIN_PLANT_COLOUR_FRACTION:
        raise NoLeafFoundError(
            f"Only {colour_share:.0%} of the detected object has a leaf-like colour "
            "(green, yellow or brown), so it does not look like a leaf.")
    solidity = mask_solidity(leaf_mask)
    if solidity < config.MIN_LEAF_SOLIDITY:
        raise NoLeafFoundError(
            f"The detected object is too fragmented to be one leaf (solidity {solidity:.2f}).")


# =============================================================================
# 2. LESION MASK  (three methods, so they can be compared fairly)
# =============================================================================
# All three take the preprocessed BGR image and the leaf mask, and return a lesion
# mask that is a SUBSET of the leaf. The leaf mask is first shrunk by a few pixels
# (see leaf_core_mask) because rim pixels are leaf/background blends.

def leaf_core_mask(leaf_mask):
    """The leaf without its blurry rim (erosion shrinks the mask inwards)."""
    kernel = mask_utils.ellipse_kernel(config.LEAF_RIM_ERODE_KERNEL)
    return cv2.erode(leaf_mask, kernel)


def lesion_mask_hue_rule(denoised_bgr, leaf_mask):
    """Method A - the baseline: every leaf pixel whose hue is not green is a lesion.

    Simple and easy to explain, but it knows nothing about the leaf it is looking
    at: a naturally yellow-green leaf is "diseased" everywhere, and faint spots
    whose hue is still green are missed.
    """
    core = leaf_core_mask(leaf_mask)
    hue = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2HSV)[:, :, 0]
    is_green = (hue >= config.GREEN_HUE_MIN) & (hue <= config.GREEN_HUE_MAX)
    return ((~is_green) & (core > 0)).astype(np.uint8) * 255


def lesion_mask_kmeans(denoised_bgr, leaf_mask):
    """Method B - K-means: cluster the leaf's (a, b) colours into k groups and call
    the least-green group (highest a) the lesion.

    Weakness: K-means ALWAYS returns k groups, even for a healthy leaf, so it splits
    a perfectly healthy leaf into "darker green" and "lighter green" and calls one
    of them diseased. Nothing here checks that the groups are really different.
    """
    core = leaf_core_mask(leaf_mask)
    lab_image = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    leaf_colours = lab_image[core > 0][:, 1:]                 # (a, b) of leaf pixels
    result = np.zeros(core.shape, np.uint8)
    if len(leaf_colours) < config.KMEANS_CLUSTERS:
        return result

    cv2.setRNGSeed(config.RANDOM_SEED)                        # reproducible clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centres = cv2.kmeans(leaf_colours, config.KMEANS_CLUSTERS, None,
                                    criteria, 3, cv2.KMEANS_PP_CENTERS)
    least_green_cluster = int(np.argmax(centres[:, 0]))       # a grows from green to red
    result[core > 0] = (labels.ravel() == least_green_cluster).astype(np.uint8) * 255
    return result


def healthy_reference_chroma(lab_image, core_mask):
    """The colour THIS leaf has when it is healthy: median (a, b) of its greenest pixels.

    Greenest = lowest a. Using the leaf's own reference means a dark-green and a
    light-green leaf are each judged against their own normal colour.
    """
    inside = core_mask > 0
    a_channel = lab_image[:, :, 1]
    greenest = inside & (a_channel <= np.percentile(a_channel[inside],
                                                    config.HEALTHY_REFERENCE_PERCENTILE))
    return np.median(lab_image[greenest][:, 1:], axis=0)


def lesion_mask_own_green(denoised_bgr, leaf_mask):
    """Method C - our choice: lesion = colour clearly different from the leaf's own green.

    Only the colour channels (a, b) are compared, NOT lightness. Veins, folds and
    shading mostly change lightness; real lesions (brown, tan, yellow) change colour.
    The absolute floor MIN_LESION_CHROMA_DISTANCE keeps a healthy leaf at ~0 lesion.
    """
    core = leaf_core_mask(leaf_mask)
    if mask_utils.area_in_pixels(core) == 0:
        return np.zeros_like(core)
    lab_image = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    healthy = healthy_reference_chroma(lab_image, core)
    distance = np.sqrt((lab_image[:, :, 1] - healthy[0]) ** 2 +
                       (lab_image[:, :, 2] - healthy[1]) ** 2)
    return ((distance > config.MIN_LESION_CHROMA_DISTANCE) & (core > 0)).astype(np.uint8) * 255


LESION_METHODS = {
    "hue_rule": lesion_mask_hue_rule,
    "kmeans": lesion_mask_kmeans,
    "own_green": lesion_mask_own_green,
}
