"""
features.py - turn one analysed leaf into a vector of numbers for the classifier.

Three families of features (names start with colour_ / texture_ / shape_, which is
how the ablation study later switches whole families on and off):

  colour  - what colours the leaf and the lesions have       (55 numbers)
  texture - how rough or smooth the surface pattern is        (20 numbers)
  shape   - how many lesions, how big, how round, how ragged  (15 numbers)

Everything is a RATIO or an average, never a raw pixel count, so a bigger photo of
the same leaf gives the same numbers.
"""
import cv2
import numpy as np
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from src import config, mask_utils
from src.segmentation import leaf_core_mask

GLCM_LEVELS = 32                       # gray levels used for texture (31 real + 1 "outside")
GLCM_DISTANCES = [1, 3]                # compare pixels 1 and 3 steps apart
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]   # horizontal, diagonal, vertical, diagonal
GLCM_PROPERTIES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
LBP_POINTS, LBP_RADIUS = 8, 1          # 8 neighbours on a circle of radius 1
LBP_BINS = LBP_POINTS + 2              # 'uniform' LBP produces values 0..9


# =============================================================================
# COLOUR
# =============================================================================
def channel_statistics(channel_values, prefix):
    """mean, standard deviation and skewness of a 1-D array of pixel values.

    Skewness tells whether the distribution has a long tail (for example a few very
    dark lesion pixels on an otherwise bright leaf). Empty regions give zeros.
    """
    if channel_values.size == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0, f"{prefix}_skew": 0.0}
    std = float(np.std(channel_values))
    skewness = float(skew(channel_values)) if std > 1e-6 else 0.0
    return {f"{prefix}_mean": float(np.mean(channel_values)),
            f"{prefix}_std": std,
            f"{prefix}_skew": skewness}


def colour_statistics(denoised_bgr, region_mask, region_name):
    """Stats of H, S, V (HSV) and L, a, b (Lab) over the pixels of `region_mask`.

    HSV describes colour the way people do (which colour, how vivid, how bright);
    Lab separates brightness (L) from colour (a: green-red, b: blue-yellow).
    6 channels x 3 statistics = 18 numbers per region.
    """
    hsv = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2LAB)
    inside = region_mask > 0
    features = {}
    for image, names in ((hsv, "HSV"), (lab, "Lab")):
        for index, channel in enumerate(names):
            prefix = f"colour_{region_name}_{channel}"
            features.update(channel_statistics(image[:, :, index][inside].astype(np.float64),
                                               prefix))
    return features


def hue_saturation_histograms(denoised_bgr, core_mask):
    """8-bin histograms of hue and saturation over the leaf (each sums to 1)."""
    hsv = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2HSV)
    inside = core_mask > 0
    features = {}
    for index, (name, upper) in enumerate((("hue", 180), ("sat", 256))):
        values = hsv[:, :, index][inside]
        counts, _ = np.histogram(values, bins=8, range=(0, upper))
        proportions = counts / max(values.size, 1)
        for bin_number, proportion in enumerate(proportions):
            features[f"colour_hist_{name}{bin_number}"] = float(proportion)
    return features


def colour_fractions(denoised_bgr, core_mask):
    """Share of leaf pixels that are green, yellow or brown/red.

    Only pixels with some colour (saturation >= 40) are counted, because the hue of
    a grey or black pixel is meaningless.
    """
    hsv = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2HSV)
    inside = core_mask > 0
    hue, saturation = hsv[:, :, 0], hsv[:, :, 1]
    coloured = inside & (saturation >= 40)
    total = max(np.count_nonzero(inside), 1)
    green = coloured & (hue >= config.GREEN_HUE_MIN) & (hue <= config.GREEN_HUE_MAX)
    yellow = coloured & (hue >= 20) & (hue < config.GREEN_HUE_MIN)
    brown = coloured & ((hue < 20) | (hue >= 170))
    return {"colour_fraction_green": np.count_nonzero(green) / total,
            "colour_fraction_yellow": np.count_nonzero(yellow) / total,
            "colour_fraction_brown": np.count_nonzero(brown) / total}


# =============================================================================
# TEXTURE
# =============================================================================
def quantise_leaf_gray(gray, core_mask):
    """Crop to the leaf and reduce brightness to levels 1..31; background becomes 0.

    GLCM counts pairs of neighbouring gray levels. Reserving level 0 for "not leaf"
    lets us delete every pair that involves background afterwards.
    """
    rows, columns = np.where(core_mask > 0)
    if rows.size == 0:
        return None
    top, bottom, left, right = rows.min(), rows.max() + 1, columns.min(), columns.max() + 1
    patch = gray[top:bottom, left:right].astype(np.int32)
    inside = core_mask[top:bottom, left:right] > 0
    quantised = 1 + (patch * (GLCM_LEVELS - 1)) // 256        # 1 .. 31
    quantised[~inside] = 0
    return quantised.astype(np.uint8)


def glcm_features(gray, core_mask):
    """Texture from the gray-level co-occurrence matrix (GLCM).

    A GLCM counts how often gray level i has a neighbour of level j at a given
    distance and direction. Smooth surfaces concentrate on the diagonal; rough or
    spotted surfaces spread out. We average over 4 directions so the result does not
    depend on how the leaf is rotated. 5 properties x 2 distances = 10 numbers.
    """
    zeros = {f"texture_glcm_{prop}_d{d}": 0.0 for d in GLCM_DISTANCES for prop in GLCM_PROPERTIES}
    quantised = quantise_leaf_gray(gray, core_mask)
    if quantised is None:
        return zeros

    matrix = graycomatrix(quantised, GLCM_DISTANCES, GLCM_ANGLES, levels=GLCM_LEVELS,
                          symmetric=True, normed=False).astype(np.float64)
    matrix[0, :, :, :] = 0            # discard pairs involving background
    matrix[:, 0, :, :] = 0
    totals = matrix.sum(axis=(0, 1), keepdims=True)
    if np.any(totals == 0):
        return zeros
    matrix /= totals                  # now every (distance, angle) slice sums to 1

    features = {}
    for prop in GLCM_PROPERTIES:
        values = graycoprops(matrix, prop)               # shape (n_distances, n_angles)
        for distance_index, distance in enumerate(GLCM_DISTANCES):
            features[f"texture_glcm_{prop}_d{distance}"] = float(
                np.nan_to_num(values[distance_index].mean()))
    return features


def lbp_features(gray, core_mask):
    """Local Binary Pattern histogram (10 numbers).

    For each pixel, LBP compares it with its 8 neighbours (brighter or darker?) and
    encodes the pattern as a number. 'Uniform' patterns are the few clean ones
    (flat area, edge, corner, spot); the histogram says how often each appears.
    """
    codes = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    values = codes[core_mask > 0]
    counts, _ = np.histogram(values, bins=LBP_BINS, range=(0, LBP_BINS))
    proportions = counts / max(values.size, 1)
    return {f"texture_lbp_{i}": float(p) for i, p in enumerate(proportions)}


# =============================================================================
# SHAPE (of the lesions)
# =============================================================================
def hu_moment_features(lesion_mask):
    """Seven Hu moments of the lesion mask, log-scaled (invariant to position,
    size and rotation of the lesion pattern). Zeros if there is no lesion."""
    names = [f"shape_hu{i}" for i in range(7)]
    if mask_utils.area_in_pixels(lesion_mask) == 0:
        return {name: 0.0 for name in names}
    hu = cv2.HuMoments(cv2.moments(lesion_mask, binaryImage=True)).flatten()
    # Hu moments span many orders of magnitude, so take -sign * log10(|value|)
    logged = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return {name: float(value) for name, value in zip(names, logged)}


def lesion_shape_features(lesions, lesion_mask, edges, core_mask):
    """Counts and shape statistics of the lesion blobs, plus Canny edge density."""
    core_area = max(mask_utils.area_in_pixels(core_mask), 1)
    features = {
        "shape_lesion_count": float(len(lesions)),
        "shape_lesion_area_ratio": mask_utils.area_in_pixels(lesion_mask) / core_area,
        "shape_edge_density": np.count_nonzero(edges) / core_area,
    }
    if lesions:
        areas = np.array([lesion.area for lesion in lesions])
        features["shape_mean_lesion_area"] = float(areas.mean() / core_area)
        features["shape_max_lesion_area"] = float(areas.max() / core_area)
        features["shape_mean_circularity"] = float(np.mean([l.circularity for l in lesions]))
        features["shape_mean_solidity"] = float(np.mean([l.solidity for l in lesions]))
        features["shape_mean_aspect_ratio"] = float(np.mean([l.aspect_ratio for l in lesions]))
    else:
        for name in ("mean_lesion_area", "max_lesion_area", "mean_circularity",
                     "mean_solidity", "mean_aspect_ratio"):
            features[f"shape_{name}"] = 0.0
    features.update(hu_moment_features(lesion_mask))
    return features


# =============================================================================
# EVERYTHING TOGETHER
# =============================================================================
def extract_features(denoised_bgr, texture_bgr, leaf_mask, lesion_mask, lesions, edges):
    """Return {feature_name: value} for one leaf (about 90 numbers).

    denoised_bgr : used for colour (true colours)
    texture_bgr  : used for texture (we pass the contrast-enhanced image)
    """
    core = leaf_core_mask(leaf_mask)
    gray = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2GRAY)

    features = {}
    features.update(colour_statistics(denoised_bgr, core, "leaf"))
    features.update(colour_statistics(denoised_bgr, lesion_mask, "lesion"))
    features.update(hue_saturation_histograms(denoised_bgr, core))
    features.update(colour_fractions(denoised_bgr, core))
    features.update(glcm_features(gray, core))
    features.update(lbp_features(gray, core))
    features.update(lesion_shape_features(lesions, lesion_mask, edges, core))
    return features


def feature_group(feature_name):
    """'colour', 'texture' or 'shape' - the prefix of a feature name."""
    return feature_name.split("_", 1)[0]
