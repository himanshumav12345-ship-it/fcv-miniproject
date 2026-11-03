"""
preprocess.py - load an image and clean it up before any measuring is done.

Input : a file path.
Output: three BGR images (original, denoised, contrast-enhanced).

Why preprocess at all? Raw photos contain sensor noise, which would create
thousands of tiny false edges and spots, and uneven lighting, which makes the same
leaf colour look different in different corners of the picture.
"""
from typing import NamedTuple

import cv2
import numpy as np

from src import config
from src.errors import InvalidImageError


class PreprocessedImage(NamedTuple):
    original: np.ndarray    # as loaded (after optional shrinking)
    denoised: np.ndarray    # Gaussian-blurred: used for colour measurements
    enhanced: np.ndarray    # denoised + CLAHE on lightness: used where contrast matters


def load_image(path):
    """Read an image from disk as a BGR array, or raise InvalidImageError.

    cv2.imread returns None instead of raising when a file is not an image,
    so we check for that explicitly.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError(f"Could not read '{path}' as an image.")
    return check_size(image)


def decode_image(raw_bytes):
    """Same as load_image, but for bytes already in memory (the GUI receives uploads this way)."""
    if not raw_bytes:
        raise InvalidImageError("The uploaded file is empty.")
    image = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("The uploaded file is not an image that can be read "
                                "(use a JPG or PNG photo).")
    return check_size(image)


def check_size(image):
    """Reject images too small to contain a measurable leaf; return the image unchanged otherwise."""
    height, width = image.shape[:2]
    if min(height, width) < config.MIN_IMAGE_SIDE:
        raise InvalidImageError(
            f"Image is only {width}x{height} pixels; need at least "
            f"{config.MIN_IMAGE_SIDE} on each side.")
    return image


def limit_size(image):
    """Shrink large photos so their longest side is MAX_IMAGE_SIDE.

    Thresholds such as 'minimum lesion area in pixels' only mean the same thing
    if every image has a similar size, so we normalise the size first.
    INTER_AREA is the right interpolation for shrinking (it averages pixels).
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= config.MAX_IMAGE_SIDE:
        return image
    scale = config.MAX_IMAGE_SIDE / longest
    new_size = (round(width * scale), round(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def denoise(image):
    """Gaussian blur: replaces each pixel by a weighted average of its neighbours."""
    size = (config.DENOISE_KERNEL, config.DENOISE_KERNEL)
    return cv2.GaussianBlur(image, size, 0)


def enhance_contrast(image):
    """CLAHE on the lightness channel only.

    We work in Lab colour space, where L is brightness and a/b are colour.
    Equalising only L boosts local contrast (dark lesions stand out more)
    WITHOUT changing the leaf's actual colours, which the classifier relies on.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=(config.CLAHE_TILE_GRID, config.CLAHE_TILE_GRID))
    lightness = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge([lightness, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def preprocess(image):
    """Run the whole preprocessing stage on an already-loaded BGR image."""
    original = limit_size(image)
    denoised = denoise(original)
    enhanced = enhance_contrast(denoised)
    return PreprocessedImage(original, denoised, enhanced)
