"""
errors.py - the two ways an input image can be unusable.

The GUI catches these and shows a friendly message instead of crashing.
"""


class InvalidImageError(Exception):
    """The file is missing, is not an image, or is too small to analyse."""


class NoLeafFoundError(Exception):
    """The image loaded fine but we could not find a leaf-sized object in it."""
