"""
pipeline.py - the ONE place where all image-processing stages are chained.

Training, evaluation scripts and the GUI all call analyze_image(), so they can never
disagree about how an image is processed.

    load -> preprocess -> leaf mask -> lesion mask -> clean-up -> edges + contours
         -> severity -> (features, used by the classifier)
"""
from dataclasses import dataclass
from typing import List

import numpy as np

from src import config, features, refine, segmentation, severity
from src.preprocess import load_image, preprocess


@dataclass
class Analysis:
    """Everything produced for one image. Saved/shown stage by stage."""
    original: np.ndarray
    denoised: np.ndarray
    enhanced: np.ndarray
    leaf_mask: np.ndarray
    raw_lesion_mask: np.ndarray      # straight from the segmentation method
    lesion_mask: np.ndarray          # after morphology + small-blob removal
    edges: np.ndarray                # Canny edge map (extra view + one feature)
    lesions: List[refine.Lesion]     # one record per lesion blob
    severity_percent: float
    severity_level: str
    lesion_method: str


def analyze_array(bgr_image, lesion_method=config.DEFAULT_LESION_METHOD):
    """Run the full pipeline on a loaded BGR image. May raise NoLeafFoundError."""
    prepared = preprocess(bgr_image)
    leaf_mask = segmentation.extract_leaf_mask(prepared.denoised)

    segment_lesions = segmentation.LESION_METHODS[lesion_method]
    raw_lesion_mask = segment_lesions(prepared.denoised, leaf_mask)
    lesion_mask = refine.clean_lesion_mask(raw_lesion_mask)

    edges = refine.compute_edge_map(prepared.enhanced, leaf_mask)
    lesions = refine.find_lesions(lesion_mask)

    percent = severity.severity_percent(leaf_mask, lesion_mask)
    return Analysis(
        original=prepared.original, denoised=prepared.denoised, enhanced=prepared.enhanced,
        leaf_mask=leaf_mask, raw_lesion_mask=raw_lesion_mask, lesion_mask=lesion_mask,
        edges=edges, lesions=lesions, severity_percent=percent,
        severity_level=severity.severity_level(percent), lesion_method=lesion_method)


def analyze_file(path, lesion_method=config.DEFAULT_LESION_METHOD):
    """Load an image file (may raise InvalidImageError) and analyse it."""
    return analyze_array(load_image(path), lesion_method)


def feature_dict(analysis):
    """The ~90 classifier features of an Analysis, as {name: value}."""
    return features.extract_features(
        denoised_bgr=analysis.denoised, texture_bgr=analysis.enhanced,
        leaf_mask=analysis.leaf_mask, lesion_mask=analysis.lesion_mask,
        lesions=analysis.lesions, edges=analysis.edges)
