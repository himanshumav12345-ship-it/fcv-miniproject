"""
severity.py - how much of the leaf is affected?

severity (%) = 100 * lesion pixels / leaf pixels

Both counts come from the SAME rim-trimmed leaf region (see segmentation.leaf_core_mask),
so the ratio compares like with like.

Limitations to state in the report (they are real):
  * It is an AREA in a flat 2-D photo. A curled, folded or partly hidden leaf, or a
    lesion on the underside, is not seen.
  * It is exactly as good as the two masks. If the lesion mask wrongly includes
    veins or shadows, severity is too high; if it misses faint spots, it is too low.
  * Pale yellowing may or may not be counted, depending on the lesion method.
  * The level cut-offs in config.SEVERITY_LEVELS are our own proposal, not an
    official plant-pathology scale.
"""
from src import config, mask_utils
from src.segmentation import leaf_core_mask


def severity_percent(leaf_mask, lesion_mask):
    """Percent of the (rim-trimmed) leaf area covered by lesions."""
    leaf_area = mask_utils.area_in_pixels(leaf_core_mask(leaf_mask))
    if leaf_area == 0:
        return 0.0
    lesion_area = mask_utils.area_in_pixels(lesion_mask)
    return 100.0 * lesion_area / leaf_area


def severity_level(percent):
    """Map a percentage to its named level using config.SEVERITY_LEVELS."""
    for upper_limit, name in config.SEVERITY_LEVELS:
        if percent <= upper_limit:
            return name
    return config.SEVERITY_LEVELS[-1][1]
