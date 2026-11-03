"""
Unit tests: small, fast checks that each building block does what its docstring says.

Run:  python -m unittest discover -s tests -v        (standard library only, nothing to install)
Each test builds a tiny synthetic image where the right answer is known in advance.
"""
import random
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src import config, dataset, mask_utils, pipeline, refine, severity
from src.errors import InvalidImageError, NoLeafFoundError
from src.preprocess import limit_size, load_image
from src.segmentation import leaf_core_mask


def synthetic_leaf(lesion=True, size=256):
    """Grey background, a green ellipse (the leaf) and optionally a brown disc (a lesion)."""
    image = np.full((size, size, 3), 140, np.uint8)
    cv2.ellipse(image, (size // 2, size // 2), (90, 60), 30, 0, 360, (40, 150, 60), -1)
    if lesion:
        cv2.circle(image, (size // 2, size // 2), 18, (30, 70, 120), -1)
    return image


class MaskTests(unittest.TestCase):
    def test_remove_small_blobs_drops_only_small_ones(self):
        mask = np.zeros((50, 50), np.uint8)
        mask[5:7, 5:7] = 255          # 4 pixels
        mask[20:40, 20:40] = 255      # 400 pixels
        cleaned = mask_utils.remove_small_blobs(mask, min_area=10)
        self.assertEqual(mask_utils.area_in_pixels(cleaned), 400)

    def test_keep_largest_component_fills_holes(self):
        mask = np.zeros((60, 60), np.uint8)
        mask[10:50, 10:50] = 255
        mask[25:30, 25:30] = 0        # a hole
        result = mask_utils.keep_largest_component(mask)
        self.assertEqual(mask_utils.area_in_pixels(result), 40 * 40)


class ContourTests(unittest.TestCase):
    def test_circle_is_round_but_pixelation_keeps_circularity_below_one(self):
        # On a pixel grid the contour runs through boundary pixel centres, so the area is
        # slightly under-counted and the staircase perimeter over-counted. A perfect circle
        # of radius 30 therefore measures about 0.88, not 1.0 (see the docstring in refine.py).
        mask = np.zeros((100, 100), np.uint8)
        cv2.circle(mask, (50, 50), 30, 255, -1)
        lesion = refine.find_lesions(mask)[0]
        self.assertTrue(0.8 < lesion.circularity < 1.0)
        self.assertAlmostEqual(lesion.solidity, 1.0, delta=0.05)   # same staircase effect: ~0.98

    def test_long_thin_rectangle_is_not_round(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[45:55, 5:95] = 255
        lesion = refine.find_lesions(mask)[0]
        self.assertLess(lesion.circularity, 0.4)
        self.assertGreater(lesion.aspect_ratio, 5)


class SeverityTests(unittest.TestCase):
    def test_percent_is_lesion_over_leaf_inside_the_same_region(self):
        leaf = np.zeros((200, 200), np.uint8)
        cv2.circle(leaf, (100, 100), 80, 255, -1)
        lesion = leaf_core_mask(leaf)
        lesion[:, 100:] = 0                       # lesion = left half of the core
        self.assertAlmostEqual(severity.severity_percent(leaf, lesion), 50.0, delta=2.0)

    def test_levels_follow_the_configured_limits(self):
        self.assertEqual(severity.severity_level(0.0), "Trace / healthy")
        self.assertEqual(severity.severity_level(10.0), "Mild")
        self.assertEqual(severity.severity_level(20.0), "Moderate")
        self.assertEqual(severity.severity_level(80.0), "Severe")


class DetectionTests(unittest.TestCase):
    def test_leaf_mask_finds_the_green_ellipse(self):
        analysis = pipeline.analyze_array(synthetic_leaf(lesion=False))
        expected = np.zeros((256, 256), np.uint8)
        cv2.ellipse(expected, (128, 128), (90, 60), 30, 0, 360, 255, -1)
        overlap = np.count_nonzero((analysis.leaf_mask > 0) & (expected > 0))
        union = np.count_nonzero((analysis.leaf_mask > 0) | (expected > 0))
        self.assertGreater(overlap / union, 0.9)

    def test_brown_disc_is_a_lesion_and_clean_leaf_is_not(self):
        with_lesion = pipeline.analyze_array(synthetic_leaf(lesion=True))
        clean = pipeline.analyze_array(synthetic_leaf(lesion=False))
        self.assertGreater(with_lesion.severity_percent, 5)
        self.assertLess(clean.severity_percent, 1)
        self.assertGreaterEqual(len(with_lesion.lesions), 1)


class InvalidInputTests(unittest.TestCase):
    def test_missing_and_text_files_raise_invalid_image(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(InvalidImageError):
                load_image(Path(folder) / "missing.jpg")
            fake = Path(folder) / "fake.jpg"
            fake.write_text("not an image")
            with self.assertRaises(InvalidImageError):
                load_image(fake)

    def test_tiny_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tiny.png"
            cv2.imwrite(str(path), np.zeros((20, 20, 3), np.uint8))
            with self.assertRaises(InvalidImageError):
                load_image(path)

    def test_blank_noise_and_blue_object_report_no_leaf(self):
        rng = np.random.default_rng(0)
        blank = np.full((256, 256, 3), 128, np.uint8)
        noise = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        blue = np.full((256, 256, 3), 140, np.uint8)
        cv2.rectangle(blue, (60, 60), (200, 200), (220, 60, 30), -1)
        for name, image in (("blank", blank), ("noise", noise), ("blue", blue)):
            with self.subTest(image=name):
                with self.assertRaises(NoLeafFoundError):
                    pipeline.analyze_array(image)

    def test_large_images_are_shrunk_to_the_maximum_side(self):
        big = np.zeros((1000, 600, 3), np.uint8)
        self.assertEqual(max(limit_size(big).shape[:2]), config.MAX_IMAGE_SIDE)


class DatasetSplitTests(unittest.TestCase):
    def test_neighbouring_image_numbers_share_a_group(self):
        a = dataset.make_group_id("Early_blight", "uuid1___RS_Erly.B 7721.JPG")
        b = dataset.make_group_id("Early_blight", "uuid2___RS_Erly.B 7722.JPG")
        c = dataset.make_group_id("Early_blight", "uuid3___RS_Erly.B 9000.JPG")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_groups_are_assigned_to_all_three_splits_with_sensible_shares(self):
        sizes = {f"g{i}": 3 for i in range(100)}
        assignment = dataset.assign_groups_to_splits(sizes, random.Random(0))
        self.assertEqual(set(assignment.values()), {"train", "val", "test"})
        test_share = sum(sizes[g] for g, s in assignment.items() if s == "test") / sum(sizes.values())
        self.assertTrue(0.12 < test_share < 0.20)

    def test_saved_split_file_has_no_group_in_two_splits(self):
        import pandas as pd
        table = pd.read_csv(config.SPLITS_CSV)
        self.assertTrue((table.groupby("group")["split"].nunique() == 1).all())


if __name__ == "__main__":
    unittest.main()
