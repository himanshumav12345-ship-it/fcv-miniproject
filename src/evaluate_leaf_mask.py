"""
evaluate_leaf_mask.py - how good is our leaf mask?

PlantVillage ships a background-removed ("segmented") copy of every image. We use
it as an independent REFERENCE for the leaf mask: reference leaf = non-black pixels.
It is not perfect ground truth (it was also produced by an algorithm, and on some
blue-background Leaf_Mold images it keeps big chunks of background), so we report
median and share-above-threshold next to the mean, and save the worst cases to look at.

Run:
    python -m src.evaluate_leaf_mask --reference /path/to/PlantVillage-Dataset/raw/segmented
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import config, mask_utils
from src.errors import NoLeafFoundError
from src.preprocess import load_image, preprocess
from src.segmentation import extract_leaf_mask

REFERENCE_THRESHOLD = 8      # pixels darker than this in the segmented image = background


def load_reference_mask(reference_dir, image_path, label):
    """Leaf mask derived from PlantVillage's own segmented image."""
    folder_name = next(k for k, v in config.CLASS_FOLDERS.items() if v == label)
    segmented_path = reference_dir / folder_name / (Path(image_path).stem + "_final_masked.jpg")
    segmented = cv2.imread(str(segmented_path))
    mask = (segmented.max(axis=2) > REFERENCE_THRESHOLD).astype(np.uint8) * 255
    mask = mask_utils.close_mask(mask, 5)
    return mask_utils.keep_largest_component(mask)


def intersection_over_union(mask_a, mask_b):
    a, b = mask_a > 0, mask_b > 0
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 1.0


def evaluate(reference_dir, split, per_class):
    table = pd.read_csv(config.SPLITS_CSV)
    table = table[table["split"] == split].groupby("label").sample(
        per_class, random_state=config.RANDOM_SEED)

    records = []
    for _, row in table.iterrows():
        image = preprocess(load_image(config.PROJECT_ROOT / row["path"]))
        reference = load_reference_mask(reference_dir, row["path"], row["label"])
        try:
            iou = intersection_over_union(extract_leaf_mask(image.denoised), reference)
        except NoLeafFoundError:
            iou = 0.0
        records.append({"path": row["path"], "label": row["label"], "iou": iou})

    result = pd.DataFrame(records)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    result.to_csv(config.RESULTS_DIR / f"leaf_mask_iou_{split}.csv", index=False)

    print(f"Leaf mask vs PlantVillage reference ({split} split, {len(result)} images)")
    print(f"  mean IoU {result.iou.mean():.3f} | median {result.iou.median():.3f} "
          f"| share with IoU >= 0.9: {(result.iou >= 0.9).mean():.1%} "
          f"| share below 0.8: {(result.iou < 0.8).mean():.1%}")
    print(result.groupby("label").iou.agg(["mean", "median"]).round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=["train", "val"],
                        help="never 'test' while designing; test is for the final run only")
    parser.add_argument("--per-class", type=int, default=50)
    args = parser.parse_args()
    evaluate(args.reference, args.split, args.per_class)
