"""
dataset.py - build the project's dataset subset and the train/val/test split.

What it does, in plain words:
  1. Take the six tomato class folders of PlantVillage.
  2. Keep at most IMAGES_PER_CLASS random images per class (balanced classes).
  3. Copy them into data/raw/<class>/.
  4. Split into train / val / test WITHOUT leaking near-duplicate images.
  5. Write data/splits/splits.csv  (path, label, class_id, group, split).

Run:
    python -m src.dataset --source /path/to/PlantVillage-Dataset/raw/color
"""
import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src import config

# File names look like  "<uuid>___RS_Erly.B 8389.JPG"
#                        ^ugly id   ^site code  ^running image number
_NAME_PATTERN = re.compile(r".+___(.*?)\s*(\d+)(?:\.\d+)?\.jpe?g$", re.IGNORECASE)


def parse_image_number(filename):
    """Return (site_code, image_number) from a PlantVillage file name, or None.

    The site code says which photo session/place the image came from; the number
    is a running counter. Consecutive numbers are usually the same leaf shot in a
    different orientation, which is why we need these two values.
    """
    match = _NAME_PATTERN.match(filename)
    if match is None:
        return None
    return match.group(1).strip(), int(match.group(2))


def make_group_id(class_name, filename):
    """Group id = class + site code + block of consecutive image numbers.

    All images with the same group id are forced into the same split.
    A handful of odd file names cannot be parsed; each becomes its own group.
    """
    parsed = parse_image_number(filename)
    if parsed is None:
        return f"{class_name}|unparsed|{filename}"
    site_code, number = parsed
    return f"{class_name}|{site_code}|{number // config.ID_BLOCK_SIZE}"


def list_source_images(source_dir):
    """Return {class_name: sorted list of image paths} for the six chosen classes."""
    images = {}
    for folder_name, class_name in config.CLASS_FOLDERS.items():
        folder = source_dir / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected class folder not found: {folder}")
        images[class_name] = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")
        )
    return images


def sample_images(paths, rng):
    """Randomly keep at most IMAGES_PER_CLASS images (all of them if fewer)."""
    if len(paths) <= config.IMAGES_PER_CLASS:
        return list(paths)
    return sorted(rng.sample(paths, config.IMAGES_PER_CLASS))


def assign_groups_to_splits(group_sizes, rng):
    """Assign whole groups to 'test', 'val' or 'train' for ONE class.

    group_sizes: {group_id: number_of_images}. Groups are shuffled, then dealt
    into test first, then val, until each reaches its share of the class's images;
    the rest is train. Because groups are indivisible the shares are approximate.
    """
    total = sum(group_sizes.values())
    test_target = config.TEST_FRACTION * total
    val_target = config.VAL_FRACTION * total

    group_ids = sorted(group_sizes)   # sort first so the shuffle is reproducible
    rng.shuffle(group_ids)

    assignment, test_count, val_count = {}, 0, 0
    for group_id in group_ids:
        if test_count < test_target:
            assignment[group_id] = "test"
            test_count += group_sizes[group_id]
        elif val_count < val_target:
            assignment[group_id] = "val"
            val_count += group_sizes[group_id]
        else:
            assignment[group_id] = "train"
    return assignment


def build_dataset(source_dir):
    rng = random.Random(config.RANDOM_SEED)
    source_images = list_source_images(source_dir)

    rows = []
    for class_id, class_name in enumerate(config.CLASS_NAMES):
        chosen = sample_images(source_images[class_name], rng)

        # copy the chosen images into the project
        target_folder = config.RAW_DIR / class_name
        target_folder.mkdir(parents=True, exist_ok=True)
        for path in chosen:
            shutil.copy2(path, target_folder / path.name)

        # group the images, then split by group
        group_of = {p.name: make_group_id(class_name, p.name) for p in chosen}
        group_sizes = defaultdict(int)
        for group_id in group_of.values():
            group_sizes[group_id] += 1
        split_of_group = assign_groups_to_splits(group_sizes, rng)

        for path in chosen:
            group_id = group_of[path.name]
            rows.append({
                "path": (config.RAW_DIR / class_name / path.name)
                        .relative_to(config.PROJECT_ROOT).as_posix(),
                "label": class_name,
                "class_id": class_id,
                "group": group_id,
                "split": split_of_group[group_id],
            })

    table = pd.DataFrame(rows)
    config.SPLITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.SPLITS_CSV, index=False)
    return table


def print_summary(table):
    counts = table.groupby(["label", "split"]).size().unstack(fill_value=0)
    counts = counts.reindex(config.CLASS_NAMES)[["train", "val", "test"]]
    counts.loc["TOTAL"] = counts.sum()
    print(counts.to_string())
    print(f"\ngroups: {table['group'].nunique()}  images: {len(table)}")
    # sanity check: no group may appear in two splits
    splits_per_group = table.groupby("group")["split"].nunique()
    assert (splits_per_group == 1).all(), "a group leaked across splits!"
    print("check passed: every group lives in exactly one split")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--source", required=True, type=Path,
                        help="PlantVillage 'raw/color' folder containing Tomato___* folders")
    args = parser.parse_args()
    print_summary(build_dataset(args.source))
