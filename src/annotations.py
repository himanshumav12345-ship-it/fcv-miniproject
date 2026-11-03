"""
annotations.py - hand-drawn lesion masks used to MEASURE how good lesion segmentation is.

PlantVillage has no lesion ground truth, so we make a small set ourselves.
Annotations are stored as shapes (polygons and ellipses) in ONE editable JSON file,
data/annotations/shapes.json, and turned into masks when needed:

    {"Early_blight_1": {"annotator": "...", "polygons": [[[x, y], ...]],
                         "ellipses": [[cx, cy, rx, ry, angle_degrees]]}, ...}

ANNOTATION PROTOCOL (so everybody draws the same thing)
  Lesion = visibly diseased leaf tissue:
    * dark brown / black / tan dead (necrotic) spots and patches
    * grey, water-soaked or dead areas (late blight)
    * yellow (chlorotic) patches or halos clearly attached to a spot or patch
  NOT lesion: veins, shadows, folds, specular highlights, the leaf rim, background,
    and the uniform pale or yellow-green colour of a whole leaf.
  When unsure about a faint feature, leave it out (conservative annotation).

Commands:
    python -m src.annotations --prepare           pick the images to annotate
    python -m src.annotations --grid  <id>        grid view with coordinates (for drawing)
    python -m src.annotations --overlay <id>      check a finished annotation
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import config

ANNOTATION_LIST_CSV = config.ANNOTATION_DIR / "to_annotate.csv"
SHAPES_JSON = config.ANNOTATION_DIR / "shapes.json"


# -----------------------------------------------------------------------------
# Which images get annotated
# -----------------------------------------------------------------------------
def prepare_annotation_list(per_class=4, split="val"):
    """Pick `per_class` random images of every class from the `split` images.

    The first half of each class is the DEVELOPMENT set (used to choose and tune
    the lesion method); the second half is the HOLD-OUT set (used once for the
    final report). The test split of the classifier is never touched.
    """
    table = pd.read_csv(config.SPLITS_CSV)
    table = table[table["split"] == split]
    # one image per group, so no two annotated images are near-duplicates of the same
    # leaf (otherwise the development and hold-out sets would leak into each other)
    table = table.groupby("group").sample(1, random_state=config.RANDOM_SEED)
    rows = []
    for label in config.CLASS_NAMES:
        chosen = table[table["label"] == label].sample(per_class, random_state=config.RANDOM_SEED)
        for position, (_, row) in enumerate(chosen.iterrows()):
            rows.append({"id": f"{label}_{position + 1}", "path": row["path"], "label": label,
                         "role": "dev" if position < per_class // 2 else "holdout"})
    config.ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(ANNOTATION_LIST_CSV, index=False)
    return result


def load_annotation_list():
    return pd.read_csv(ANNOTATION_LIST_CSV)


# -----------------------------------------------------------------------------
# Shapes <-> masks
# -----------------------------------------------------------------------------
def load_shapes():
    if not SHAPES_JSON.exists():
        return {}
    return json.loads(SHAPES_JSON.read_text())


def save_shapes(shapes):
    config.ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    SHAPES_JSON.write_text(json.dumps(shapes, indent=1))


def shapes_to_mask(shape_record, image_shape):
    """Rasterise polygons and ellipses into a 0/255 mask of the given (height, width)."""
    mask = np.zeros(image_shape[:2], np.uint8)
    for polygon in shape_record.get("polygons", []):
        points = np.round(np.array(polygon)).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    for cx, cy, rx, ry, angle in shape_record.get("ellipses", []):
        cv2.ellipse(mask, (int(round(cx)), int(round(cy))), (int(round(rx)), int(round(ry))),
                    angle, 0, 360, 255, thickness=cv2.FILLED)
    return mask


def load_annotation_mask(annotation_id, image_shape):
    """Mask for one annotated image, or None if it has not been annotated yet."""
    record = load_shapes().get(annotation_id)
    return None if record is None else shapes_to_mask(record, image_shape)


# -----------------------------------------------------------------------------
# Views that make drawing and checking easier
# -----------------------------------------------------------------------------
def make_grid_view(image, scale=4, step=16):
    """Enlarge the image and draw a labelled coordinate grid (original-pixel units)."""
    big = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for position in range(0, image.shape[1] + 1, step):
        major = position % (step * 2) == 0
        colour = (255, 255, 255) if major else (200, 200, 200)
        cv2.line(big, (position * scale, 0), (position * scale, big.shape[0]), colour, 1)
        cv2.line(big, (0, position * scale), (big.shape[1], position * scale), colour, 1)
        if major:
            cv2.putText(big, str(position), (position * scale + 2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(big, str(position), (2, position * scale + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return big


def make_overlay_view(image, mask, scale=4):
    """Enlarged image with the annotation outlined in red, for checking."""
    big = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    big_mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(big_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(big, contours, -1, (0, 0, 255), 1)
    return big


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--grid")
    parser.add_argument("--overlay")
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "annotation_views")
    args = parser.parse_args()

    if args.prepare:
        print(prepare_annotation_list().to_string(index=False))
    for annotation_id, kind in ((args.grid, "grid"), (args.overlay, "overlay")):
        if not annotation_id:
            continue
        row = load_annotation_list().set_index("id").loc[annotation_id]
        image = cv2.imread(str(config.PROJECT_ROOT / row["path"]))
        if kind == "grid":
            view = make_grid_view(image)
        else:
            view = make_overlay_view(image, load_annotation_mask(annotation_id, image.shape))
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"{annotation_id}_{kind}.png"
        cv2.imwrite(str(path), view)
        print(path)


if __name__ == "__main__":
    main()
