"""
visualize.py - SEE what every stage did.

Saves one picture per stage plus a montage, so you can point at the screen in the viva
and say "this is the input of stage 3, this is its output".

Run on one image:      python -m src.visualize --image path/to/leaf.jpg
Run on random samples: python -m src.visualize --sample 6 --split val
Pictures go to results/stages/<image name>/
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import config, pipeline
from src.errors import InvalidImageError, NoLeafFoundError
from src.refine import draw_lesion_overlay

TILE_SIZE = 256


def paint_mask(image_bgr, mask, colour=(0, 0, 255), strength=0.55):
    """Tint the masked pixels of an image with a colour (semi-transparent)."""
    tint = np.zeros_like(image_bgr)
    tint[:] = colour
    blended = cv2.addWeighted(image_bgr, 1 - strength, tint, strength, 0)
    result = image_bgr.copy()
    result[mask > 0] = blended[mask > 0]
    return result


def stage_panels(analysis):
    """List of (title, BGR image) for every pipeline stage, in order."""
    original = analysis.original
    leaf_only = original.copy()
    leaf_only[analysis.leaf_mask == 0] = 0
    overlay = draw_lesion_overlay(original, analysis.leaf_mask, analysis.lesions)
    return [
        ("1 original", original),
        ("2 denoised (Gaussian)", analysis.denoised),
        ("3 contrast (CLAHE)", analysis.enhanced),
        ("4 leaf mask", leaf_only),
        ("5 raw lesion mask", paint_mask(original, analysis.raw_lesion_mask)),
        ("6 cleaned lesion mask", paint_mask(original, analysis.lesion_mask)),
        ("7 Canny edge map", cv2.cvtColor(analysis.edges, cv2.COLOR_GRAY2BGR)),
        (f"8 contours | {analysis.severity_percent:.1f}% {analysis.severity_level}", overlay),
    ]


def add_title(tile, title):
    """Put a title strip above an image."""
    strip = np.full((22, tile.shape[1], 3), 255, np.uint8)
    cv2.putText(strip, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return np.vstack([strip, tile])


def make_montage(panels, columns=4):
    """Arrange titled panels in a grid."""
    tiles = [add_title(cv2.resize(image, (TILE_SIZE, TILE_SIZE)), title) for title, image in panels]
    while len(tiles) % columns:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    return np.vstack(rows)


def save_stage_images(analysis, folder, stem, save_each=True):
    """Write every stage as its own PNG and one montage.png into `folder`."""
    folder = Path(folder) / stem
    folder.mkdir(parents=True, exist_ok=True)
    panels = stage_panels(analysis)
    if save_each:
        for title, image in panels:
            name = title.split("|")[0].strip().replace(" ", "_").replace("(", "").replace(")", "")
            cv2.imwrite(str(folder / f"{name}.png"), image)
    cv2.imwrite(str(folder / "montage.png"), make_montage(panels))
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--image", type=Path, help="analyse this one image")
    parser.add_argument("--sample", type=int, default=0, help="analyse N random dataset images")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--method", default=config.DEFAULT_LESION_METHOD,
                        choices=["hue_rule", "kmeans", "own_green"])
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "stages")
    args = parser.parse_args()

    paths = []
    if args.image:
        paths.append(args.image)
    if args.sample:
        table = pd.read_csv(config.SPLITS_CSV)
        table = table[table["split"] == args.split].sample(args.sample, random_state=config.RANDOM_SEED)
        paths += [config.PROJECT_ROOT / p for p in table["path"]]

    for path in paths:
        try:
            analysis = pipeline.analyze_file(path, args.method)
        except (InvalidImageError, NoLeafFoundError) as error:
            print(f"{path.name}: skipped ({error})")
            continue
        short_name = path.stem.split("___")[-1].replace(" ", "_")   # drop the long uuid prefix
        folder = save_stage_images(analysis, args.out, short_name)
        print(f"{path.name}: severity {analysis.severity_percent:.1f}% "
              f"({analysis.severity_level}), {len(analysis.lesions)} lesions -> {folder}")


if __name__ == "__main__":
    main()
