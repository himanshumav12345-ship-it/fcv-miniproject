"""
evaluate_lesions.py - which lesion-segmentation method is best, measured on annotated masks.

For every annotated image and every method we compare the method's cleaned lesion mask
with the hand-drawn mask, ONLY inside the rim-trimmed leaf (the same region the methods
are allowed to work in).

Metrics (P = predicted lesion pixels, G = annotated lesion pixels):
  Dice      = 2|P and G| / (|P| + |G|)      overlap, 1 = perfect
  IoU       = |P and G| / |P or G|          stricter overlap
  precision = |P and G| / |P|               of what we flagged, how much is really lesion
  recall    = |P and G| / |G|               of the real lesion, how much we found
  Healthy leaves have an EMPTY annotation, so overlap scores are undefined for them.
  There we report the false-alarm area: % of the leaf wrongly flagged (ideal = 0).

Run:
    python -m src.evaluate_lesions --role dev          # while designing / tuning
    python -m src.evaluate_lesions --role holdout      # final numbers, run once
"""
import argparse

import cv2
import numpy as np
import pandas as pd

from src import annotations, config, mask_utils, pipeline, refine
from src.preprocess import load_image
from src.segmentation import LESION_METHODS, leaf_core_mask
from src.visualize import add_title, paint_mask


def overlap_counts(predicted, truth):
    """Pixel counts of the overlap between two 0/255 masks."""
    p, g = predicted > 0, truth > 0
    return {"tp": int(np.count_nonzero(p & g)), "pred": int(np.count_nonzero(p)),
            "truth": int(np.count_nonzero(g)), "union": int(np.count_nonzero(p | g))}


def scores_from_counts(counts):
    tp, pred, truth, union = counts["tp"], counts["pred"], counts["truth"], counts["union"]
    return {"dice": 2 * tp / (pred + truth) if pred + truth else np.nan,
            "iou": tp / union if union else np.nan,
            "precision": tp / pred if pred else np.nan,
            "recall": tp / truth if truth else np.nan}


def evaluate_one_image(row, method_name):
    """Run one lesion method on one annotated image and score it."""
    image = load_image(config.PROJECT_ROOT / row["path"])
    analysis = pipeline.analyze_array(image, lesion_method=method_name)
    truth = annotations.load_annotation_mask(row["id"], image.shape)
    truth = cv2.resize(truth, analysis.leaf_mask.shape[::-1], interpolation=cv2.INTER_NEAREST)

    region = leaf_core_mask(analysis.leaf_mask)
    truth_in_leaf = cv2.bitwise_and(truth, region)
    predicted = cv2.bitwise_and(analysis.lesion_mask, region)

    counts = overlap_counts(predicted, truth_in_leaf)
    record = {"id": row["id"], "label": row["label"], "role": row["role"], "method": method_name,
              "is_healthy": row["label"] == "Healthy",
              "leaf_area": mask_utils.area_in_pixels(region),
              "false_alarm_percent": 100.0 * counts["pred"] / max(mask_utils.area_in_pixels(region), 1),
              "true_percent": 100.0 * counts["truth"] / max(mask_utils.area_in_pixels(region), 1),
              "predicted_percent": 100.0 * counts["pred"] / max(mask_utils.area_in_pixels(region), 1)}
    record.update(counts)
    record.update(scores_from_counts(counts))
    return record, analysis, truth_in_leaf


def evaluate_methods(role, methods):
    annotated = annotations.load_annotation_list()
    annotated = annotated[annotated["role"] == role]
    records, details = [], {}
    for _, row in annotated.iterrows():
        for method in methods:
            record, analysis, truth = evaluate_one_image(row, method)
            records.append(record)
            details[(row["id"], method)] = (analysis, truth)
    return pd.DataFrame(records), details


def summarise(table):
    """One row per method: mean over diseased images, pooled scores and healthy false alarms."""
    rows = []
    for method, group in table.groupby("method", sort=False):
        diseased = group[~group["is_healthy"]]
        healthy = group[group["is_healthy"]]
        pooled = scores_from_counts({k: int(diseased[k].sum()) for k in ("tp", "pred", "truth", "union")})
        rows.append({
            "method": method,
            "mean_dice": diseased["dice"].mean(),
            "mean_iou": diseased["iou"].mean(),
            "mean_precision": diseased["precision"].mean(),
            "mean_recall": diseased["recall"].mean(),
            "pooled_dice": pooled["dice"],
            "healthy_false_alarm_%": healthy["false_alarm_percent"].mean(),
            "severity_mae_pp": (diseased["predicted_percent"] - diseased["true_percent"]).abs().mean(),
            "n_diseased": len(diseased), "n_healthy": len(healthy)})
    return pd.DataFrame(rows).set_index("method")


def save_failure_montage(table, details, method, path, worst=4):
    """Rows = the worst diseased images for a method. Columns: original | annotation |
    prediction | errors (blue = missed lesion, red = false alarm)."""
    diseased = table[(table["method"] == method) & (~table["is_healthy"])].sort_values("dice")
    rows = []
    for _, record in diseased.head(worst).iterrows():
        analysis, truth = details[(record["id"], method)]
        region = leaf_core_mask(analysis.leaf_mask)
        predicted = cv2.bitwise_and(analysis.lesion_mask, region)
        errors = analysis.original.copy()
        errors[(predicted > 0) & (truth == 0)] = (0, 0, 255)       # false alarm: red
        errors[(predicted == 0) & (truth > 0)] = (255, 100, 0)     # missed: blue
        errors[(predicted > 0) & (truth > 0)] = (0, 200, 0)        # hit: green
        tiles = [("original", analysis.original), ("annotation", paint_mask(analysis.original, truth, (0, 200, 0))),
                 ("prediction", paint_mask(analysis.original, predicted)),
                 (f"{record['id']} Dice {record['dice']:.2f}", errors)]
        rows.append(np.hstack([add_title(cv2.resize(image, (220, 220)), title) for title, image in tiles]))
    cv2.imwrite(str(path), np.vstack(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", default="dev", choices=["dev", "holdout"])
    parser.add_argument("--tag", default="", help="suffix for output file names")
    args = parser.parse_args()

    methods = list(LESION_METHODS)
    table, details = evaluate_methods(args.role, methods)
    summary = summarise(table)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    suffix = f"_{args.role}{args.tag}"
    table.to_csv(config.RESULTS_DIR / f"lesion_per_image{suffix}.csv", index=False)
    summary.to_csv(config.RESULTS_DIR / f"lesion_method_comparison{suffix}.csv")
    for method in methods:
        save_failure_montage(table, details, method,
                             config.RESULTS_DIR / f"lesion_failures_{method}{suffix}.png")

    pd.set_option("display.width", 200)
    print(f"Lesion methods on the {args.role} annotations "
          f"({summary['n_diseased'].iloc[0]} diseased + {summary['n_healthy'].iloc[0]} healthy images)")
    print(summary.drop(columns=["n_diseased", "n_healthy"]).round(3).to_string())


if __name__ == "__main__":
    main()
