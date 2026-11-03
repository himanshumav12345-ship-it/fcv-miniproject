"""
make_figures.py - build every picture used in the report from the REAL pipeline and results files.

Nothing here is drawn by hand or invented: each image is produced by calling the project's own
functions on a real PlantVillage leaf, and each chart is read from results/*.csv.

Run from the project root:   python -m report.make_figures
Output:                      report/figures/*.png
"""
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from skimage.feature import graycomatrix, local_binary_pattern

from src import annotations, config, features, mask_utils, pipeline, refine, segmentation
from src.preprocess import enhance_contrast, load_image, preprocess
from src.visualize import paint_mask

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
DPI = 220
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 8, "axes.titlesize": 8,
                     "axes.linewidth": 0.6})

# One real leaf is used for the Chapter 5 snapshots, so the reader can follow it from figure to figure.
DEMO = next((config.RAW_DIR / "Early_blight").glob("*RS_Erly.B 7852.JPG"))


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved", name)


def strip(panels, name, width=7.0, columns=None, wspace=0.05):
    """A row of titled images. panels = [(title, image, cmap or None)]."""
    columns = columns or len(panels)
    rows = -(-len(panels) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(width, width / columns * 1.12 * rows),
                             gridspec_kw={"wspace": wspace, "hspace": 0.18})
    axes = np.atleast_1d(axes).ravel()
    for axis in axes:
        axis.axis("off")
    for axis, (title, image, cmap) in zip(axes, panels):
        axis.imshow(image, cmap=cmap, interpolation="nearest")
        axis.set_title(title, fontsize=7, pad=2)
    save(fig, name)


# -----------------------------------------------------------------------------
# Chapter 3: pipeline block diagram
# -----------------------------------------------------------------------------
def pipeline_diagram():
    boxes = [("Input leaf\nphoto", "#e8eef0"),
             ("Pre-processing\nGaussian blur\nCLAHE (L only)", "#dfe9ee"),
             ("Leaf mask\nLab chroma,\nOtsu, hysteresis", "#dfe9ee"),
             ("Lesion mask\ncolour distance\nfrom own green", "#dfe9ee"),
             ("Refinement\nopen, close,\nsmall-blob filter", "#dfe9ee"),
             ("Contours and\nCanny edges", "#dfe9ee"),
             ("90 features\ncolour, texture,\nshape", "#f3e3da"),
             ("SVM\nclassifier", "#f3e3da")]
    fig, ax = plt.subplots(figsize=(7.0, 1.9))
    ax.set_xlim(0, 8.0)
    ax.set_ylim(-1.1, 1.0)
    ax.axis("off")
    for i, (text, colour) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((i + 0.06, -0.38), 0.82, 0.9, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc=colour, ec="#33505a", lw=0.7))
        ax.text(i + 0.47, 0.07, text, ha="center", va="center", fontsize=5.2, linespacing=1.25)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(i + 1.06, 0.07), xytext=(i + 0.9, 0.07),
                        arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#33505a"))
    # outputs
    ax.add_patch(FancyBboxPatch((3.06, -1.02), 0.82, 0.42, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc="#fbece6", ec="#b8431f", lw=0.7))
    ax.text(3.47, -0.81, "Severity %\nlesion / leaf", ha="center", va="center", fontsize=5.2)
    ax.annotate("", xy=(3.47, -0.58), xytext=(3.47, -0.38), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#b8431f"))
    ax.add_patch(FancyBboxPatch((7.06, -1.02), 0.82, 0.42, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc="#fbece6", ec="#b8431f", lw=0.7))
    ax.text(7.47, -0.81, "Disease label\n+ confidence", ha="center", va="center", fontsize=5.2)
    ax.annotate("", xy=(7.47, -0.58), xytext=(7.47, -0.38), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#b8431f"))
    save(fig, "fig_pipeline.png")


# -----------------------------------------------------------------------------
# Chapter 5: one snapshot per function family, all on the same real leaf
# -----------------------------------------------------------------------------
def chapter5_figures():
    image = load_image(DEMO)
    prepared = preprocess(image)
    analysis = pipeline.analyze_array(image)
    original, denoised, enhanced = prepared.original, prepared.denoised, prepared.enhanced

    # 5.1 colour spaces ------------------------------------------------------
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    strip([("BGR photo", rgb(original), None),
           ("H (hue)", hsv[:, :, 0], "gray"), ("S (saturation)", hsv[:, :, 1], "gray"),
           ("V (value)", hsv[:, :, 2], "gray"), ("L (lightness)", lab[:, :, 0], "gray"),
           ("a (green-red)", lab[:, :, 1], "gray"), ("b (blue-yellow)", lab[:, :, 2], "gray")],
          "fig5_colour_spaces.png")

    # 5.2 Gaussian blur: zoom on a patch so the noise is visible --------------
    ys, xs = np.where(analysis.leaf_mask > 0)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    box = (slice(cy - 40, cy + 40), slice(cx - 40, cx + 40))
    panels = [("original (zoom)", rgb(original[box]), None)]
    for kernel in (3, 5, 9):
        blurred = cv2.GaussianBlur(original, (kernel, kernel), 0)
        suffix = " (used)" if kernel == config.DENOISE_KERNEL else ""
        panels.append((f"{kernel}x{kernel} kernel{suffix}", rgb(blurred[box]), None))
    strip(panels, "fig5_gaussian.png", width=6.0)

    # 5.3 CLAHE ---------------------------------------------------------------
    lightness_before = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)[:, :, 0]
    lightness_after = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)[:, :, 0]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95), gridspec_kw={"wspace": 0.12, "width_ratios": [1, 1, 1.25, 1]})
    for axis, (title, img, cmap) in zip(axes[[0, 1, 3]], [("L before", lightness_before, "gray"),
                                                          ("L after CLAHE", lightness_after, "gray"),
                                                          ("result (colour)", rgb(enhanced), None)]):
        axis.imshow(img, cmap=cmap)
        axis.set_title(title, fontsize=7, pad=2)
        axis.axis("off")
    inside = analysis.leaf_mask > 0
    axes[2].hist(lightness_before[inside], bins=64, range=(0, 255), alpha=0.65, label="before", color="#7a8f95")
    axes[2].hist(lightness_after[inside], bins=64, range=(0, 255), alpha=0.65, label="after", color="#b8431f")
    axes[2].set_title("histogram of L (leaf only)", fontsize=7, pad=2)
    axes[2].legend(fontsize=6, frameon=False)
    axes[2].tick_params(labelsize=6)
    axes[2].set_yticks([])
    save(fig, "fig5_clahe.png")

    # 5.4 thresholding: chroma distance, Otsu, hysteresis ----------------------
    lab_f = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB).astype(np.float32)
    background = segmentation.estimate_background_chroma(lab_f)
    distance = segmentation.chroma_distance_map(lab_f, background)
    high = segmentation.otsu_threshold(distance)
    low = config.HYSTERESIS_LOW_FACTOR * high
    sure = (distance > high).astype(np.uint8) * 255
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.7), gridspec_kw={"wspace": 0.32, "width_ratios": [1, 1.3, 1, 1, 1]})
    axes[0].imshow(distance, cmap="magma")
    axes[0].set_title("colour distance\nto background", fontsize=6.6, pad=2)
    axes[1].hist(distance.ravel(), bins=80, color="#7a8f95")
    axes[1].axvline(high, color="#b8431f", lw=1)
    axes[1].axvline(low, color="#17606a", lw=1, ls="--")
    axes[1].set_yscale("log")
    axes[1].set_title(f"histogram: Otsu {high:.1f},\nlow {low:.1f}", fontsize=6.6, pad=2)
    axes[1].tick_params(labelsize=6)
    axes[2].imshow(sure, cmap="gray")
    axes[2].set_title("above Otsu only", fontsize=6.6, pad=2)
    axes[3].imshow(analysis.leaf_mask, cmap="gray")
    axes[3].set_title("after hysteresis,\nopening, largest blob", fontsize=6.6, pad=2)
    axes[4].imshow(rgb(original * (analysis.leaf_mask[:, :, None] > 0)))
    axes[4].set_title("leaf cut out", fontsize=6.6, pad=2)
    for axis in axes[[0, 2, 3, 4]]:
        axis.axis("off")
    save(fig, "fig5_threshold.png")

    # 5.5 morphology ----------------------------------------------------------
    raw = analysis.raw_lesion_mask
    opened = mask_utils.open_mask(raw, config.LESION_OPEN_KERNEL)
    closed = mask_utils.close_mask(opened, config.LESION_CLOSE_KERNEL)
    cleaned = mask_utils.remove_small_blobs(closed, config.MIN_LESION_AREA_PIXELS)
    core = segmentation.leaf_core_mask(analysis.leaf_mask)
    rim = cv2.subtract(analysis.leaf_mask, core)
    strip([("raw lesion mask", raw, "gray"), ("after opening 3x3", opened, "gray"),
           ("after closing 3x3", closed, "gray"), ("small blobs removed", cleaned, "gray"),
           ("leaf mask", analysis.leaf_mask, "gray"), ("rim removed by erosion 7x7", rim, "gray")],
          "fig5_morphology.png")

    # 5.6 Canny ---------------------------------------------------------------
    grey = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    inside_grey = grey[core > 0]
    median = float(np.median(inside_grey))
    low_c = int(max(0, (1 - config.CANNY_SIGMA) * median))
    high_c = int(min(255, (1 + config.CANNY_SIGMA) * median))
    plain = cv2.Canny(grey, low_c, high_c)
    strip([("grey image (CLAHE)", grey, "gray"),
           (f"Canny over the whole image\nlow {low_c}, high {high_c}", plain, "gray"),
           ("used: leaf core only", analysis.edges, "gray")],
          "fig5_canny.png", width=5.4)

    # 5.7 contours and shape measures ------------------------------------------
    lesions = sorted(analysis.lesions, key=lambda item: -item.area)
    outline = original.copy()
    contours = [lesion.contour for lesion in lesions]
    cv2.drawContours(outline, contours, -1, (0, 0, 255), 1)
    if lesions:
        hull = cv2.convexHull(lesions[0].contour)
        cv2.drawContours(outline, [hull], -1, (0, 220, 255), 1)
        x, y, w, h = cv2.boundingRect(lesions[0].contour)
        cv2.rectangle(outline, (x - 1, y - 1), (x + w + 1, y + h + 1), (255, 160, 0), 1)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1, 1, 0.9], "wspace": 0.08})
    axes[0].imshow(rgb(outline))
    axes[0].set_title("findContours: red = lesions,\nyellow = hull, orange = box of the largest", fontsize=6.6, pad=2)
    axes[1].imshow(rgb(cv2.drawContours(np.zeros_like(original), contours, -1, (255, 255, 255), -1)))
    axes[1].set_title("lesion blobs\n(filled contours)", fontsize=6.6, pad=2)
    axes[0].axis("off")
    axes[1].axis("off")
    axes[2].axis("off")
    if lesions:
        big = lesions[0]
        hu = cv2.HuMoments(cv2.moments(analysis.lesion_mask, binaryImage=True)).flatten()
        lines = [f"largest lesion",
                 f"area  {big.area:.0f} px", f"perimeter  {big.perimeter:.1f} px",
                 f"circularity  {big.circularity:.2f}", f"solidity  {big.solidity:.2f}",
                 f"aspect ratio  {big.aspect_ratio:.2f}", "",
                 f"regions found  {len(lesions)}", "", "Hu moments (raw), whole mask",
                 *(f"h{i + 1}  {value:.2e}" for i, value in enumerate(hu[:3])), "..."]
        axes[2].text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=6.8, family="DejaVu Sans Mono",
                     linespacing=1.35)
    save(fig, "fig5_contours.png")

    # 5.8 connected components and K-means ------------------------------------
    count, labels = cv2.connectedComponents(cleaned)
    coloured = np.zeros((*labels.shape, 3), np.uint8)
    rng = np.random.default_rng(3)
    palette = rng.integers(60, 255, (count, 3))
    for index in range(1, count):
        coloured[labels == index] = palette[index]
    kmeans_mask = segmentation.lesion_mask_kmeans(denoised, analysis.leaf_mask)
    leaf_colours = lab_f[core > 0][:, 1:]
    cv2.setRNGSeed(config.RANDOM_SEED)
    _, cluster_labels, centres = cv2.kmeans(leaf_colours, config.KMEANS_CLUSTERS, None,
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5),
                                            3, cv2.KMEANS_PP_CENTERS)
    cluster_image = np.full(original.shape[:2], 255, np.uint8)
    order = np.argsort(centres[:, 0])
    shades = {int(cluster): int(50 + 90 * rank) for rank, cluster in enumerate(order)}
    cluster_image[core > 0] = [shades[int(c)] for c in cluster_labels.ravel()]
    strip([(f"labelled components: {count - 1} blobs", coloured, None),
           ("K-means, k=3, on (a, b)", cluster_image, "gray"),
           ("K-means 'lesion' cluster", kmeans_mask, "gray")],
          "fig5_components_kmeans.png", width=5.4)

    # 5.9 histogram, GLCM, LBP -------------------------------------------------
    inside = core > 0
    hue = hsv_core = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)[:, :, 0]
    histogram = cv2.calcHist([cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)], [0], core, [180], [0, 180]).ravel()
    quantised = features.quantise_leaf_gray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY), core)
    matrix = graycomatrix(quantised, [1], [0], levels=features.GLCM_LEVELS, symmetric=True, normed=False)[:, :, 0, 0].astype(float)
    matrix[0, :] = 0
    matrix[:, 0] = 0
    lbp = local_binary_pattern(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY), features.LBP_POINTS,
                               features.LBP_RADIUS, method="uniform")
    lbp_counts = np.bincount(lbp[inside].astype(int), minlength=features.LBP_BINS) / inside.sum()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.95), gridspec_kw={"wspace": 0.28})
    axes[0].bar(np.arange(180), histogram / histogram.sum(), width=1.0, color=plt.cm.hsv(np.arange(180) / 180))
    axes[0].axvspan(config.GREEN_HUE_MIN, config.GREEN_HUE_MAX, color="#17606a", alpha=0.12)
    axes[0].set_title("histogram of leaf-pixel hue\n(shaded = green band 35-85)", fontsize=6.6, pad=2)
    axes[0].set_xlabel("hue (OpenCV 0-179)", fontsize=6.5)
    axes[1].imshow(np.log1p(matrix[1:, 1:]), cmap="viridis", origin="upper")
    axes[1].set_title("GLCM (distance 1, 0 deg), log scale\nbright diagonal = smooth texture", fontsize=6.6, pad=2)
    axes[1].set_xlabel("neighbour level", fontsize=6.5)
    axes[1].set_ylabel("pixel level", fontsize=6.5)
    axes[2].bar(np.arange(features.LBP_BINS), lbp_counts, color="#17606a")
    axes[2].set_title("LBP (8 neighbours, radius 1)\nuniform-pattern histogram", fontsize=6.6, pad=2)
    axes[2].set_xlabel("pattern code", fontsize=6.5)
    for axis in axes:
        axis.tick_params(labelsize=6)
    save(fig, "fig5_texture.png")


# -----------------------------------------------------------------------------
# Chapter 4: lesion methods side by side on three annotated hold-out leaves
# -----------------------------------------------------------------------------
def lesion_method_figure():
    shapes = annotations.load_annotation_list()
    chosen = ["Septoria_leaf_spot_3", "Healthy_4", "Late_blight_3"]
    methods = ["hue_rule", "kmeans", "own_green"]
    titles = {"hue_rule": "hue rule", "kmeans": "K-means", "own_green": "own-green (used)"}
    fig, axes = plt.subplots(len(chosen), 4, figsize=(7.0, 5.4), gridspec_kw={"wspace": 0.04, "hspace": 0.2})
    for row, name in enumerate(chosen):
        path = config.PROJECT_ROOT / shapes.loc[shapes["id"] == name, "path"].iloc[0]
        image = load_image(path)
        truth = annotations.load_annotation_mask(name, image.shape)
        results = {m: pipeline.analyze_array(image, m) for m in methods}
        base = results["own_green"].original
        shown = rgb(base).copy()
        contours, _ = cv2.findContours(truth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(shown, contours, -1, (255, 235, 0), 1)
        core = segmentation.leaf_core_mask(results["own_green"].leaf_mask)
        truth = np.where(core > 0, truth, 0).astype(np.uint8)       # same rule as evaluate_lesions
        true_percent = 100 * np.count_nonzero(truth) / max(np.count_nonzero(core), 1)
        axes[row, 0].imshow(shown)
        axes[row, 0].set_title(f"{name.replace('_', ' ')}\nannotated lesions in yellow ({true_percent:.1f}%)", fontsize=6.4, pad=2)
        for column, method in enumerate(methods, start=1):
            analysis = results[method]
            axes[row, column].imshow(rgb(paint_mask(analysis.original, analysis.lesion_mask)))
            axes[row, column].set_title(f"{titles[method]}: {analysis.severity_percent:.1f}%", fontsize=6.6, pad=2)
        for axis in axes[row]:
            axis.axis("off")
    save(fig, "fig4_lesion_methods.png")


# -----------------------------------------------------------------------------
# Chapter 4: results charts (read from the saved result files)
# -----------------------------------------------------------------------------
def results_charts():
    ablation = pd.read_csv(config.RESULTS_DIR / "ablation_features.csv")
    leakage = pd.read_csv(config.RESULTS_DIR / "ablation_leakage.csv")
    model_selection = pd.read_csv(config.RESULTS_DIR / "model_selection.csv")

    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.1), gridspec_kw={"width_ratios": [2.4, 1], "wspace": 0.55})
    labels = ablation["features"].str.replace("+", "+\n", regex=False).str.replace(" ", "\n", n=1)
    axes[0].barh(range(len(ablation)), ablation["val_macro_f1"], color="#17606a")
    axes[0].set_yticks(range(len(ablation)))
    axes[0].set_yticklabels(ablation["features"], fontsize=6.3)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.4, 0.95)
    for i, value in enumerate(ablation["val_macro_f1"]):
        axes[0].text(value + 0.005, i, f"{value:.3f}", va="center", fontsize=6)
    axes[0].set_title("validation macro-F1\nby feature group", fontsize=6.8)
    axes[0].tick_params(axis="x", labelsize=6)
    axes[1].bar([0, 1], leakage["accuracy"], color=["#b8431f", "#17606a"], width=0.6)
    axes[1].set_ylim(0.8, 0.95)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["random\n(leaky)", "grouped"], fontsize=6.3)
    for i, value in enumerate(leakage["accuracy"]):
        axes[1].text(i, value + 0.003, f"{value * 100:.1f}%", ha="center", fontsize=6)
    axes[1].set_title("CV accuracy,\nsame model", fontsize=6.8)
    axes[1].tick_params(axis="y", labelsize=6)
    for axis in axes:
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    save(fig, "fig4_ablations.png")

    iou = pd.read_csv(config.RESULTS_DIR / "leaf_mask_iou_train.csv")["iou"]
    fig, axis = plt.subplots(figsize=(3.45, 1.7))
    axis.hist(iou, bins=30, range=(0.4, 1.0), color="#17606a")
    axis.axvline(iou.mean(), color="#b8431f", lw=1)
    axis.text(iou.mean() - 0.01, axis.get_ylim()[1] * 0.9, f"mean {iou.mean():.3f}", ha="right", color="#b8431f", fontsize=6.8)
    axis.set_xlabel("IoU with the PlantVillage background-removed mask", fontsize=7)
    axis.set_ylabel("images (of 300)", fontsize=7)
    axis.tick_params(labelsize=6.5)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    save(fig, "fig4_leaf_iou.png")


def gui_figures():
    folder = config.RESULTS_DIR / "gui"
    full = cv2.imread(str(folder / "gui_early_blight.png"))
    cv2.imwrite(str(OUT / "fig4_gui_result.png"), full)
    invalid = cv2.imread(str(folder / "gui_error_invalid_image.png"))
    no_leaf = cv2.imread(str(folder / "gui_error_no_leaf.png"))
    crops = [image[150:520, 30:420] for image in (invalid, no_leaf)]
    height = min(c.shape[0] for c in crops)
    cv2.imwrite(str(OUT / "fig4_gui_errors.png"), np.hstack([c[:height] for c in crops]))
    print("saved gui figures")


if __name__ == "__main__":
    pipeline_diagram()
    chapter5_figures()
    lesion_method_figure()
    results_charts()
    gui_figures()
