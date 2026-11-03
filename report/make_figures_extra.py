"""
make_figures_extra.py - additional report pictures and the numbers printed beside them.

Everything is computed from the project's own functions, the cached feature tables and the saved
model. Nothing is drawn by hand. Numbers that the text quotes are written to
report/figures/extra_numbers.json, so build_report.py reads them instead of re-typing them.

Run from the project root:   python -m report.make_figures_extra
"""
import json
import time
from pathlib import Path

import cv2
import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

from src import config, features, mask_utils, pipeline, predict, segmentation
from src.features import feature_group
from src.preprocess import load_image, preprocess
from report.make_figures import DEMO, OUT, rgb, save, strip

plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 8, "axes.titlesize": 8,
                     "axes.linewidth": 0.6})
NUMBERS = {}
CLASS_LABEL = {"Healthy": "Healthy", "Bacterial_spot": "Bacterial spot", "Early_blight": "Early blight",
               "Late_blight": "Late blight", "Leaf_Mold": "Leaf mold", "Septoria_leaf_spot": "Septoria leaf spot"}
FAMILY_COLOUR = {"colour": "#17606a", "texture": "#b8431f", "shape": "#8a8f3c"}


# -----------------------------------------------------------------------------
# Chapter 3: what the six classes look like
# -----------------------------------------------------------------------------
def dataset_samples():
    splits = pd.read_csv(config.SPLITS_CSV)
    train = splits[splits["split"] == "train"]
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(4, 6, figsize=(7.0, 4.85), gridspec_kw={"wspace": 0.03, "hspace": 0.03})
    for column, name in enumerate(config.CLASS_NAMES):
        paths = train[train["label"] == name]["path"].to_numpy()
        for row, path in enumerate(rng.choice(paths, 4, replace=False)):
            axis = axes[row, column]
            axis.imshow(rgb(load_image(ROOT / path)))
            axis.axis("off")
            if row == 0:
                axis.set_title(CLASS_LABEL[name], fontsize=7.5, pad=3)
    save(fig, "fig3_samples.png")


# -----------------------------------------------------------------------------
# Chapter 4: one test leaf per class through the whole system
# -----------------------------------------------------------------------------
def class_gallery():
    splits = pd.read_csv(config.SPLITS_CSV)
    test = splits[splits["split"] == "test"]
    rng = np.random.default_rng(11)
    rows = []
    for name in config.CLASS_NAMES:
        path = ROOT / rng.choice(test[test["label"] == name]["path"].to_numpy())
        rows.append((name, path))
    fig, axes = plt.subplots(4, 6, figsize=(7.0, 5.0), gridspec_kw={"wspace": 0.03, "hspace": 0.05})
    gallery = []
    for column, (name, path) in enumerate(rows):
        result = predict.predict_file(path)
        analysis = result.analysis
        gallery.append({"true": name, "predicted": result.disease, "confidence": round(result.confidence, 2),
                        "severity": round(analysis.severity_percent, 1), "level": analysis.severity_level,
                        "lesions": len(analysis.lesions), "file": path.name})
        panels = [rgb(analysis.original),
                  rgb(__import__("src.visualize", fromlist=["paint_mask"]).paint_mask(analysis.original, analysis.lesion_mask)),
                  analysis.edges,
                  rgb(__import__("src.refine", fromlist=["draw_lesion_overlay"]).draw_lesion_overlay(
                      analysis.original, analysis.leaf_mask, analysis.lesions))]
        for row, panel in enumerate(panels):
            axis = axes[row, column]
            axis.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            axis.axis("off")
        axes[0, column].set_title(CLASS_LABEL[name], fontsize=7.5, pad=3)
        axes[3, column].text(0.5, -0.08, f"{CLASS_LABEL[result.disease]} {result.confidence:.0%}\n"
                             f"severity {analysis.severity_percent:.1f}%", transform=axes[3, column].transAxes,
                             ha="center", va="top", fontsize=6.3, linespacing=1.3)
    for row, label in enumerate(["photo", "diseased pixels", "Canny edges", "regions measured"]):
        axes[row, 0].text(-0.04, 0.5, label, transform=axes[row, 0].transAxes, rotation=90, ha="right",
                          va="center", fontsize=6.8)
    save(fig, "fig4_class_gallery.png")
    NUMBERS["gallery"] = gallery


# -----------------------------------------------------------------------------
# Chapter 4: severity by class, confidence of right and wrong predictions
# -----------------------------------------------------------------------------
def test_predictions():
    saved = joblib.load(config.MODELS_DIR / "classifier.joblib")
    table = pd.read_csv(config.DATA_DIR / "features_own_green.csv")
    table = table[table["status"] == "ok"]
    test = table[table["split"] == "test"].reset_index(drop=True)
    proba = saved["model"].predict_proba(test[saved["feature_names"]].to_numpy())
    best = proba.argmax(axis=1)
    correct = best == test["class_id"].to_numpy()
    return saved, table, test, proba, best, correct


def severity_and_confidence():
    saved, table, test, proba, best, correct = test_predictions()
    severity = 100 * test["shape_lesion_area_ratio"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), gridspec_kw={"wspace": 0.22, "width_ratios": [1.25, 1]})
    ax = axes[0]
    data = [severity[test["label"] == name] for name in config.CLASS_NAMES]
    box = ax.boxplot(data, orientation="vertical", widths=0.55, patch_artist=True, showfliers=False,
                     medianprops=dict(color="#b8431f", lw=1.2))
    for patch in box["boxes"]:
        patch.set(facecolor="#d9e5e8", edgecolor="#33505a", lw=0.7)
    for whisker in box["whiskers"] + box["caps"]:
        whisker.set(color="#33505a", lw=0.7)
    rng = np.random.default_rng(0)
    for i, values in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(values)), values, s=2.0, color="#17606a", alpha=0.35, lw=0)
    for limit in (5, 15, 35):
        ax.axhline(limit, color="#8a8f95", lw=0.5, ls=":")
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([CLASS_LABEL[n].replace(" ", "\n", 1) for n in config.CLASS_NAMES], fontsize=6.3)
    ax.set_ylabel("measured severity (% of leaf)", fontsize=7)
    ax.set_title("Severity by true class (757 test images)", fontsize=7.5)
    ax.tick_params(labelsize=6.3)
    ax = axes[1]
    confidence = proba.max(axis=1)
    bins = np.linspace(0.2, 1.0, 17)
    ax.hist(confidence[correct], bins=bins, color="#17606a", alpha=0.8, label=f"correct ({correct.sum()})")
    ax.hist(confidence[~correct], bins=bins, color="#b8431f", alpha=0.85, label=f"wrong ({(~correct).sum()})")
    ax.set_xlabel("confidence of the predicted class", fontsize=7)
    ax.set_ylabel("images", fontsize=7)
    ax.set_title("Confidence of right and wrong labels", fontsize=7.5)
    ax.legend(fontsize=6.3, frameon=False, loc="upper left")
    ax.tick_params(labelsize=6.3)
    save(fig, "fig4_severity_confidence.png")
    medians = {name: float(np.median(severity[test["label"] == name])) for name in config.CLASS_NAMES}
    p90 = {name: float(np.percentile(severity[test["label"] == name], 90)) for name in config.CLASS_NAMES}
    healthy = severity[test["label"] == "Healthy"]
    NUMBERS["healthy_above_5"] = float((healthy > 5).mean())
    NUMBERS["healthy_n"] = int(len(healthy))
    NUMBERS["severity_median"] = medians
    NUMBERS["severity_p90"] = p90
    NUMBERS["confidence"] = {
        "median_correct": float(np.median(confidence[correct])), "median_wrong": float(np.median(confidence[~correct])),
        "share_correct_above_0.8": float((confidence[correct] > 0.8).mean()),
        "share_wrong_below_0.6": float((confidence[~correct] < 0.6).mean()),
        "n_wrong": int((~correct).sum()),
        "accuracy_when_conf_above_0.8": float(correct[confidence > 0.8].mean()),
        "share_conf_above_0.8": float((confidence > 0.8).mean()),
        "accuracy_when_conf_below_0.6": float(correct[confidence < 0.6].mean()),
        "n_conf_below_0.6": int((confidence < 0.6).sum()),
    }


# -----------------------------------------------------------------------------
# Chapter 4: which features matter, and how the classes sit in feature space
# -----------------------------------------------------------------------------
def feature_view():
    saved, table, test, *_ = test_predictions()
    names = saved["feature_names"]
    train = table[table["split"] == "train"]
    scores, _ = f_classif(train[names].to_numpy(), train["class_id"].to_numpy())
    order = np.argsort(scores)[::-1][:15]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"wspace": 0.55, "width_ratios": [1.15, 1]})
    ax = axes[0]
    labels = [names[i].replace("colour_", "").replace("texture_", "").replace("shape_", "").replace("_", " ")
              for i in order]
    colours = [FAMILY_COLOUR[feature_group(names[i])] for i in order]
    ax.barh(range(15), scores[order], color=colours, height=0.7)
    ax.set_yticks(range(15))
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("ANOVA F-value on the training set", fontsize=7)
    ax.set_title("Fifteen most class-separating features", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=6.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=f) for f, c in FAMILY_COLOUR.items()], fontsize=6.3,
              frameon=False, loc="lower right")
    scaler = StandardScaler().fit(train[names].to_numpy())
    pca = PCA(n_components=2, random_state=config.RANDOM_SEED).fit(scaler.transform(train[names].to_numpy()))
    points = pca.transform(scaler.transform(test[names].to_numpy()))
    ax = axes[1]
    palette = ["#17606a", "#b8431f", "#8a8f3c", "#5b3a8c", "#d49a1f", "#3b86c4"]
    for class_id, name in enumerate(config.CLASS_NAMES):
        chosen = test["class_id"].to_numpy() == class_id
        ax.scatter(points[chosen, 0], points[chosen, 1], s=4, color=palette[class_id], alpha=0.65, lw=0,
                   label=CLASS_LABEL[name])
    ax.set_xlabel(f"component 1 ({100 * pca.explained_variance_ratio_[0]:.0f}% of variance)", fontsize=7)
    ax.set_ylabel(f"component 2 ({100 * pca.explained_variance_ratio_[1]:.0f}%)", fontsize=7)
    ax.set_title("Test images in the first two PCA axes", fontsize=7.5)
    ax.legend(fontsize=5.6, frameon=False, markerscale=1.6, loc="best", handletextpad=0.2)
    ax.tick_params(labelsize=6.3)
    save(fig, "fig4_features_pca.png")
    NUMBERS["top_features"] = [{"name": names[i], "F": float(scores[i])} for i in order]
    counts = {}
    for i in order:
        counts[feature_group(names[i])] = counts.get(feature_group(names[i]), 0) + 1
    NUMBERS["top15_by_family"] = counts
    NUMBERS["pca_variance"] = [float(v) for v in pca.explained_variance_ratio_]


# -----------------------------------------------------------------------------
# Chapter 5: extra laboratory techniques on the same demo leaf
# -----------------------------------------------------------------------------
def psnr(clean, other):
    mse = np.mean((clean.astype(np.float64) - other.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def filter_comparison():
    image = load_image(DEMO)
    clean = cv2.GaussianBlur(image, (3, 3), 0) if False else image
    rng = np.random.default_rng(5)
    gaussian_noise = np.clip(image.astype(np.float64) + rng.normal(0, 15, image.shape), 0, 255).astype(np.uint8)
    salt = image.copy()
    hits = rng.random(image.shape[:2])
    salt[hits < 0.02] = 0
    salt[hits > 0.98] = 255
    filters = [("none", lambda x: x), ("box 5x5 (blur)", lambda x: cv2.blur(x, (5, 5))),
               ("Gaussian 5x5", lambda x: cv2.GaussianBlur(x, (5, 5), 0)),
               ("median 5", lambda x: cv2.medianBlur(x, 5)),
               ("bilateral (9, 50, 50)", lambda x: cv2.bilateralFilter(x, 9, 50, 50))]
    ys, xs = np.where(segmentation_leaf_mask(image) > 0)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    box = (slice(cy - 36, cy + 36), slice(cx - 36, cx + 36))
    fig, axes = plt.subplots(2, 5, figsize=(6.6, 2.85), gridspec_kw={"wspace": 0.04, "hspace": 0.28})
    table = []
    for row, (noise_name, noisy) in enumerate([("Gaussian noise (sigma 15)", gaussian_noise),
                                               ("salt-and-pepper noise (4%)", salt)]):
        for column, (name, function) in enumerate(filters):
            result = function(noisy)
            start = time.perf_counter()
            for _ in range(50):
                function(noisy)
            milliseconds = (time.perf_counter() - start) / 50 * 1000
            score = psnr(image, result)
            axes[row, column].imshow(rgb(result[box]))
            axes[row, column].axis("off")
            axes[row, column].set_title(f"{name}\n{score:.1f} dB" if column else f"{name}\n{score:.1f} dB",
                                        fontsize=6, pad=2, linespacing=1.15)
            table.append({"noise": noise_name, "filter": name, "psnr": round(score, 1), "ms": round(milliseconds, 2)})
        axes[row, 0].text(-0.06, 0.5, noise_name.split(" (")[0], transform=axes[row, 0].transAxes, rotation=90,
                          ha="right", va="center", fontsize=6.3)
    save(fig, "fig5_filters.png")
    NUMBERS["filters"] = table


def segmentation_leaf_mask(image):
    return pipeline.analyze_array(image).leaf_mask


def derivative_filters():
    image = load_image(DEMO)
    prepared = preprocess(image)
    grey = cv2.cvtColor(prepared.enhanced, cv2.COLOR_BGR2GRAY)
    grey_f = grey.astype(np.float32)
    sx = cv2.Sobel(grey_f, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(grey_f, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sx, sy)
    lap = cv2.Laplacian(cv2.GaussianBlur(grey, (5, 5), 0), cv2.CV_32F, ksize=3)
    scharr = cv2.magnitude(cv2.Scharr(grey_f, cv2.CV_32F, 1, 0), cv2.Scharr(grey_f, cv2.CV_32F, 0, 1))

    def scale(a):
        return np.clip(255 * a / np.percentile(a, 99), 0, 255).astype(np.uint8)

    strip([("grey (after CLAHE)", grey, "gray"), ("Sobel d/dx", scale(np.abs(sx)), "gray"),
           ("Sobel d/dy", scale(np.abs(sy)), "gray"), ("Sobel magnitude", scale(magnitude), "gray"),
           ("Scharr magnitude", scale(scharr), "gray"), ("Laplacian of Gaussian", scale(np.abs(lap)), "gray")],
          "fig5_derivatives.png", width=7.0)
    # custom kernel with filter2D: unsharp masking and a hand-written Laplacian sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    sharpened = cv2.filter2D(prepared.denoised, -1, kernel)
    blurred = cv2.GaussianBlur(prepared.denoised, (0, 0), 2.0)
    unsharp = cv2.addWeighted(prepared.denoised, 1.8, blurred, -0.8, 0)
    ys, xs = np.where(pipeline.analyze_array(image).leaf_mask > 0)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    box = (slice(cy - 40, cy + 40), slice(cx - 40, cx + 40))
    strip([("blurred (pipeline input)", rgb(prepared.denoised[box]), None),
           ("filter2D, 3x3 sharpen kernel", rgb(sharpened[box]), None),
           ("unsharp mask (sigma 2, 1.8/-0.8)", rgb(unsharp[box]), None)],
          "fig5_sharpen.png", width=4.8)
    NUMBERS["gradient"] = {"sobel_mean": float(magnitude.mean()), "laplacian_std": float(lap.std())}


def canny_sweep():
    image = load_image(DEMO)
    prepared = preprocess(image)
    analysis = pipeline.analyze_array(image)
    core = segmentation_core(analysis.leaf_mask)
    grey = cv2.cvtColor(prepared.enhanced, cv2.COLOR_BGR2GRAY)
    median = float(np.median(grey[core > 0]))
    panels, rows = [], []
    configurations = []
    for sigma in (0.1, 0.33, 0.66):
        configurations.append((f"auto, sigma {sigma}", int((1 - sigma) * median), int(min(255, (1 + sigma) * median))))
    for low, high in ((40, 80), (100, 200), (160, 255)):
        configurations.append((f"fixed {low}/{high}", low, high))
    for label, low, high in configurations:
        edges = cv2.Canny(grey, low, high)
        density = float((edges[core > 0] > 0).sum() / (core > 0).sum())
        panels.append((f"{label}\nthresholds {low}/{high}, density {density:.2f}" if label.startswith("auto") else f"{label}\nedge density {density:.2f}", edges, "gray"))
        rows.append({"setting": label, "low": low, "high": high, "density": round(density, 3)})
    strip(panels, "fig5_canny_sweep.png", width=5.6, columns=3, wspace=0.08)
    NUMBERS["canny_sweep"] = rows


def segmentation_core(mask):
    return segmentation.leaf_core_mask(mask)


def threshold_comparison():
    image = load_image(DEMO)
    prepared = preprocess(image)
    analysis = pipeline.analyze_array(image)
    grey = cv2.cvtColor(prepared.denoised, cv2.COLOR_BGR2GRAY)
    reference = analysis.leaf_mask > 0
    fixed = (grey > 127).astype(np.uint8) * 255
    otsu_value, otsu = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_mean = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 4)
    adaptive_gauss = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 4)

    def orient(mask):
        mask = mask > 0
        if mask.mean() > 0.5:                      # threshold polarity is arbitrary: show the better side
            mask = ~mask
        a = (mask & reference).sum() / max((mask | reference).sum(), 1)
        b = ((~mask) & reference).sum() / max(((~mask) | reference).sum(), 1)
        mask = mask if a >= b else ~mask
        return mask.astype(np.uint8) * 255, float(max(a, b))

    candidates = [("fixed 127 on grey", fixed), (f"Otsu on grey ({otsu_value:.0f})", otsu),
                  ("adaptive mean, 31x31", adaptive_mean), ("adaptive Gaussian, 31x31", adaptive_gauss),
                  ("used: chroma distance,\nOtsu, hysteresis", analysis.leaf_mask)]
    panels, rows = [], []
    for label, mask in candidates:
        shown, score = orient(mask)
        title = label if "used" in label else f"{label}\noverlap with used mask {score:.2f}"
        panels.append((title, shown, "gray"))
        rows.append({"method": label.replace("\n", " "), "agreement": round(score, 3)})
    strip(panels, "fig5_threshold_compare.png", width=5.8, columns=3, wspace=0.08)
    NUMBERS["threshold_compare"] = rows


def equalisation():
    image = load_image(DEMO)
    denoised = cv2.GaussianBlur(image, (5, 5), 0)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    inside = pipeline.analyze_array(image).leaf_mask > 0
    panels, rows = [], []

    def record(label, new_l):
        values = new_l[inside].astype(np.float64)
        rows.append({"method": label, "std": round(float(values.std()), 1),
                     "clipped": round(float(((new_l[inside] <= 2) | (new_l[inside] >= 253)).mean() * 100), 1)})
        merged = cv2.cvtColor(cv2.merge([new_l, lab[:, :, 1], lab[:, :, 2]]), cv2.COLOR_LAB2BGR)
        panels.append((f"{label}\nspread {rows[-1]['std']}", rgb(merged), None))

    record("no equalisation", lightness)
    record("equalizeHist (global)", cv2.equalizeHist(lightness))
    for clip in (1.0, 2.0, 4.0, 8.0):
        record(f"CLAHE clip {clip:g}" + (" (used)" if clip == 2.0 else ""),
               cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lightness))
    strip(panels, "fig5_equalise.png", width=7.0)
    NUMBERS["equalise"] = rows


def morphology_operations():
    image = load_image(DEMO)
    analysis = pipeline.analyze_array(image)
    raw = analysis.raw_lesion_mask
    operations = [("raw mask", raw),
                  ("erode 3x3", cv2.erode(raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))),
                  ("dilate 3x3", cv2.dilate(raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))),
                  ("gradient (outline)", cv2.morphologyEx(raw, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))),
                  ("top-hat 5x5", cv2.morphologyEx(raw, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))),
                  ("black-hat 5x5", cv2.morphologyEx(raw, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))))]
    strip([(label, mask, "gray") for label, mask in operations], "fig5_morph_ops.png", width=5.8, columns=3, wspace=0.08)
    rows = []
    shapes = {"rectangle": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}
    for shape_name, shape in shapes.items():
        for size in (3, 5, 7):
            element = cv2.getStructuringElement(shape, (size, size))
            opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, element)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, element)
            cleaned = mask_utils.remove_small_blobs(closed, config.MIN_LESION_AREA_PIXELS)
            count = cv2.connectedComponents(cleaned)[0] - 1
            rows.append({"shape": shape_name, "size": size, "pixels": int((cleaned > 0).sum()), "blobs": int(count)})
    NUMBERS["morph_table"] = rows
    NUMBERS["raw_pixels"] = int((raw > 0).sum())


def resize_comparison():
    image = load_image(DEMO)
    small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    methods = [("nearest", cv2.INTER_NEAREST), ("bilinear", cv2.INTER_LINEAR), ("bicubic", cv2.INTER_CUBIC),
               ("Lanczos", cv2.INTER_LANCZOS4)]
    panels = []
    for name, method in methods:
        up = cv2.resize(small, (256, 256), interpolation=method)
        panels.append((f"{name} to 256", rgb(up[60:160, 60:160]), None))
    rows = []
    strip(panels, "fig5_resize.png", width=5.6)
    NUMBERS["resize"] = rows


ROOT = config.PROJECT_ROOT

if __name__ == "__main__":
    dataset_samples()
    class_gallery()
    severity_and_confidence()
    feature_view()
    filter_comparison()
    derivative_filters()
    canny_sweep()
    threshold_comparison()
    equalisation()
    morphology_operations()
    resize_comparison()
    (OUT / "extra_numbers.json").write_text(json.dumps(NUMBERS, indent=1))
    print("numbers written")
