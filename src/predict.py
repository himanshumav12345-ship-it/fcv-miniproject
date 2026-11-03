"""
predict.py - the ONE function the GUI (and the command line) call to analyse a leaf photo.

    result = predict_file("leaf.jpg")
    result.disease, result.confidence, result.severity_percent, result.severity_level
    result.analysis.<any intermediate image>      (for showing the processing stages)

It never crashes on bad input. Instead it raises one of two clear errors that the GUI
turns into a message box:
    InvalidImageError - not an image, unreadable, or too small
    NoLeafFoundError  - no leaf-sized object could be separated from the background

Command line:  python -m src.predict path/to/leaf.jpg [--save-stages]
"""
import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from src import config, pipeline, visualize
from src.errors import InvalidImageError, NoLeafFoundError
from src.preprocess import load_image


@dataclass
class Prediction:
    disease: str
    confidence: float                # probability of the predicted class, 0..1
    probabilities: dict              # {class name: probability}
    severity_percent: float
    severity_level: str
    analysis: pipeline.Analysis      # all intermediate images and measurements
    warnings: list                   # plain-language cautions for the user


@lru_cache(maxsize=1)
def load_model():
    """Load the trained classifier once; later calls reuse it (the GUI loads it at start-up)."""
    model_path = config.MODELS_DIR / "classifier.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path}. Run: python -m src.train")
    return joblib.load(model_path)


def make_warnings(disease, confidence, severity_percent):
    """Cautions that keep the user from over-trusting a single number."""
    notes = []
    if confidence < 0.60:
        notes.append("Low confidence: the leaf resembles several classes. Treat the label as a guess.")
    if disease == "Healthy" and severity_percent > config.SEVERITY_LEVELS[0][0]:
        notes.append("Classified healthy, but the lesion mask covers "
                     f"{severity_percent:.1f}% of the leaf (shadows, veins or natural yellowing "
                     "can cause this).")
    if disease != "Healthy" and severity_percent <= config.SEVERITY_LEVELS[0][0]:
        notes.append("A disease was predicted but very little lesion area was measured; faint "
                     "symptoms are often missed by colour thresholding, so severity may be too low.")
    return notes


def predict_array(bgr_image):
    """Analyse an already-loaded BGR image."""
    saved = load_model()
    analysis = pipeline.analyze_array(bgr_image, saved["lesion_method"])
    feature_values = pipeline.feature_dict(analysis)
    vector = np.array([[feature_values[name] for name in saved["feature_names"]]])

    probabilities = saved["model"].predict_proba(vector)[0]
    best = int(np.argmax(probabilities))
    disease = saved["class_names"][best]
    return Prediction(
        disease=disease, confidence=float(probabilities[best]),
        probabilities={name: float(p) for name, p in zip(saved["class_names"], probabilities)},
        severity_percent=analysis.severity_percent, severity_level=analysis.severity_level,
        analysis=analysis,
        warnings=make_warnings(disease, float(probabilities[best]), analysis.severity_percent))


def predict_file(path):
    """Load an image file and analyse it. Raises InvalidImageError / NoLeafFoundError."""
    return predict_array(load_image(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("--save-stages", action="store_true",
                        help="also save the stage-by-stage pictures to results/stages/")
    args = parser.parse_args()
    try:
        result = predict_file(args.image)
    except (InvalidImageError, NoLeafFoundError) as error:
        raise SystemExit(f"Cannot analyse this image: {error}")

    print(f"Disease    : {result.disease}  ({result.confidence:.0%} confidence)")
    ranked = sorted(result.probabilities.items(), key=lambda item: -item[1])[:3]
    print("Top 3      : " + ", ".join(f"{name} {p:.0%}" for name, p in ranked))
    print(f"Severity   : {result.severity_percent:.1f}% of the leaf ({result.severity_level}), "
          f"{len(result.analysis.lesions)} lesion regions")
    for note in result.warnings:
        print(f"Note       : {note}")
    if args.save_stages:
        folder = visualize.save_stage_images(result.analysis, config.RESULTS_DIR / "stages", args.image.stem)
        print(f"Stage pictures saved to {folder}")


if __name__ == "__main__":
    main()
