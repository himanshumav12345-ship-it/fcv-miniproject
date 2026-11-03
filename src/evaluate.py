"""
evaluate.py - measure the classifier honestly.

Two kinds of experiment, kept strictly apart:

  EXPERIMENTS (--ablations)   use TRAIN to fit and VALIDATION to score.
      * which feature families matter?        colour / texture / shape and combinations
      * does the lesion method matter?        features from hue_rule, kmeans, own_green
      * how much does a leaky split inflate accuracy?   (random vs grouped folds)

  FINAL TEST (--final)        scores the saved model on the TEST split, exactly once.
      The test split is never used to choose anything. To make "once" real, the script
      refuses to run again unless you pass --again (and says so in the report).

Run:  python -m src.evaluate --ablations
      python -m src.evaluate --final
"""
import argparse
import json
from collections import Counter

import cv2
import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

matplotlib.use("Agg")                       # draw to files, no window needed
import matplotlib.pyplot as plt             # noqa: E402

from src import config, modeling            # noqa: E402

TEST_RESULTS_JSON = config.RESULTS_DIR / "test_results.json"


# -----------------------------------------------------------------------------
# Experiments on TRAIN + VALIDATION only
# -----------------------------------------------------------------------------
def best_svm_settings():
    """The tuned SVM settings found by train.py (read from results/best_params.json)."""
    saved = json.loads((config.RESULTS_DIR / "best_params.json").read_text())
    return saved["classifier"], saved["params"]


def score_on_validation(table, columns, classifier_name, params):
    """Fit on the train split, score on the validation split. Returns (accuracy, macro-F1)."""
    x_train, y_train, _ = modeling.split_xy(table, columns, ["train"])
    x_val, y_val, _ = modeling.split_xy(table, columns, ["val"])
    # select__k may be larger than the number of columns in a small feature subset
    params = dict(params)
    if isinstance(params.get("select__k"), int):
        params["select__k"] = min(params["select__k"], len(columns))
    model = modeling.make_pipeline(classifier_name, **params).fit(x_train, y_train)
    predictions = model.predict(x_val)
    return accuracy_score(y_val, predictions), f1_score(y_val, predictions, average="macro")


def feature_family_ablation(classifier_name, params):
    table, names = modeling.load_feature_table()
    rows = []
    for label, families in [("colour only", {"colour"}), ("texture only", {"texture"}),
                            ("shape only", {"shape"}),
                            ("colour + texture", {"colour", "texture"}),
                            ("colour + shape", {"colour", "shape"}),
                            ("all three", {"colour", "texture", "shape"})]:
        columns = modeling.select_columns(names, families)
        accuracy, macro_f1 = score_on_validation(table, columns, classifier_name, params)
        rows.append({"features": label, "n_features": len(columns),
                     "val_accuracy": accuracy, "val_macro_f1": macro_f1})
    return pd.DataFrame(rows)


def lesion_method_ablation(classifier_name, params):
    rows = []
    for method in ("hue_rule", "kmeans", "own_green"):
        try:
            table, names = modeling.load_feature_table(method)
        except FileNotFoundError:
            continue
        accuracy, macro_f1 = score_on_validation(table, names, classifier_name, params)
        rows.append({"lesion method": method, "val_accuracy": accuracy, "val_macro_f1": macro_f1})
    return pd.DataFrame(rows)


def leakage_demonstration(classifier_name, params):
    """Same data, same model: grouped folds versus ordinary random folds.

    Random folds put near-duplicate images (same leaf, different angle) on both sides of
    the split, so the model is partly tested on images it has effectively seen.
    Uses train + validation only; the test split is not involved.
    """
    table, names = modeling.load_feature_table()
    x, y, groups = modeling.split_xy(table, names, ["train", "val"])
    model = modeling.make_pipeline(classifier_name, **params)
    random_folds = KFold(5, shuffle=True, random_state=config.RANDOM_SEED)
    grouped_folds = GroupKFold(5)
    random_pred = cross_val_predict(model, x, y, cv=random_folds)
    grouped_pred = cross_val_predict(model, x, y, cv=grouped_folds, groups=groups)
    return pd.DataFrame([
        {"cross-validation": "random image-level folds (leaky)",
         "accuracy": accuracy_score(y, random_pred), "macro_f1": f1_score(y, random_pred, average="macro")},
        {"cross-validation": "grouped folds (used in this project)",
         "accuracy": accuracy_score(y, grouped_pred), "macro_f1": f1_score(y, grouped_pred, average="macro")}])


def run_ablations():
    classifier_name, params = best_svm_settings()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    for title, builder, filename in [
            ("Feature families (fit on train, scored on validation)", feature_family_ablation, "ablation_features.csv"),
            ("Lesion method used to compute the features", lesion_method_ablation, "ablation_lesion_method.csv"),
            ("Effect of leaky vs grouped cross-validation (train+val)", leakage_demonstration, "ablation_leakage.csv")]:
        result = builder(classifier_name, params)
        result.to_csv(config.RESULTS_DIR / filename, index=False)
        print(f"\n{title}\n{result.round(3).to_string(index=False)}")


# -----------------------------------------------------------------------------
# FINAL TEST - once
# -----------------------------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, path):
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(config.CLASS_NAMES)))
    figure, axes = plt.subplots(figsize=(7.5, 6.5))
    ConfusionMatrixDisplay(matrix, display_labels=config.CLASS_NAMES).plot(
        ax=axes, cmap="Blues", xticks_rotation=35, colorbar=False)
    axes.set_title("Test-set confusion matrix (rows = true class)")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return matrix


def save_misclassified_montage(test_rows, y_pred, path, how_many=12):
    """A picture of test images the model got wrong, labelled true -> predicted."""
    wrong = test_rows[y_pred != test_rows["class_id"].to_numpy()].copy()
    wrong["predicted"] = [config.CLASS_NAMES[i] for i in y_pred[y_pred != test_rows["class_id"].to_numpy()]]
    wrong = wrong.sample(min(how_many, len(wrong)), random_state=config.RANDOM_SEED)
    tiles = []
    for _, row in wrong.iterrows():
        image = cv2.resize(cv2.imread(str(config.PROJECT_ROOT / row["path"])), (200, 200))
        cv2.putText(image, f"true: {row['label']}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, f"pred: {row['predicted']}", (3, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append(image)
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


def run_final_test(allow_repeat):
    if TEST_RESULTS_JSON.exists() and not allow_repeat:
        raise SystemExit(f"{TEST_RESULTS_JSON.name} already exists: the test split was already "
                         "scored. Re-running would make the number optimistic if you changed "
                         "anything in between. Use --again only if the code was NOT changed.")
    saved = joblib.load(config.MODELS_DIR / "classifier.joblib")
    full_table = pd.read_csv(modeling.features_csv_path(saved["lesion_method"]))
    test_all = full_table[full_table["split"] == "test"].reset_index(drop=True)
    test_ok = test_all[test_all["status"] == "ok"].reset_index(drop=True)

    x_test = test_ok[saved["feature_names"]].to_numpy()
    y_true = test_ok["class_id"].to_numpy()
    y_pred = saved["model"].predict(x_test)

    # images the pipeline could not analyse count as ERRORS in the headline number
    n_failed = len(test_all) - len(test_ok)
    correct = int((y_pred == y_true).sum())
    results = {
        "classifier": saved["classifier"], "n_test_images": len(test_all),
        "n_pipeline_failures": n_failed,
        "accuracy_all_images": correct / len(test_all),
        "accuracy_analysed_images": correct / len(test_ok),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "per_class": classification_report(y_true, y_pred, target_names=config.CLASS_NAMES,
                                           output_dict=True, zero_division=0)}
    TEST_RESULTS_JSON.write_text(json.dumps(results, indent=1))
    matrix = save_confusion_matrix(y_true, y_pred, config.RESULTS_DIR / "confusion_matrix_test.png")
    save_misclassified_montage(test_ok, y_pred, config.RESULTS_DIR / "test_misclassified.png")

    print(f"TEST SET ({len(test_all)} images, scored once) - model: {saved['classifier']}")
    print(f"  accuracy {results['accuracy_all_images']:.3f}  macro-F1 {results['macro_f1']:.3f}  "
          f"(pipeline failures: {n_failed})")
    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=3, zero_division=0))
    confusions = Counter()
    for true_i, pred_i in zip(y_true, y_pred):
        if true_i != pred_i:
            confusions[(config.CLASS_NAMES[true_i], config.CLASS_NAMES[pred_i])] += 1
    print("most frequent mistakes (true -> predicted):")
    for (true_name, pred_name), count in confusions.most_common(6):
        print(f"  {true_name} -> {pred_name}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--again", action="store_true", help="allow re-scoring the test split")
    args = parser.parse_args()
    if args.ablations:
        run_ablations()
    if args.final:
        run_final_test(args.again)
    if not (args.ablations or args.final):
        parser.print_help()
