"""
train.py - train the disease classifier with a strict train / validation / test routine.

  1. TUNE   : for each classifier (SVM, Random Forest, k-NN) pick hyper-parameters with
              5-fold GROUPED cross-validation on the TRAIN split only. "Grouped" means
              images of the same leaf block are never split across folds.
  2. CHOOSE : score the three tuned classifiers on the VALIDATION split; the best macro-F1
              wins. Validation was never used for tuning, so this comparison is fair.
  3. REFIT  : refit the winner (scaler, feature selection and classifier all again) on
              TRAIN + VALIDATION.
  4. TEST   : NOT touched here. evaluate.py scores the test split exactly once.

Run:  python -m src.train
Writes: models/classifier.joblib, results/model_selection.csv, results/best_params.json
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold

from src import config, modeling


def tune_classifier(name, x_train, y_train, groups_train):
    """Grouped 5-fold grid search on the training data. Returns the fitted GridSearchCV."""
    search = GridSearchCV(
        estimator=modeling.make_pipeline(name),
        param_grid=modeling.PARAMETER_GRIDS[name],
        scoring="f1_macro",
        cv=GroupKFold(n_splits=5),
        n_jobs=2,
        refit=True,
    )
    search.fit(x_train, y_train, groups=groups_train)
    return search


def calibrate_for_confidence(pipeline, x, y, groups):
    """Wrap the whole pipeline so predict_proba gives trustworthy percentages.

    A raw SVM only outputs distances to its decision boundary. CalibratedClassifierCV
    learns, from out-of-fold predictions, how to turn them into probabilities. The folds
    are GROUPED (same rule as tuning) and wrap the WHOLE pipeline, so even the scaler and
    feature selection are re-fitted without the held-out fold.
    """
    grouped_folds = list(GroupKFold(n_splits=5).split(x, y, groups))
    return CalibratedClassifierCV(pipeline, method="sigmoid", ensemble=False, cv=grouped_folds)


def main():
    table, feature_names = modeling.load_feature_table()
    x_train, y_train, g_train = modeling.split_xy(table, feature_names, ["train"])
    x_val, y_val, _ = modeling.split_xy(table, feature_names, ["val"])
    print(f"train {len(y_train)} images, val {len(y_val)} images, {len(feature_names)} features")

    rows, searches = [], {}
    for name in modeling.PARAMETER_GRIDS:
        started = time.time()
        search = tune_classifier(name, x_train, y_train, g_train)
        searches[name] = search
        predictions = search.predict(x_val)
        rows.append({"classifier": name,
                     "cv_macro_f1_train": search.best_score_,
                     "val_accuracy": accuracy_score(y_val, predictions),
                     "val_macro_f1": f1_score(y_val, predictions, average="macro"),
                     "best_params": json.dumps({k: v for k, v in search.best_params_.items()},
                                               default=str),
                     "seconds": round(time.time() - started)})
        print(f"{name:14s} CV F1 {rows[-1]['cv_macro_f1_train']:.3f} | "
              f"val acc {rows[-1]['val_accuracy']:.3f} F1 {rows[-1]['val_macro_f1']:.3f} "
              f"| {rows[-1]['best_params']}")

    summary = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(config.RESULTS_DIR / "model_selection.csv", index=False)

    best_name = summary.sort_values("val_macro_f1", ascending=False).iloc[0]["classifier"]
    best_params = searches[best_name].best_params_
    print(f"\nchosen on validation macro-F1: {best_name}")

    # refit on train + validation: scaler, feature selection and classifier all start fresh
    x_all, y_all, groups_all = modeling.split_xy(table, feature_names, ["train", "val"])
    plain_pipeline = modeling.make_pipeline(best_name, **best_params)
    final_model = calibrate_for_confidence(plain_pipeline, x_all, y_all, groups_all)
    final_model.fit(x_all, y_all)

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": final_model, "feature_names": feature_names,
                 "class_names": config.CLASS_NAMES, "classifier": best_name,
                 "params": best_params, "lesion_method": config.DEFAULT_LESION_METHOD},
                config.MODELS_DIR / "classifier.joblib")
    (config.RESULTS_DIR / "best_params.json").write_text(
        json.dumps({"classifier": best_name, "params": best_params}, indent=1, default=str))
    print(f"saved model -> {config.MODELS_DIR / 'classifier.joblib'}")


if __name__ == "__main__":
    main()
