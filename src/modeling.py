"""
modeling.py - shared helpers for loading features and building classifier pipelines.

KEY IDEA - why scaling and feature selection live INSIDE the Pipeline:
    A scikit-learn Pipeline calls fit() on its steps using ONLY the rows it is fitted
    on. So StandardScaler learns its mean/std, and SelectKBest learns which features
    to keep, from the training rows alone. Nothing from validation or test data ever
    influences them. (Fitting a scaler on the whole dataset first would leak
    information about the test images into training.)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config
from src.build_features import features_csv_path
from src.features import feature_group


def load_feature_table(method=config.DEFAULT_LESION_METHOD):
    """Read the cached feature CSV. Returns (table of ok rows, list of feature names)."""
    table = pd.read_csv(features_csv_path(method))
    table = table[table["status"] == "ok"].reset_index(drop=True)
    names = [c for c in table.columns if c.split("_")[0] in ("colour", "texture", "shape")]
    return table, names


def select_columns(feature_names, groups):
    """Keep only the feature columns whose family (colour/texture/shape) is in `groups`."""
    return [name for name in feature_names if feature_group(name) in groups]


def split_xy(table, columns, split_names):
    """Features, integer labels and group ids of the rows belonging to the given splits."""
    rows = table[table["split"].isin(split_names)]
    return rows[columns].to_numpy(), rows["class_id"].to_numpy(), rows["group"].to_numpy()


def make_pipeline(classifier_name, **classifier_options):
    """scaler -> keep the k best features -> classifier.  k is set later (tuned in CV)."""
    if classifier_name == "svm":
        # no probability=True here: it is slow and deprecated. Confidence scores are added
        # at the very end by calibrate_for_confidence() in train.py.
        classifier = SVC(kernel="rbf", class_weight="balanced", random_state=config.RANDOM_SEED)
    elif classifier_name == "random_forest":
        classifier = RandomForestClassifier(class_weight="balanced", n_jobs=2,
                                            random_state=config.RANDOM_SEED)
    elif classifier_name == "knn":
        classifier = KNeighborsClassifier()
    else:
        raise ValueError(f"unknown classifier '{classifier_name}'")
    pipe = Pipeline([
        ("scale", StandardScaler()),
        # f_classif scores each feature by how well it separates the classes
        # (ANOVA F-value); only the k highest-scoring features are kept.
        ("select", SelectKBest(f_classif, k="all")),
        ("classify", classifier),
    ])
    pipe.set_params(**classifier_options)
    return pipe


# Hyper-parameters tried for each classifier. Names follow scikit-learn's
# "<step>__<parameter>" convention. select__k = how many features to keep.
PARAMETER_GRIDS = {
    "svm": {"select__k": [30, 60, "all"],
            "classify__C": [1, 10, 100],
            "classify__gamma": ["scale", 0.01, 0.001]},
    "random_forest": {"select__k": [30, 60, "all"],
                      "classify__n_estimators": [200, 400],
                      "classify__max_depth": [None, 20]},
    "knn": {"select__k": [30, 60, "all"],
            "classify__n_neighbors": [3, 5, 7, 11]},
}
