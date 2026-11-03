"""
build_features.py - run the shared pipeline on every dataset image and save the features.

Output: data/features_<method>.csv with one row per image:
    path, label, class_id, group, split, status, <about 90 feature columns>
`status` is "ok", or the reason the image could not be analysed (e.g. no leaf found).
Failed images keep a row (so nothing silently disappears) but have empty features.

Run:  python -m src.build_features                       (default lesion method)
      python -m src.build_features --method hue_rule     (for the ablation)
"""
import argparse
import time

import pandas as pd
from joblib import Parallel, delayed

from src import config, pipeline
from src.errors import InvalidImageError, NoLeafFoundError


def features_csv_path(method):
    return config.DATA_DIR / f"features_{method}.csv"


def features_for_one_image(relative_path, method):
    """Return ({feature: value}, status) for one image."""
    try:
        analysis = pipeline.analyze_file(config.PROJECT_ROOT / relative_path, method)
    except (NoLeafFoundError, InvalidImageError) as error:
        return {}, f"failed: {type(error).__name__}"
    return pipeline.feature_dict(analysis), "ok"


def build(method, n_jobs):
    table = pd.read_csv(config.SPLITS_CSV)
    started = time.time()
    results = Parallel(n_jobs=n_jobs)(
        delayed(features_for_one_image)(path, method) for path in table["path"])
    print(f"processed {len(table)} images in {time.time() - started:.0f} s")

    feature_rows = pd.DataFrame([features for features, _ in results])
    table["status"] = [status for _, status in results]
    combined = pd.concat([table, feature_rows], axis=1)

    config.DATA_DIR.mkdir(exist_ok=True)
    combined.to_csv(features_csv_path(method), index=False)

    failed = combined[combined["status"] != "ok"]
    print(f"ok: {len(combined) - len(failed)}   failed: {len(failed)}")
    if len(failed):
        print(failed.groupby(["split", "label"]).size().to_string())
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", default=config.DEFAULT_LESION_METHOD,
                        choices=["hue_rule", "kmeans", "own_green"])
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    build(args.method, args.jobs)
