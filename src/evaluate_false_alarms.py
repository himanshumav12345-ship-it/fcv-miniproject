"""
evaluate_false_alarms.py - an annotation-free check of the lesion methods on many images.

Idea: a healthy leaf should have (almost) no lesion area. So for every method we measure
the lesion % on many healthy leaves (false alarms) and on leaves of each disease class,
then ask: how well does the lesion % alone separate healthy from diseased leaves?
That separation is the ROC AUC: 0.5 = no better than chance, 1.0 = perfect.

Limits: PlantVillage labels are per image, so a "diseased" leaf can have little visible
lesion and a "healthy" one can have shading. This is a coarse but large-sample check;
it complements, not replaces, the annotated-mask evaluation.

Only the TRAIN split is used, so the test set stays untouched.

Run:  python -m src.evaluate_false_alarms --per-class 100
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config, pipeline
from src.errors import NoLeafFoundError
from src.segmentation import LESION_METHODS


def measure(per_class):
    table = pd.read_csv(config.SPLITS_CSV)
    table = table[table["split"] == "train"].groupby("label").sample(
        per_class, random_state=config.RANDOM_SEED)
    records = []
    for _, row in table.iterrows():
        for method in LESION_METHODS:
            try:
                analysis = pipeline.analyze_file(config.PROJECT_ROOT / row["path"], method)
            except NoLeafFoundError:
                continue
            records.append({"label": row["label"], "method": method,
                            "lesion_percent": analysis.severity_percent})
    return pd.DataFrame(records)


def report(result):
    result["diseased"] = result["label"] != "Healthy"
    pd.set_option("display.width", 200)

    print("Median lesion % per class (healthy should be ~0):")
    print(result.pivot_table(index="label", columns="method", values="lesion_percent",
                             aggfunc="median").reindex(config.CLASS_NAMES).round(1).to_string())

    healthy = result[~result["diseased"]]
    print("\nHealthy leaves - how much lesion area is (wrongly) flagged:")
    print(healthy.groupby("method")["lesion_percent"].describe(percentiles=[0.5, 0.9])
          [["mean", "50%", "90%", "max"]].round(1).to_string())

    print("\nHealthy-vs-diseased separation using lesion % alone (ROC AUC):")
    for method, group in result.groupby("method"):
        print(f"  {method:10s} AUC = {roc_auc_score(group['diseased'], group['lesion_percent']):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=100)
    args = parser.parse_args()
    result = measure(args.per_class)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    result.to_csv(config.RESULTS_DIR / "false_alarm_check_train.csv", index=False)
    report(result)
