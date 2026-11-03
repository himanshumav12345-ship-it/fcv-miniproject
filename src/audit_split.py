"""
audit_split.py - check that the grouped split really reduces near-duplicate leakage.

Idea: if a test image has a near-identical twin in the training set, a classifier
can "remember" it and the test accuracy looks better than it should.
For every test image we measure how far its NEAREST training image (same class)
is, using a simple colour-histogram distance. Then we repeat the measurement for a
plain random image-level split of the same sizes.

  grouped split  -> nearest training image should be FARTHER (fewer twins)
  random split   -> nearest training image is closer (twins leak across)

Run:  python -m src.audit_split
"""
import cv2
import numpy as np
import pandas as pd

from src import config


def colour_histogram(image_path):
    """12x6x6-bin HSV histogram, normalised so the bins sum to 1."""
    image = cv2.imread(str(config.PROJECT_ROOT / image_path))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 6], [0, 180, 0, 256, 0, 256])
    return (hist / hist.sum()).flatten()


def nearest_train_distance(test_hists, train_hists):
    """Hellinger distance from each test histogram to its closest train histogram.

    With sqrt-histograms, the dot product is the Bhattacharyya coefficient BC,
    and sqrt(1 - BC) is the Hellinger distance (0 = identical colour layout).
    """
    sqrt_test, sqrt_train = np.sqrt(test_hists), np.sqrt(train_hists)
    similarity = np.clip(sqrt_test @ sqrt_train.T, 0.0, 1.0)
    return np.sqrt(1.0 - similarity).min(axis=1)


def audit():
    table = pd.read_csv(config.SPLITS_CSV)
    hists = np.stack([colour_histogram(p) for p in table["path"]])
    rng = np.random.default_rng(config.RANDOM_SEED)

    grouped, randomised = [], []
    for label, rows in table.groupby("label"):
        index = rows.index.to_numpy()
        is_test = (rows["split"] == "test").to_numpy()
        is_train = (rows["split"] == "train").to_numpy()

        grouped.append(nearest_train_distance(hists[index[is_test]], hists[index[is_train]]))

        # random split with the same number of train and test images
        shuffled = rng.permutation(index)
        n_test, n_train = is_test.sum(), is_train.sum()
        randomised.append(nearest_train_distance(hists[shuffled[:n_test]],
                                                 hists[shuffled[n_test:n_test + n_train]]))

    grouped, randomised = np.concatenate(grouped), np.concatenate(randomised)
    print("Distance from each test image to its nearest same-class train image")
    print("(smaller = a near-duplicate twin exists in train)")
    print(f"  grouped split : median {np.median(grouped):.3f}   "
          f"share below 0.15: {np.mean(grouped < 0.15):.1%}")
    print(f"  random split  : median {np.median(randomised):.3f}   "
          f"share below 0.15: {np.mean(randomised < 0.15):.1%}")


if __name__ == "__main__":
    audit()
