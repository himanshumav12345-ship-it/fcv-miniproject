# Plant leaf disease detection with classical computer vision

Detects which of six tomato-leaf conditions a photo shows (healthy, bacterial spot, early blight,
late blight, leaf mold, Septoria leaf spot), outlines the diseased regions, and estimates how much
of the leaf they cover. **No deep learning**: OpenCV image processing, hand-crafted features and a
scikit-learn SVM. Dataset: PlantVillage (tomato subset, 4,800 images).

> **Status:** the image pipeline, training, evaluation, the `predict` command, the GUI and the Word report are done
> and tested. `CODE_FLOW.md` explains how the files fit together (read it before the viva).

---

## 1. Setup (Windows first; the Mac/Linux lines are in brackets)

The 4,800 chosen PlantVillage images (`data/raw`), the split (`data/splits/splits.csv`), the cached features
(`data/features_*.csv`), the trained model (`models/classifier.joblib`) and all results are already in this
folder, so nothing needs downloading. Developed and tested on **Python 3.13**. Open a terminal **in the project
folder** (the one that contains `src`) and run:

```
python -m venv .venv
.venv\Scripts\activate             [Mac/Linux: python3 -m venv .venv  then  source .venv/bin/activate]
pip install -r requirements.txt
```

PowerShell may refuse to run the activate script ("running scripts is disabled"). Use `cmd` instead, or run
`Set-ExecutionPolicy -Scope Process Bypass` first. Always start the project's programs as `python -m src.<name>`
from this folder, never as `python src\<name>.py` (that breaks the imports).

Only if you deleted `data/raw`: `bash scripts/fetch_dataset.sh ../PlantVillage-source` rebuilds it. It needs
Git Bash (or WSL) and `git`, downloads about 250 MB, and then runs `python -m src.dataset --source <download>/raw/color`.

## 2. Run

**Just try it (needs nothing else):**

```
python -m src.gui                        # opens http://127.0.0.1:5000 ; choose or drop a tomato-leaf photo
python -m src.predict path\to\leaf.jpg --save-stages   # the same analysis in the terminal, plus stage pictures
python -m unittest discover -s tests -t . -v          # 24 unit tests, nothing extra to install
```

**Re-create everything from the images** (about 6 minutes on 2 CPU cores):

```
python -m src.build_features             # 2 min: runs the pipeline on all 4,800 images -> data/features_own_green.csv
python -m src.train                      # 2.5 min: tune 3 classifiers, choose on validation, save models/classifier.joblib
python -m src.evaluate --ablations       # seconds: which features matter, leaky vs grouped CV (uses train+val only)
python -m src.evaluate --final           # scores the TEST split. Refuses to run twice (see section 5)
python -m src.visualize --sample 6 --split val        # pictures in results/stages/<image>/montage.png
```

Optional checks that produced the numbers in section 4:

```
# the "which lesion method feeds the classifier" table in the ablations needs the other two feature tables first
python -m src.build_features --method hue_rule     # 2 min each (the tables are already included)
python -m src.build_features --method kmeans
python -m src.evaluate --ablations                 # now includes the lesion-method comparison

python -m src.audit_split                                                      # does the grouped split leak less?
python -m src.evaluate_leaf_mask --reference ..\PlantVillage-source\raw\segmented   # needs the fetch script's download
python -m src.evaluate_lesions --role dev        # then --role holdout (final lesion numbers)
python -m src.evaluate_false_alarms --per-class 100
```

**The report** (Word, in `report/`): `pip install -r requirements-report.txt`, then

```
python -m report.make_figures            # the Chapter 4-5 pictures, from the real pipeline and results
python -m report.make_figures_extra      # extra comparisons and the numbers quoted in the text
python -m report.build_report --toc-pages report/toc_pages.json     # writes report/FCV_Mini_Project_Report.docx
```

The contents-page numbers come from `report/toc_pages.json` and are typed text; after editing the report, check
them by hand. `report/lab_template.docx` is the department template converted from `.doc`.

**If something goes wrong**

- *"Can't get attribute" / a scikit-learn pickle error when loading the model:* your scikit-learn version differs
  from the one that saved `models/classifier.joblib` (`requirements.txt` pins 1.9.1). Either install the pinned
  versions, or just run `python -m src.train` once, which writes a model that matches your installation.
- *`pip install` cannot find a pinned version:* your Python is older than 3.11. Install Python 3.13, or change the
  `==` in `requirements.txt` to `>=` for the failing line.
- *`evaluate --final` says it already ran:* that is on purpose (the test split is scored once). After retraining
  you may pass `--again`, and then say so in the report.
- *The port is busy:* `python -m src.gui --port 5001`.

## 3. How the pipeline works (the part to explain in the viva)

One function, `pipeline.analyze_array()`, chains every stage. Training, evaluation and the GUI all use
it, so they cannot disagree. Every threshold lives in `src/config.py` with a comment on why.

| # | Stage (file) | Input | Output | What it does, in plain words |
|---|---|---|---|---|
| 1 | Preprocess (`preprocess.py`) | photo | denoised + contrast-enhanced copies | **Gaussian blur** averages each pixel with its neighbours so sensor noise does not create fake edges and spots. **CLAHE** boosts local contrast on the *lightness channel only* (in Lab space), so dark lesions stand out without changing the leaf's real colours. |
| 2 | Leaf mask (`segmentation.extract_leaf_mask`) | denoised image | black/white mask of the leaf | Measure the background colour from the photo border; mark every pixel whose **colour** (Lab a, b, ignoring brightness) is far from it, using **Otsu's threshold**. **Hysteresis** also keeps weaker pixels *connected* to sure leaf (recovers pale or dead edge tissue). Keep the biggest blob, fill holes. Ignoring brightness is what removes cast shadows. |
| 3 | Lesion mask (`segmentation.lesion_mask_own_green`) | image + leaf mask | mask of diseased pixels | Find *this leaf's own healthy colour* (median colour of its greenest 25% of pixels). A pixel is a lesion if its colour is clearly different from that. Only colour is compared, because veins, folds and shade change brightness but not colour. |
| 4 | Clean-up (`refine.clean_lesion_mask`) | raw lesion mask | cleaned mask | **Opening** (erode then dilate) removes specks; **closing** (dilate then erode) joins broken pieces; blobs under 8 px are dropped. |
| 5 | Edges and contours (`refine`) | cleaned mask, enhanced image | outlines + measurements | **Canny** finds sharp brightness changes: used as an extra picture and one feature (edge density). It does **not** edit the lesion mask. `findContours` outlines each lesion; area, perimeter, circularity, solidity, aspect ratio are measured. |
| 6 | Features (`features.py`) | image + masks | 90 numbers | 55 **colour** (mean/std/skew of H,S,V,L,a,b for leaf and lesions; hue/saturation histograms; green/yellow/brown shares), 20 **texture** (GLCM contrast, homogeneity etc.; LBP histogram), 15 **shape** (lesion count, area ratio, circularity, Hu moments, edge density). |
| 7 | Classify (`modeling.py`, `train.py`) | 90 numbers | one of 6 labels + confidence | Scale features, keep the 60 most informative, **RBF-kernel SVM**. |
| 8 | Severity (`severity.py`) | leaf mask + lesion mask | % and level | `100 x lesion pixels / leaf pixels`, in the same rim-trimmed region. Levels: <5% trace, 5-15% mild, 15-35% moderate, >35% severe (our own cut-offs, not an official scale). |

**Things you should be able to justify**

- *Why not just "non-green = diseased"?* That is method A (`lesion_mask_hue_rule`), kept as the baseline.
  It calls naturally yellow leaves diseased (65% flagged on a yellow Septoria leaf whose true lesion share
  was 13%). Comparing with the leaf's own healthy colour fixes the worst of that.
- *Why not K-means?* It always returns k clusters, so on a healthy leaf it invents "lesions": about 1/3 of
  every healthy leaf is flagged (measured), and its healthy-vs-diseased AUC is 0.26, worse than chance.
- *Why compare Lab a, b and not brightness?* Adding brightness raised false alarms on healthy leaves from
  about 3% to about 18% in a test on the dev images, because veins and folds are darker.
- *Why CLAHE only on lightness?* Equalising colour channels would shift the colours the classifier relies on.
- *Why erode the leaf mask by ~3 px before finding lesions?* The outermost pixels blend leaf and background,
  look "not green", and would be counted as lesions.
- *Circularity caveat:* on a pixel grid a perfect circle scores only about 0.67 (radius 3) to 0.89 (radius 60),
  not 1.0. Use it as a relative round-vs-elongated measure.

## 3b. The GUI (how to explain it)

Start it with `python -m src.gui`. It needs the trained model (`python -m src.train`) and `flask`
(listed in `requirements.txt`). It serves one page on `127.0.0.1` only, so it is not visible to other computers.

| Part | File | What it does |
|---|---|---|
| Page | `src/gui_page.html` | Choose or drop a photo, send it to the server, draw the six stage pictures, the disease label, the confidence bars, the severity ruler and any warnings. Plain HTML, CSS and JavaScript, no framework. |
| Server | `src/gui.py` | One route, `POST /analyse`. It decodes the upload, calls `predict.predict_array()` (the same code path as training and testing), and returns the result as JSON. |
| Errors | `src/gui.py` | `InvalidImageError`, `NoLeafFoundError`, a missing model and oversized uploads each become a short message with advice. A bug never shows a stack trace. |

What the user sees, in plain words: *input* is one photo; *output* is the label with the probability of each
likely class, the percentage of leaf area flagged as diseased on a four-band ruler, how many separate regions
were measured, and six pictures showing what each processing step did. Nothing in the GUI is computed
separately: if the page and the evaluation scripts ever disagreed, that would be a bug.

Limits to state: the GUI only shows what the pipeline produces, so it inherits the pipeline's limits
(section 5): colour-based severity, lab-style training photos, no field-photo validation.

## 4. Results (all measured; nothing here is estimated)

**Final test set, scored once: 757 images, accuracy 90.8%, macro-F1 0.909** (95% interval about +/-2 points).

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Healthy | 0.962 | 0.984 | 0.973 |
| Bacterial spot | 0.991 | 0.928 | 0.959 |
| Early blight | 0.790 | 0.838 | 0.813 |
| Late blight | 0.923 | 0.900 | 0.911 |
| Leaf mold | 0.900 | 0.929 | 0.914 |
| Septoria leaf spot | 0.896 | 0.868 | 0.882 |

Confusion matrix: `results/confusion_matrix_test.png`; misclassified examples: `results/test_misclassified.png`.
The common mistakes are between the spotty diseases (Leaf mold to Early blight 9, Early blight to Septoria 8,
Late blight to Early blight 8, Septoria to Early blight 8). Several of those images are ambiguous even to a person.

**Model choice** (tuned with 5-fold grouped CV on train, compared on validation):

| Classifier | CV macro-F1 (train) | Val accuracy | Val macro-F1 |
|---|---|---|---|
| SVM (RBF), C=10, 60 features | 0.891 | 0.891 | 0.891 |
| Random forest | 0.859 | 0.866 | 0.865 |
| k-NN (k=5) | 0.836 | 0.854 | 0.852 |

**Which features matter** (fit on train, scored on validation): colour only 0.835; texture only 0.704;
shape only 0.528; colour+texture 0.874; colour+shape 0.877; all three **0.891**.

**Why the grouped split matters** (same model, train+val): leaky random folds **92.7%** versus grouped folds
**88.7%**. PlantVillage photographs one leaf several times and numbers the photos consecutively, so a random
split puts near-twins in train and test. We split by blocks of 20 consecutive image numbers.
`src/audit_split.py` shows this cuts very close train/test lookalikes from about 12-15% to about 5-7%.
It does not remove them entirely.

**Leaf mask vs PlantVillage's own background-removed images** (300 train images): mean IoU **0.924**,
median 0.946. Weakest on Leaf_Mold (mean 0.87, some blue-background photos).

**Lesion segmentation vs 12 hand-annotated hold-out masks** (10 diseased + 2 healthy; run once):

| Method | Dice | Healthy-leaf false alarm | Severity error (pp) |
|---|---|---|---|
| Hue rule (baseline) | 0.406 | 1.2% | 12.1 |
| K-means | 0.296 | 39.8% | 21.6 |
| Own-green (used) | 0.407 | 0.6% | 6.6 |

Annotation-free check on 600 train images: healthy-vs-diseased AUC of the lesion percentage is 0.860 for
own-green, 0.826 for the hue rule and 0.262 for K-means.

**Read these segmentation numbers carefully.**

- Dice around 0.4 is low. Colour thresholds miss faint yellow patches, tiny spots and pale dead areas.
  Own-green **ties** the baseline on Dice; its advantage is fewer false alarms on yellow/healthy leaves
  and about half the severity error. It is not a large improvement.
- The annotation set is tiny (24 images) so every segmentation number has wide uncertainty.
- **The 24 masks were drawn by Claude (an AI), not by a plant pathologist.** They follow a written protocol
  (`src/annotations.py`), were checked visually, and are conservative, but they are approximate. For a more
  defensible report, edit `data/annotations/shapes.json` or redraw several yourself and re-run
  `python -m src.evaluate_lesions`.
- Failure examples (blue = missed lesion, red = false alarm): `results/lesion_failures_*_holdout.png`.
  Typical failures: veins along the midrib flagged as lesion (Leaf_Mold_3), a large grey dead area missed
  (Late_blight_4), faint bacterial specks missed (Bacterial_spot_3).
- Classification accuracy does **not** validate the lesion mask. Features built from the K-means
  masks classified as well as, or marginally better than, the others (validation accuracy 0.906 vs 0.891 for
  own-green, within noise at 762 images) even though K-means makes the worst masks. Judge masks and severity
  on masks, not on accuracy.

## 5. Honest limits (state these in the report)

- **Severity** = lesion area / leaf area in a flat photo. It is only as good as the two masks: it
  under-counts faint disease and over-counts if veins or shadows leak into the lesion mask; curled or folded
  leaves and the leaf underside are not seen. Level cut-offs are our proposal. It has been validated only
  against the 24 annotated masks (average error about 6.6 percentage points on the hold-out set).
- **Lab-condition data.** PlantVillage has plain backgrounds and one leaf per photo. No field photos were
  tested, and accuracy there would be lower. The classifier has six classes only and no "not a tomato leaf"
  class.
- **Leaf-likeness guard** (`segmentation.check_looks_like_leaf`) rejects noise, uniformly coloured objects and
  non-plant colours. It cannot reject any green, yellow or brown object (a skin-tone blob passed in a test).
- **Two dataset images cannot be analysed** (fully dead dark-brown Late_blight leaves whose colour matches the
  background); the pipeline reports "no leaf found" for them. Neither is in the test split.
- **Near-duplicate leakage is reduced, not eliminated**, so 90.8% is probably still somewhat optimistic.
- **Test set discipline.** Scaling and feature selection are fitted only on the training rows (inside the
  scikit-learn Pipeline); tuning uses grouped CV on train; the model is chosen on validation, refitted on
  train+val, then the test split is scored once. `evaluate --final` refuses to run again unless `--again` is
  passed. The leaf-likeness guard was added after the test run; a rebuild of the feature table afterwards was
  byte-identical, so it changed nothing for dataset images.
- Tested with OpenCV 5.0.0 only. The code uses only long-standing OpenCV functions, but 4.x was not tested.

## 6. Project layout

```
src/config.py       every path, seed and threshold, each with the reason for its value
src/dataset.py       build the 4,800-image subset and the grouped train/val/test split
src/preprocess.py    load, resize, Gaussian blur, CLAHE
src/segmentation.py  leaf mask, leaf-likeness check, three lesion-mask methods
src/mask_utils.py    morphology helpers (open, close, largest blob, small-blob removal)
src/refine.py        lesion clean-up, Canny edge map, contour measurements, overlay drawing
src/features.py      90 colour / texture / shape features
src/severity.py      lesion area / leaf area and level names
src/pipeline.py      the single chain used everywhere
src/build_features.py, train.py, modeling.py, evaluate.py     training and evaluation
src/annotations.py, evaluate_lesions.py, evaluate_leaf_mask.py, evaluate_false_alarms.py, audit_split.py
src/predict.py       analyse one image: label, confidence, severity, warnings
src/gui.py, src/gui_page.html   the GUI: Flask server (gui.py) + one HTML page with its own CSS and JavaScript
src/visualize.py     save a picture of every stage
data/                raw/ (images), splits/splits.csv, annotations/shapes.json, features_*.csv
models/classifier.joblib      results/   tables, figures, failure examples
tests/               24 unit tests: pipeline on synthetic images with known answers, and the GUI server
report/              the Word report: make_figures*.py (pictures), build_report.py + docx_helpers.py (document),
                     lab_template.docx (department template), figures/ (every picture used), the finished .docx/.pdf
CODE_FLOW.md         how one photo and the whole project flow through the files
requirements.txt, requirements-report.txt     packages for the project / for rebuilding the report
```

`predict.py` raises `InvalidImageError` (unreadable file, not an image, smaller than 64 px) or
`NoLeafFoundError` (no leaf-like object); the GUI shows these as a short message instead of a result.

## 7. Differences from the original plan document

- The split is grouped by blocks of consecutive image ids (the plan said stratified random).
- Lesion segmentation: the plan's HSV rule and K-means were both built and measured; the default is a third
  method ("own-green") added after the real images showed the failures described above.
- Feature selection is `SelectKBest` (ANOVA F-score) inside the pipeline; confidence comes from a calibrated
  SVM (`CalibratedClassifierCV` with grouped folds).
- A leaf-likeness check and a split-leakage audit were added.
- The GUI is a local web page (Flask) instead of a Tkinter window, because the development machine had no
  Tkinter and a web page could be run and checked end to end there. The user-facing job is the same.
- Not done yet: any test on real field photos.
