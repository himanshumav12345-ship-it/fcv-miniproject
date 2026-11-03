# How the code fits together

Read this top to bottom once and you can explain the whole project. Every command is run from the project
folder as `python -m src.<name>`.

## 1. The big picture

The project does **two different jobs**. Do not mix them up in the viva.

```
JOB A - TEACH THE COMPUTER (offline, run once)           JOB B - USE IT ON A PHOTO (online, every time)

 PlantVillage images (data/raw)                           one leaf photo (file or GUI upload)
        |  src/dataset.py                                         |
        v                                                         |  src/predict.py  (one function)
 data/splits/splits.csv (train / val / test)                      v
        |  src/build_features.py                         src/pipeline.py  analyze_array()
        |      calls src/pipeline.py on every image        (the SAME chain used by Job A)
        v                                                         |
 data/features_own_green.csv  (90 numbers per image)              +--> 90 features --> models/classifier.joblib
        |  src/train.py                                           |                    --> disease + confidence
        v                                                         +--> lesion mask / leaf mask
 models/classifier.joblib                                                              --> severity %
        |  src/evaluate.py  (scores the test split once)          v
        v                                               GUI page (src/gui.py + gui_page.html) or the terminal
 results/*.csv, *.png, test_results.json
```

The key design rule: **`pipeline.py` is the only place where the image-processing stages are chained.**
Training, testing and the GUI all call it, so the numbers in the report are exactly what the GUI would show.

## 2. What happens to ONE photo (the call chain)

`src/gui.py` (or `src/predict.py`) → `predict.predict_array(image)` → `pipeline.analyze_array(image)`:

| Step | Function (file) | What it does | Output |
|---|---|---|---|
| 0 | `load_image` / `decode_image` (`preprocess.py`) | read the file, refuse unreadable or tiny images (`InvalidImageError`), shrink to at most 256 px | BGR image |
| 1 | `preprocess` → `denoise`, `enhance_contrast` (`preprocess.py`) | Gaussian blur 5x5; CLAHE on the Lab lightness channel only | `denoised`, `enhanced` images |
| 2 | `extract_leaf_mask` (`segmentation.py`) | colour of the photo border = background; colour distance (Lab a,b) of every pixel to it; Otsu + hysteresis threshold; open; keep largest blob; fill holes; `check_looks_like_leaf` rejects non-leaves (`NoLeafFoundError`) | `leaf_mask` |
| 3 | `lesion_mask_own_green` (`segmentation.py`) | healthy colour of this leaf = median of its greenest 25%; pixels farther than 14 from it (colour only) are lesions. The other two methods, `lesion_mask_hue_rule` and `lesion_mask_kmeans`, are the baselines kept for comparison | `raw_lesion_mask` |
| 4 | `clean_lesion_mask` (`refine.py`, with `mask_utils.py`) | opening 3x3, closing 3x3, delete blobs under 8 px | `lesion_mask` |
| 5 | `compute_edge_map`, `find_lesions` (`refine.py`) | Canny edges inside the leaf; `findContours` on each lesion, measure area, perimeter, circularity, solidity, aspect ratio | `edges`, list of `Lesion` |
| 6 | `severity_percent`, `severity_level` (`severity.py`) | 100 x lesion pixels / leaf pixels, then Trace / Mild / Moderate / Severe | percent + level |
| 7 | `feature_dict` → `features.extract_features` (`features.py`) | 90 numbers: 55 colour, 20 texture (GLCM, LBP), 15 shape (lesion stats, Hu moments, edge density) | feature vector |
| 8 | `predict_array` (`predict.py`) | feed the vector to the saved model: StandardScaler → best 60 features → SVM, probabilities calibrated; adds plain-language warnings | label, confidence, probabilities |
| 9 | `prediction_to_json` (`gui.py`) | pack six stage pictures (as PNG text), label, bars, severity ruler into JSON | what the page draws |

All thresholds (5x5, clip 2.0, 14, 8 px, 0.33 ...) live in **`src/config.py`**, each with the reason for its value.

## 3. File by file

### Core (the system itself)

| File | Role |
|---|---|
| `src/config.py` | every path, seed, threshold and severity limit in one place |
| `src/errors.py` | the two ways an input can be unusable: `InvalidImageError`, `NoLeafFoundError` |
| `src/preprocess.py` | load / decode, size checks, resize, Gaussian blur, CLAHE |
| `src/segmentation.py` | leaf mask, leaf-likeness check, the three lesion methods |
| `src/mask_utils.py` | small morphology helpers: open, close, largest blob, remove small blobs, area |
| `src/refine.py` | lesion clean-up, Canny edge map, contour measurements (`Lesion`), overlay drawing |
| `src/features.py` | the 90 features (`extract_features`) and `feature_group` (colour / texture / shape) |
| `src/severity.py` | lesion area over leaf area, and the level names |
| `src/pipeline.py` | **the one chain**: `analyze_array`, `analyze_file`, `feature_dict`, and the `Analysis` record |
| `src/predict.py` | **the one prediction function**: `predict_array`, `predict_file`, command line (`python -m src.predict photo.jpg`) |
| `src/gui.py`, `src/gui_page.html` | the GUI: Flask server (route `/analyse`) and the single web page |
| `src/visualize.py` | save a montage of every stage for an image (`python -m src.visualize`) |

### Training and evaluation (Job A)

| File | Role | Reads → writes |
|---|---|---|
| `src/dataset.py` | choose 800 images per class (seed 42), group neighbouring image numbers in blocks of 20, split the groups into train / val / test | PlantVillage folders → `data/raw/`, `data/splits/splits.csv` |
| `src/build_features.py` | run the pipeline on all 4,800 images in parallel | `splits.csv` → `data/features_<method>.csv` |
| `src/modeling.py` | build the scikit-learn pipeline (scaler → SelectKBest → SVM / forest / k-NN) and the parameter grids | used by train and evaluate |
| `src/train.py` | tune 3 classifiers with grouped 5-fold CV on train, pick the best on validation, refit on train+val, calibrate probabilities | features CSV → `models/classifier.joblib`, `results/model_selection.csv`, `best_params.json` |
| `src/evaluate.py` | `--ablations`: which features matter, leaky vs grouped CV; `--final`: score the test split **once** | model + features → `results/test_results.json`, confusion matrix, misclassified montage, ablation CSVs |
| `src/annotations.py` | the 24 hand-made lesion masks (`data/annotations/shapes.json`): list, load, draw to mask | used by the lesion evaluation |
| `src/evaluate_lesions.py` | Dice, false-alarm and severity error of the three lesion methods on the annotated leaves (`--role dev` / `holdout`) | → `results/lesion_*.csv`, failure montages |
| `src/evaluate_leaf_mask.py` | IoU of our leaf mask against PlantVillage's own background-removed images | → `results/leaf_mask_iou_train.csv` |
| `src/evaluate_false_alarms.py` | lesion percentage on many images without annotations (healthy vs diseased AUC) | → `results/false_alarm_check_train.csv` |
| `src/audit_split.py` | check how many test images still have a very close look-alike in train | prints a table |

### Tests

| File | Role |
|---|---|
| `tests/test_pipeline.py` | 15 tests on synthetic images with known answers (a green disc with a brown square of known area, etc.) |
| `tests/test_gui.py` | 9 tests of the web server: valid photo, bad file, no leaf, too large, missing model |

### Report (`report/`)

| File | Role |
|---|---|
| `make_figures.py` | the Chapter 5 pictures (one leaf through every OpenCV function), pipeline diagram, lesion-method comparison, ablation charts, GUI screenshots |
| `make_figures_extra.py` | extra comparisons (filters, edges, thresholds, morphology, equalisation), per-class gallery, severity / confidence plots; writes the numbers the text quotes to `figures/extra_numbers.json` |
| `docx_helpers.py` | IEEE-style Word building blocks: styles, figures, tables, equations, two-column sections |
| `build_report.py` | writes the whole report on top of `lab_template.docx` (cover, certificate, contents from the template; the chapters are built here) |
| `find_toc_pages.py` | reads the rendered PDF and stores the contents-page numbers in `toc_pages.json` |
| `lab_template.docx`, `figures/` | the department template and every picture used |

## 4. The order to run things (and why)

1. `src.dataset` - only if `data/raw` is missing. Makes the data and the split.
2. `src.build_features` - slow step (2 min); caches the 90 numbers per image so training does not repeat it.
3. `src.train` - uses **train and validation only**; saves the model.
4. `src.evaluate --ablations` - analysis using train+val only.
5. `src.evaluate --final` - the test split, once.
6. `src.gui` / `src.predict` - use the saved model on new photos (needs steps 1-3 done, or the model that is included).

## 5. Things a viva examiner may ask, and where the answer is

| Question | Answer in |
|---|---|
| Why is there no data leakage? | `dataset.py` (blocks of 20 ids), `modeling.py` (scaler and selector inside the Pipeline), `audit_split.py` |
| Why not "non-green = diseased"? | `segmentation.lesion_mask_hue_rule` vs `lesion_mask_own_green`; README section 4 |
| Why does Canny not fix the lesion mask? | `pipeline.analyze_array`: the lesion mask is finished before the edge map is computed; report Section 5.8 |
| How is confidence computed? | `train.calibrate_for_confidence` (sigmoid calibration, grouped folds) |
| Why is the test result trustworthy? | `evaluate.run_final_test` refuses to run twice; test never used in tuning |
| What are the weak points? | README section 5; report Chapter 6 |
