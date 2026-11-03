"""
build_report.py - assemble the FCV mini-project report on top of the department's Word template.

The cover, certificate and contents pages come from the template (only placeholders are filled in);
everything after them is IEEE-style: Times New Roman 10 pt, two columns, numbered figures/tables/equations.
Every number in the text comes from results/*.csv|json (see the comments next to the tables).

Run from the project root:
    python -m report.make_figures                 # pictures
    python -m report.build_report [--toc-pages report/toc_pages.json]
"""
import argparse
import copy
import json
import re
import shutil
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_COLOR_INDEX, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from report import docx_helpers as H
from report.docx_helpers import COLUMN_WIDTH, FULL_WIDTH, Wide

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).parent / "lab_template.docx"      # the department template, converted from .doc to .docx
FIG = Path(__file__).parent / "figures"
OUTPUT = Path(__file__).parent / "FCV_Mini_Project_Report.docx"

TITLE = "Plant Leaf Disease Detection and Severity Estimation using Classical Computer Vision"
STUDENTS = [("Tejas Bharagva", "240962448", "33"), ("Mav Himanshu Mukesh", "240962588", "54")]

# Figure / table numbers follow the order below (the build asserts that the document really has this order),
# so text can refer to {fig:key} and {tab:key} before the item appears.
FIG_ORDER = ["samples", "pipeline", "stages", "leaf_iou", "methods", "failures", "confusion", "ablation", "wrong",
             "features_pca", "gallery", "sevconf", "gui", "gui_err",
             "colour", "resize", "gauss", "filters", "sharpen", "clahe", "equalise", "thresh", "threshcmp",
             "morph", "morphops", "derivs", "canny", "cannysweep", "contours", "kmeans", "texture"]
TAB_ORDER = ["split", "features", "params", "modules", "lesion", "models", "per_class", "gallery_t", "severity",
             "functions", "filters_t", "equalise_t", "morph_t", "summary"]
F = {key: i + 1 for i, key in enumerate(FIG_ORDER)}


def roman(n):
    out = ""
    for value, symbol in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= value:
            out += symbol
            n -= value
    return out


T = {key: roman(i + 1) for i, key in enumerate(TAB_ORDER)}
NUM = json.loads((Path(__file__).parent / "figures" / "extra_numbers.json").read_text())


def resolve(text):
    """Replace {fig:key} and {tab:key} by their numbers."""
    text = re.sub(r"\{fig:(\w+)\}", lambda m: str(F[m.group(1)]), text)
    return re.sub(r"\{tab:(\w+)\}", lambda m: T[m.group(1)], text)


_add_runs = H.add_runs


def add_runs_resolved(paragraph, text, *args, **kwargs):
    return _add_runs(paragraph, resolve(text), *args, **kwargs)


H.add_runs = add_runs_resolved


def table(doc, key, *args, **kwargs):
    assert H.Counter.tables + 1 == TAB_ORDER.index(key) + 1, f"table order: {key}"
    H.add_table(doc, *args, **kwargs)


# =============================================================================
# front pages (from the template)
# =============================================================================
def replace_text(paragraph, old, new, highlight=False):
    for run in paragraph.runs:
        if old in run.text:
            run.text = run.text.replace(old, new)
            if highlight:
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
            return True
    raise ValueError(f"'{old}' not found in paragraph")


def fill_front_matter(doc):
    paragraphs = doc.paragraphs
    # cover
    replace_text(paragraphs[9], "TITLE", TITLE)
    for run in paragraphs[9].runs:
        run.font.size = Pt(20)
    first = paragraphs[18]
    first.runs[0].text = f"{STUDENTS[0][0]} (Reg. No. {STUDENTS[0][1]}, Roll No. {STUDENTS[0][2]}, Sec. [__])"
    second = copy.deepcopy(first._p)
    first._p.addnext(second)
    from docx.text.paragraph import Paragraph
    second_paragraph = Paragraph(second, first._parent)
    second_paragraph.runs[0].text = f"{STUDENTS[1][0]} (Reg. No. {STUDENTS[1][1]}, Roll No. {STUDENTS[1][2]}, Sec. [__])"
    for paragraph in (first, second_paragraph):
        for run in paragraph.runs:
            run.font.size = Pt(12)
    # the two paragraphs share a highlight on the section placeholder
    # certificate
    paragraphs = doc.paragraphs                     # indices shifted by one after the insertion
    certificate = next(p for p in paragraphs if p.text.startswith("This is to certify"))
    replace_text(certificate, "MiniProject Title", TITLE)
    replace_text(certificate, "Student(s) (Reg. No. 000000)",
                 f"{STUDENTS[0][0]} (Reg. No. {STUDENTS[0][1]}) and {STUDENTS[1][0]} (Reg. No. {STUDENTS[1][1]})")
    replace_text(certificate, "------------------ENGINEERING", "COMPUTER SCIENCE AND ENGINEERING")
    date = next(p for p in paragraphs if p.text.strip() == "00/00/2026")
    replace_text(date, "00/00/2026", "[DD/MM/2026]", highlight=True)
    tidy_page_breaks(doc)


def tidy_page_breaks(doc):
    """The template pushes the contents onto its own page with many empty paragraphs; longer names on the
    certificate then spill over. Replace the filler with a real page break and drop a few spacers."""
    paragraphs = doc.paragraphs
    toc = next(p for p in paragraphs if p.text.strip() == "TABLE OF CONTENTS")
    previous = toc._p.getprevious()
    while previous is not None and previous.tag == qn("w:p") and not "".join(previous.itertext()).strip():
        before = previous.getprevious()
        if previous.find(".//" + qn("w:numPr")) is None:      # keep the empty numbered "2." examiner line
            previous.getparent().remove(previous)
        previous = before
    toc.paragraph_format.page_break_before = True
    start = next(i for i, p in enumerate(doc.paragraphs) if p.text.strip() == "CERTIFICATE")
    spacers = [p for p in doc.paragraphs[start - 8:start] if not p.text.strip()]
    for spacer in spacers[:2]:
        spacer._p.getparent().remove(spacer._p)


TOC_ENTRIES = ["ABSTRACT", "CHAPTER 1:", "CHAPTER 2:", "CHAPTER 3:", "CHAPTER 4:", "CHAPTER 5:",
               "CHAPTER 6:", "CHAPTER 7:"]


def fill_toc(doc, pages):
    """Add dotted leaders and page numbers to the template's contents lines."""
    entries = [p for p in doc.paragraphs
               if any(p.text.strip().startswith(key) for key in TOC_ENTRIES)
               and p.text.strip() != "ABSTRACT" or p.text.strip() == "ABSTRACT"]
    entries = [p for p in entries if p.text.strip().split(":")[0] in
               ("ABSTRACT", "CHAPTER 1", "CHAPTER 2", "CHAPTER 3", "CHAPTER 4", "CHAPTER 5", "CHAPTER 6", "CHAPTER 7")]
    keys = ["ABSTRACT", "1", "2", "3", "4", "5", "6", "7"]
    for paragraph, key in zip(entries, keys):
        paragraph.paragraph_format.tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        paragraph.alignment = None
        source = paragraph.runs[0]
        run = copy.deepcopy(source._r)
        paragraph._p.append(run)
        for text in run.findall(qn("w:t")):
            text.text = "\t" + (str(pages.get(key, "")) if pages else "")
            text.set(qn("xml:space"), "preserve")


# =============================================================================
# body
# =============================================================================
def check(figure_key):
    assert H.Counter.figures + 1 == F[figure_key], f"figure order: {figure_key} should be {F[figure_key]}"


def fig(doc, key, name, caption, width=COLUMN_WIDTH - 0.05):
    check(key)
    H.add_figure(doc, FIG / name, caption, width)


CLASS = {"Healthy": "Healthy", "Bacterial_spot": "Bacterial spot", "Early_blight": "Early blight",
         "Late_blight": "Late blight", "Leaf_Mold": "Leaf mold", "Septoria_leaf_spot": "Septoria leaf spot"}


def abstract(doc):
    H.add_chapter(doc, "ABSTRACT")
    H.add_text(doc, "Plant diseases reduce crop quality and yield, and recognising them by eye needs experience. "
               "This project builds a disease detector for tomato leaves using only classical computer vision and "
               "classical machine learning, with no deep learning. A leaf photograph is smoothed with a Gaussian "
               "filter and contrast-enhanced with CLAHE; the leaf is separated from its background by Lab colour "
               "distance, Otsu thresholding and hysteresis; diseased pixels are found by their colour distance from the "
               "leaf's own healthy green; the lesion mask is cleaned with morphological opening and closing and "
               "outlined with contours, and a Canny edge map is produced as an additional view and feature. From the masks, 90 colour, "
               "texture (GLCM, LBP) and lesion-shape (contour measures, Hu moments) features are extracted and "
               "classified by a support vector machine into six PlantVillage classes. Severity is the percentage of "
               "leaf area covered by lesions. On a held-out test set of 757 images, scored once, the system reaches "
               "**90.8% accuracy (macro-F1 0.909)**. The leaf mask has a mean IoU of 0.924, and lesion overlap (Dice) "
               "is about 0.41 on 12 annotated hold-out leaves, which is modest. A local web interface displays every "
               "processing stage. Limitations are stated: laboratory-condition photographs, approximate reference "
               "masks and colour-based severity.", style="R Abstract")
    H.add_text(doc, "*Index Terms*—plant disease detection, classical computer vision, colour segmentation, morphology, "
               "Canny edge detection, GLCM, support vector machine, severity estimation, PlantVillage.", style="R Keywords")


def chapter1(doc):
    H.add_chapter(doc, "CHAPTER 1: INTRODUCTION")
    H.add_heading2(doc, "1.1 Background")
    H.add_text(doc, "Diseases of crop leaves lower yield and quality, and the earlier a disease is noticed the cheaper it is "
               "to contain. Scouting by eye is slow, subjective and depends on the observer's experience. Image-based "
               "methods aim to give a first, repeatable assessment from a photograph of the leaf [17]. Recent work reports "
               "very high accuracy with deep convolutional networks on the PlantVillage photographs [2], but those models need large "
               "training sets and offer little insight into why a leaf was labelled the way it was; studies of "
               "such systems also stress that results on controlled photographs do not automatically carry over to the "
               "field [17], [18].")
    H.add_heading2(doc, "1.2 Motivation and approach")
    H.add_text(doc, "This mini-project takes the opposite route and builds the system from the techniques of the Foundations of "
               "Computer Vision laboratory: filtering, colour-space conversion, thresholding, morphology, edge detection, "
               "contour analysis and texture description, followed by a classical classifier. Every stage "
               "produces an image or a number that can be inspected, which makes the system explainable and each design "
               "choice testable. The cost is accuracy on difficult cases, and the report measures that cost instead of "
               "hiding it.")
    H.add_heading2(doc, "1.3 Scope and contributions")
    H.add_bullets(doc, [
        "A complete classical pipeline for tomato leaves: pre-processing, leaf and lesion segmentation, refinement, "
        "contour measurement and a 90-value feature vector (Chapter 3).",
        "Three lesion-segmentation methods compared on hand-annotated masks, including failure cases (Chapter 4).",
        "A split-leakage audit of PlantVillage, whose consecutive image numbers are photographs of the same leaf, and a "
        "grouped train/validation/test protocol that reduces it.",
        "Severity estimated from measured lesion and leaf masks, and a local web GUI that shows every stage.",
    ])
    H.add_text(doc, "Chapter 2 states the problem and objectives, Chapter 3 the methodology, Chapter 4 the results with "
               "snapshots, Chapter 5 the OpenCV and scikit-image functions used from the laboratory sessions, and Chapter 6 "
               "the conclusions, limitations and future work.")


def chapter2(doc):
    H.add_chapter(doc, "CHAPTER 2: PROBLEM STATEMENT & OBJECTIVES")
    H.add_heading2(doc, "2.1 Problem statement")
    H.add_text(doc, "Given one photograph of a tomato leaf, the system must (i) reject files that are not usable leaf images, "
               "(ii) locate the leaf, (iii) locate the diseased regions on it, (iv) assign one of six categories "
               "(healthy or one of five diseases) and (v) estimate how much of the leaf is affected, using classical "
               "computer vision and a classical machine-learning classifier, and present the stages and results in a "
               "graphical interface.")
    H.add_heading2(doc, "2.2 Objectives")
    H.add_bullets(doc, [
        "Build a reproducible pipeline whose every stage can be displayed and explained.",
        "Segment the leaf and the lesions with colour-space methods and compare at least two lesion methods against "
        "annotated masks instead of assuming that “not green” means “diseased”.",
        "Describe each leaf with colour, texture and lesion-shape features and train a classifier with proper "
        "train/validation/test separation, fitting scaling and feature selection on training data only.",
        "Estimate severity as lesion area over leaf area and state its limitations.",
        "Report only measured results, including failures, and provide a GUI that handles invalid input.",
    ])
    H.add_heading2(doc, "2.3 Scope and assumptions")
    H.add_text(doc, "The system is trained and evaluated on the tomato subset of PlantVillage [1]: one leaf per photograph on a "
               "plain background, six classes. Severity levels are working proposals, not an official plant-pathology "
               "scale, and the output is an aid to inspection, not a diagnosis.")


def chapter3(doc):
    H.add_chapter(doc, "CHAPTER 3: METHODOLOGY")
    H.add_heading2(doc, "3.1 Dataset and data split")
    H.add_text(doc, "PlantVillage [1] contains 54,306 laboratory photographs of leaves. We use six tomato classes (healthy, "
               "bacterial spot, early blight, late blight, leaf mold, Septoria leaf spot), sampling 800 images per class "
               "with a fixed seed (42), 4,800 images in total. Two of them (completely dead, dark-brown leaves whose "
               "colour matches the background) cannot be analysed and are reported as “no leaf found”.")
    H.add_text(doc, "PlantVillage photographs the same leaf several times and numbers the files consecutively, so a random "
               "split puts near-twins in both training and test sets and inflates accuracy. We therefore group "
               "images into blocks of 20 consecutive numbers (759 groups) and assign whole groups to the training, "
               "validation or test split (Table {tab:split}). A nearest-twin audit (`src/audit_split.py`) shows that the share of "
               "test images with a very close colour match in the training set falls from about 12–15% for a random "
               "split to about 5–7%. Leakage is reduced, not removed.")
    table(doc, "split", ["Class", "Train", "Val", "Test"],
                [["Healthy", 545, 128, 127], ["Bacterial spot", 554, 121, 125], ["Early blight", 542, 128, 130],
                 ["Late blight", 557, 123, 120], ["Leaf mold", 540, 134, 126], ["Septoria leaf spot", 543, 128, 129],
                 ["Total", 3281, 762, 757]],
                [1.4, 0.65, 0.65, 0.65], "Images per class in each grouped split", bold_rows=(6,))
    H.add_text(doc, "Fig. {fig:samples} shows four random training photographs of each class. Every photograph is a single "
               "leaf on a plain grey, violet or blue background, but appearance varies strongly inside a class (late blight "
               "ranges from a few olive patches to a leaf mostly covered with grey-brown tissue) and the lesions of different "
               "diseases overlap in colour, yellow-brown spots on early blight, Septoria and bacterial spot above all. Many leaves "
               "are curled, partly shadowed or cut by the frame. This is why a single colour rule is not enough and why the "
               "classifier also uses texture and shape.")
    with Wide(doc):
        fig(doc, "samples", "fig3_samples.png", "Four random training photographs of each class (fixed seed, not selected).", 6.3)
    H.add_heading2(doc, "3.2 System overview")
    H.add_text(doc, "Fig. {fig:pipeline} shows the processing chain. One function chains all stages and is called by the training "
               "scripts, the evaluation scripts and the GUI, so the three can never disagree about how an image is processed. "
               "Every threshold lives in a single configuration file with the reason for its value.")
    with Wide(doc):
        fig(doc, "pipeline", "fig_pipeline.png", "Processing chain. Blue boxes produce images or masks; the two outputs "
            "(severity and disease label) are shown in red.", FULL_WIDTH)
    H.add_heading2(doc, "3.3 Pre-processing")
    H.add_text(doc, "Photographs larger than 256 pixels on the longest side are shrunk (area interpolation), so that pixel-based "
               "thresholds mean the same thing for every image. A 5×5 Gaussian filter,")
    H.add_equation(doc, "*G*(*x*, *y*) = [1 / (2π*σ*^{2})] exp[ −(*x*^{2} + *y*^{2}) / (2*σ*^{2}) ]")
    H.add_text(doc, "removes sensor noise that would otherwise create many false edges and specks. Contrast is then raised "
               "with CLAHE [5] (clip limit 2.0, 8×8 tiles) applied to the lightness channel L of Lab space only, so that "
               "lesions stand out without changing the leaf's colours, which the classifier relies on. The blurred image "
               "is used for colour measurements and the CLAHE image for edges and texture.", style="R Body Flush")
    H.add_heading2(doc, "3.4 Leaf segmentation")
    H.add_text(doc, "The photo border (12 pixels) is assumed to be background. Its median chromatic colour "
               "(*a*_{bg}, *b*_{bg}) in Lab space is measured, and each pixel is labelled by its colour distance to it:")
    H.add_equation(doc, "*d*_{bg} = √[ (*a* − *a*_{bg})^{2} + (*b* − *b*_{bg})^{2} ]")
    H.add_text(doc, "Lightness is ignored on purpose: a cast shadow is darker than the background but has the same colour, so "
               "it disappears. A first attempt based on HSV saturation kept cast shadows as part of the leaf, "
               "which is why the chroma distance was adopted; it reaches a mean IoU of 0.924 (Chapter 4). Otsu's method [3] chooses the threshold that "
               "maximises the between-class variance,", style="R Body Flush")
    H.add_equation(doc, "*t*_{Otsu} = argmax_{t} *ω*_{0}(*t*) *ω*_{1}(*t*) [ *μ*_{0}(*t*) − *μ*_{1}(*t*) ]^{2}")
    H.add_text(doc, "and hysteresis thresholding [4] keeps weaker pixels (above 0.7× the Otsu value) only if they touch "
               "pixels above it, which recovers pale or dead leaf tissue without admitting background noise. Opening "
               "(3×3) removes specks, the largest connected component is kept and its holes are filled. A mask "
               "covering less than 5% or more than 97% of the image, or one that has too little plant colour (fewer "
               "than 30% of pixels with hue 5–95 and saturation ≥ 40) or is too fragmented (solidity < 0.35), is "
               "rejected as “no leaf found”. The two limits were set just below the minimum measured on all 4,798 "
               "analysable images (0.34 and 0.42), so no dataset image is rejected; the check is coarse.", style="R Body Flush")
    H.add_heading2(doc, "3.5 Lesion segmentation")
    H.add_text(doc, "Three methods take the blurred image and the leaf mask (shrunk by a 7×7 erosion, because rim pixels "
               "blend leaf and background) and return a lesion mask:")
    H.add_bullets(doc, [
        "**A, hue rule (baseline):** every leaf pixel whose HSV hue lies outside the green band 35–85 is a lesion.",
        "**B, K-means [14]:** the leaf's (*a*, *b*) colours are clustered into three groups and the least green group "
        "is called the lesion. K-means always returns three groups, even for a healthy leaf.",
        "**C, own-green (used):** the leaf's healthy reference colour (*a*_{ref}, *b*_{ref}) is the median chromatic "
        "colour of its greenest quarter of pixels; a pixel is a lesion if",
    ])
    H.add_equation(doc, "*d*_{ref} = √[ (*a* − *a*_{ref})^{2} + (*b* − *b*_{ref})^{2} ] > *τ*,   *τ* = 14")
    H.add_text(doc, "Only colour is compared, not lightness, because veins, folds and shading change lightness; adding "
               "lightness raised healthy-leaf false alarms from about 3% to about 18%. The floor *τ* keeps a healthy leaf "
               "at roughly zero lesion area. It was not retuned: a small grid on the 12 development annotations changed "
               "Dice by at most 0.02, which is noise at that sample size.", style="R Body Flush")
    H.add_heading2(doc, "3.6 Refinement, contours and edges")
    H.add_text(doc, "The raw lesion mask is opened (3×3 ellipse) to delete speckles, closed (3×3) to join a lesion "
               "split into nearby fragments, and blobs under 8 pixels are dropped. Each remaining blob is outlined with "
               "`findContours` and measured: area *A*, perimeter *P*, solidity (area over convex-hull area), aspect "
               "ratio of the bounding box, and circularity")
    H.add_equation(doc, "*C* = 4π*A* / *P*^{2}")
    H.add_text(doc, "which is 1 for an ideal circle. On a pixel grid a digital circle scores only about 0.67 at radius "
               "3 and 0.89 at radius 60, so circularity is used as a relative round-versus-elongated measure.",
               style="R Body Flush")
    H.add_text(doc, "The Canny detector [4] is run on the CLAHE grey image inside the leaf core. Its two thresholds are derived "
               "from the median brightness *m* of the leaf pixels,")
    H.add_equation(doc, "*T*_{low} = (1 − *σ*)*m*,   *T*_{high} = (1 + *σ*)*m*,   *σ* = 0.33")
    H.add_text(doc, "so dark and bright photos are both handled. Canny gives an edge picture for the interface and one "
               "feature (edge density). **It does not edit or correct the lesion mask.**", style="R Body Flush")
    H.add_heading2(doc, "3.7 Feature extraction")
    H.add_text(doc, "Table {tab:features} lists the 90 features. Colour features describe the leaf and the lesion regions separately; "
               "texture uses the grey-level co-occurrence matrix (GLCM) [6], averaged over four directions on 32 grey "
               "levels at distances 1 and 3, and uniform local binary patterns (LBP) [7] with 8 neighbours and radius 1; "
               "shape features summarise the lesion blobs and add the seven Hu moments [8] of the lesion mask, "
               "log-scaled.")
    table(doc, "features", ["Family", "No.", "Contents"],
                [["Colour", 55, "mean, std, skew of H, S, V, L, a, b over leaf and over lesion (36); 8-bin hue and "
                  "saturation histograms (16); green, yellow, brown fractions (3)"],
                 ["Texture", 20, "GLCM contrast, dissimilarity, homogeneity, energy, correlation at two distances (10); "
                  "LBP histogram (10)"],
                 ["Shape", 15, "lesion count, area ratio, edge density, mean and largest lesion area, mean circularity, "
                  "solidity and aspect ratio (8); Hu moments (7)"],
                 ["Total", 90, ""]],
                [0.6, 0.35, 2.4], "The 90 features by family", align=["L", "C", "L"], bold_rows=(3,))
    H.add_heading2(doc, "3.8 Classification")
    H.add_text(doc, "Scaling, feature selection and the classifier form one scikit-learn pipeline [10]: standardisation, "
               "then the *k* best features by ANOVA F-score, then the classifier. Because the whole pipeline is fitted "
               "inside each cross-validation fold, nothing is learned from held-out images. Three classifiers were "
               "tuned by grid search with grouped 5-fold cross-validation on the training set (macro-F1): an RBF-kernel "
               "support vector machine [9] (*C* ∈ {1, 10, 100}, *γ* ∈ {scale, 0.01, 0.001}), a random forest "
               "and *k*-nearest neighbours, each with *k* ∈ {30, 60, all} features for the selector. The best classifier on "
               "the validation split was refitted on training plus validation images, and its scores were turned into "
               "probabilities by sigmoid calibration [13] with grouped folds, so the interface can show a trustworthy "
               "confidence. The test split was scored once.")
    H.add_heading2(doc, "3.9 Severity estimation")
    H.add_text(doc, "Severity is the share of leaf area covered by lesions, measured in the same rim-trimmed region:")
    H.add_equation(doc, "*S* = 100 × *A*_{lesion} / *A*_{leaf}   [%]")
    H.add_text(doc, "It is reported as Trace/healthy (≤ 5%), Mild (≤ 15%), Moderate (≤ 35%) or Severe. "
               "These limits are our proposal. Severity is an area in a flat photograph and is only as good as the "
               "two masks.", style="R Body Flush")
    H.add_heading2(doc, "3.10 Evaluation protocol")
    H.add_text(doc, "Classification is scored by accuracy, macro-F1 and the confusion matrix. Masks are scored by the "
               "intersection over union and the Dice coefficient [15] against reference masks:")
    H.add_equation(doc, "IoU = |*A* ∩ *B*| / |*A* ∪ *B*|,   Dice = 2|*A* ∩ *B*| / (|*A*| + |*B*|)")
    H.add_text(doc, "The leaf mask is compared with PlantVillage's own background-removed images (300 training images). "
               "For lesions, 24 leaves (12 development, 12 hold-out; 10 diseased and 2 healthy in each) were annotated "
               "following a written protocol. **These reference masks were drawn by the project's AI assistant, not by a plant "
               "pathologist**; they were checked visually and are conservative, but they are approximate, and the hold-out "
               "set was scored once. Healthy-leaf false-alarm rate and severity error (in percentage points) are also reported.",
               style="R Body Flush")
    H.add_heading2(doc, "3.11 Implementation and interface")
    H.add_text(doc, "The code is Python 3.13 with OpenCV 5.0.0 [11], scikit-image [12], scikit-learn [10] and NumPy [19]; 24 "
               "unit tests check the building blocks on synthetic images with known answers. The graphical interface is a "
               "small local web page served by Flask [20] on the user's own computer. It sends the chosen photo to the same "
               "prediction function used everywhere else and shows the six stage images, the label with its "
               "probabilities and a severity ruler. Unreadable files, tiny images and photographs without a leaf produce a "
               "short message instead of a result.")
    H.add_heading2(doc, "3.12 Parameters and software structure")
    H.add_text(doc, "Table {tab:params} lists the values that matter, each defined once in `src/config.py` with the reason "
               "next to it, and Table {tab:modules} lists the modules. Training, evaluation and the interface all call the "
               "same pipeline module, so a parameter change reaches every result consistently. The 24 unit tests (15 for the "
               "pipeline, 9 for the interface) use synthetic images whose correct answer is known, for example a green disc with "
               "a brown square of known area.")
    table(doc, "params", ["Parameter", "Value", "Why"],
          [["Longest image side", "256 px", "pixel thresholds mean the same for every photo"],
           ["Gaussian kernel", "5\u00d75", "removes speckle; larger kernels erase small lesions"],
           ["CLAHE", "clip 2.0, 8\u00d78", "contrast on L only; caps noise gain"],
           ["Background border", "12 px", "photo edge is assumed to be background"],
           ["Hysteresis low / high", "0.7\u00d7Otsu / Otsu", "recovers pale leaf tissue"],
           ["Rim erosion", "7\u00d77", "rim pixels blend leaf and background"],
           ["Healthy reference", "greenest 25%", "median (a, b) of the leaf itself"],
           ["Lesion colour floor *\u03c4*", "14 (Lab a, b)", "healthy leaf stays near zero lesion"],
           ["Opening, closing", "3\u00d73 ellipse", "speckle off, split lesions joined"],
           ["Smallest lesion", "8 px", "smaller blobs are treated as noise"],
           ["Canny *\u03c3*", "0.33", "thresholds from the median grey level"],
           ["Severity limits", "5, 15, 35 %", "proposed, not an official scale"],
           ["Random seed", "42", "sampling, splits and models"],
           ["Split block", "20 ids", "keeps near-duplicate photos together"]],
          [1.25, 0.9, 1.3], "Main parameters of the pipeline", align=["L", "L", "L"])
    table(doc, "modules", ["Module", "Role"],
          [["`preprocess`, `segmentation`", "load and clean images; leaf mask; the three lesion methods"],
           ["`refine`, `mask_utils`", "morphology, contours, Canny, lesion measures"],
           ["`pipeline`, `features`", "the one chain of stages; the 90 features"],
           ["`severity`, `predict`", "lesion over leaf area; label, confidence, warnings"],
           ["`dataset`, `build_features`, `train`", "subset and grouped split; feature cache; model fitting"],
           ["`evaluate*`, `audit_split`", "test scores, mask scores, false alarms, leakage audit"],
           ["`gui`, `visualize`", "local web interface; stage pictures"]],
          [1.25, 2.1], "Modules in the project", align=["L", "L"])


def chapter4(doc):
    H.add_chapter(doc, "CHAPTER 4: RESULTS & SNAPSHOTS")
    H.add_heading2(doc, "4.1 Processing stages")
    H.add_text(doc, "Fig. {fig:stages} follows one bacterial-spot test leaf through the pipeline: the denoised and contrast-enhanced "
               "images, the leaf mask, the raw and cleaned lesion masks, the Canny edge map and the final contours with "
               "bounding boxes. This leaf is measured at 6.9% (Mild).")
    with Wide(doc):
        fig(doc, "stages", "stages_montage.png", "Stage-by-stage output for a bacterial-spot leaf.", FULL_WIDTH)
    H.add_heading2(doc, "4.2 Leaf segmentation")
    H.add_text(doc, "Against PlantVillage's own background-removed images the leaf mask reaches a mean IoU of **0.924** "
               "(median 0.946) on 300 training images (Fig. {fig:leaf_iou}). The tail on the left is mostly leaf-mold photographs on "
               "blue backgrounds (class mean 0.87).")
    fig(doc, "leaf_iou", "fig4_leaf_iou.png", "Leaf-mask IoU on 300 training images.")
    H.add_heading2(doc, "4.3 Lesion segmentation")
    H.add_text(doc, "Fig. {fig:methods} compares the three methods on annotated leaves, and Table {tab:lesion} gives the scores on the 12 "
               "development and the 12 hold-out annotations. Dice is low for every method (about 0.4), "
               "because colour thresholds miss faint yellowing, tiny specks and pale dead tissue. K-means is "
               "rejected: it fills about half of a healthy leaf (49.6% in Fig. {fig:methods}) and flags 40% of healthy-leaf area on "
               "the hold-out set. The hue rule over-calls a naturally yellow leaf (65.0% against a true 13.2% in the first "
               "row). The own-green method halves that error (28.9%) but still over-calls it.")
    H.add_text(doc, "The numbers support a modest claim. On the hold-out set own-green ties the hue rule on Dice (0.407 against "
               "0.406) and on the development set it is lower (0.380 against 0.415). Its consistent advantage is the "
               "severity error (6.9 and 6.6 points against 8.3 and 12.1) and the separation of healthy from diseased "
               "leaves: on 600 training images the area of the lesion mask separates the two groups with an AUC of 0.860, "
               f"against 0.826 for the hue rule and 0.262 for K-means. Typical failures (Fig. {F['failures']}) are veins along the "
               "midrib flagged as lesion, a large grey dead area missed (a hold-out late-blight leaf with a true "
               "29.3% measured at 7.4%), and faint bacterial specks missed.", style="R Body Flush")
    with Wide(doc):
        fig(doc, "methods", "fig4_lesion_methods.png", "Lesion masks (red) of the three methods on three annotated "
            "hold-out leaves; the percentage is the measured severity. Yellow outlines in the first column are the "
            "annotated lesions.", 6.0)
        table(doc, "lesion", ["Method", "Dev. Dice", "Dev. false alarm %", "Dev. severity MAE", "Hold-out Dice",
                          "Hold-out false alarm %", "Hold-out severity MAE"],
                    [["Hue rule (baseline)", "0.415", "2.0", "8.3", "0.406", "1.2", "12.1"],
                     ["K-means", "0.412", "31.3", "7.5", "0.296", "39.8", "21.6"],
                     ["Own-green (used)", "0.380", "2.6", "6.9", "0.407", "0.6", "6.6"]],
                    [1.55, 0.8, 0.95, 0.95, 0.8, 1.0, 0.95],
                    "Lesion segmentation on 12 development and 12 hold-out annotated leaves (10 diseased, 2 healthy "
                    "each). False alarm = share of healthy-leaf area flagged; MAE in percentage points of severity.")
        fig(doc, "failures", "lesion_failures_own_green_holdout.png", "Own-green errors on hold-out leaves. "
            "Blue marks missed lesion area and red a false alarm.", 2.1)
    H.add_heading2(doc, "4.4 Classification")
    H.add_text(doc, f"The SVM was selected on validation macro-F1 (Table {T['models']}) and then scored once on the 757 test images: "
               f"**accuracy 90.8%, macro-F1 0.909** (95% interval about ±2 points). Table {T['per_class']} and Fig. {F['confusion']} show "
               "that healthy and bacterial-spot leaves are recognised best and early blight is the weakest class "
               "(F1 0.813). Most errors are between the spotty diseases, for example leaf mold to early blight (9 images), "
               "early blight to Septoria (8), late blight to early blight (8) and Septoria to early blight (8); several "
               f"of these are ambiguous even to a person (Fig. {F['wrong']}).")
    table(doc, "models", ["Classifier", "CV macro-F1", "Val. acc.", "Val. macro-F1"],
                [["SVM (RBF), *C* = 10, 60 features", "0.891", "0.891", "0.891"],
                 ["Random forest (400 trees)", "0.859", "0.866", "0.865"],
                 ["*k*-NN (*k* = 5)", "0.836", "0.854", "0.852"]],
                [1.65, 0.6, 0.55, 0.6], "Model selection (grouped 5-fold CV on training data; validation split)",
                align=["L", "C", "C", "C"])
    table(doc, "per_class", ["Class", "Precision", "Recall", "F1", "Support"],
                [["Healthy", "0.962", "0.984", "0.973", 127], ["Bacterial spot", "0.991", "0.928", "0.959", 125],
                 ["Early blight", "0.790", "0.838", "0.813", 130], ["Late blight", "0.923", "0.900", "0.911", 120],
                 ["Leaf mold", "0.900", "0.929", "0.914", 126], ["Septoria leaf spot", "0.896", "0.868", "0.882", 129],
                 ["Macro average", "0.910", "0.908", "0.909", 757]],
                [1.2, 0.5, 0.5, 0.5, 0.5], "Test-set results (757 images, scored once)", bold_rows=(6,))
    fig(doc, "confusion", "confusion_matrix_test.png", "Test-set confusion matrix.", COLUMN_WIDTH - 0.25)
    H.add_heading2(doc, "4.5 Ablations")
    H.add_text(doc, f"Fig. {F['ablation']} shows what each feature family contributes: colour alone gives 0.835 validation macro-F1, texture "
               "0.704 and shape 0.528, and all three together 0.891. The right panel shows why the split matters: the "
               "same model scores 92.7% with ordinary random folds and 88.7% with grouped folds, so about four points "
               "of apparent accuracy come from near-duplicate photographs. Finally, features built from the three lesion "
               "methods classify about equally well (validation accuracy 0.894 hue rule, 0.906 K-means, 0.891 own-green; "
               "differences within noise at 762 images) even though K-means makes the worst masks. **Classification "
               "accuracy therefore does not validate the lesion masks**; masks and severity have to be judged on "
               "masks.")
    fig(doc, "ablation", "fig4_ablations.png", "Left: feature-family ablation. Right: random versus grouped "
        "cross-validation for the same model.")
    fig(doc, "wrong", "test_misclassified.png", "Misclassified test leaves (true label above, prediction below).")
    top = NUM["top_features"]
    H.add_text(doc, f"Fig. {{fig:features_pca}} (left) ranks the features by their ANOVA F-value on the training set, which is the "
               f"score the selector uses. Ten of the fifteen best are texture features (LBP bins 0, 1, 2, 4, 6, 7 and 9, and GLCM "
               f"correlation and contrast) and five are colour features (a saturation-histogram bin, the spread of *a*, and the mean "
               f"lightness, value and saturation of the leaf); no shape feature appears. The leading LBP bin has F = {top[0]['F']:.0f}. "
               f"These are one-feature-at-a-time scores and the ten LBP bins sum to one, so they are redundant with each other; the ranking "
               f"says what separates the classes, not which features the SVM needs. On the right, the first two principal "
               f"components of the 90 features (fitted on training images, test images drawn) explain only "
               f"{100 * NUM['pca_variance'][0]:.0f}% and {100 * NUM['pca_variance'][1]:.0f}% of the variance. Healthy leaves form a compact group, "
               f"bacterial spot sits to the right, and early blight, late blight, leaf mold and Septoria overlap in the middle, the same "
               f"pairs that the confusion matrix confuses.")
    with Wide(doc):
        fig(doc, "features_pca", "fig4_features_pca.png", "Left: the fifteen features with the highest F-value. Right: test images in "
            "the first two PCA axes of the 90 features.", 5.6)
    H.add_heading2(doc, "4.6 One leaf per class")
    g = NUM["gallery"]
    H.add_text(doc, f"Fig. {{fig:gallery}} runs one test leaf of each class through the whole system. The six leaves were drawn "
               f"at random from the test split with a fixed seed, not chosen for looks, and Table {{tab:gallery_t}} lists what the "
               f"system returned. Five of the six labels are right; the Septoria leaf is called late blight with 87% confidence. "
               f"The healthy leaf receives {g[0]['severity']}% (\u201cMild\u201d) because a dark, folded part of the leaf is "
               f"counted as lesion, the shading false alarm discussed in Chapter 6. The bacterial-spot leaf has {g[1]['lesions']} "
               f"small regions that together cover only {g[1]['severity']}% of the leaf. The Canny row shows that edges follow "
               f"the veins and the leaf outline everywhere; the lesion boundaries are not distinguishable from them, which "
               f"is why Canny is not used to edit the mask.")
    with Wide(doc):
        fig(doc, "gallery", "fig4_class_gallery.png", "One random test leaf per class. Columns: true class; rows: photo, diseased "
            "pixels, Canny edges and the regions measured. Predicted label, confidence and severity are below each column.", 5.9)
        table(doc, "gallery_t", ["True class", "Predicted", "Confidence", "Severity", "Level", "Regions"],
              [[CLASS[x["true"]], CLASS[x["predicted"]], f"{x['confidence']:.0%}", f"{x['severity']}%", x["level"], x["lesions"]]
               for x in g],
              [1.4, 1.4, 0.8, 0.8, 1.2, 0.7], "Output of the system for the leaves of Fig. {fig:gallery}",
              align=["L", "L", "C", "C", "L", "C"])
    H.add_heading2(doc, "4.7 Severity and confidence")
    sm, p90, cf = NUM["severity_median"], NUM["severity_p90"], NUM["confidence"]
    H.add_text(doc, f"Fig. {{fig:sevconf}} (left) shows the measured severity of all 757 test images by their true class. The medians "
               f"rise from {sm['Healthy']:.1f}% (healthy) and {sm['Bacterial_spot']:.1f}% (bacterial spot) and {sm['Leaf_Mold']:.1f}% (leaf mold) to "
               f"{sm['Septoria_leaf_spot']:.1f}% (Septoria), {sm['Early_blight']:.1f}% (early blight) and {sm['Late_blight']:.1f}% (late blight), which "
               f"is the order one expects, since blights cover large areas. The spread is wide, however: the 90th percentile "
               f"of the healthy leaves is {p90['Healthy']:.1f}% and {100 * NUM['healthy_above_5']:.0f}% of healthy test leaves exceed the 5% "
               f"limit of \u201cTrace\u201d. A severity of 6\u201310% therefore does not prove disease, and bacterial spot and leaf "
               f"mold, whose symptoms are small or pale, are mostly measured as trace (medians {sm['Bacterial_spot']:.1f}% and {sm['Leaf_Mold']:.1f}%). Severity is a rough indicator, not a diagnosis.")
    H.add_text(doc, f"Fig. {{fig:sevconf}} (right) shows the confidence shown in the interface. Of the {757 - cf['n_wrong']} correct labels, "
               f"{100 * cf['share_correct_above_0.8']:.0f}% have a confidence above 0.8, but so do most of the {cf['n_wrong']} wrong ones "
               f"(median confidence {cf['median_wrong']:.2f} against {cf['median_correct']:.2f} for the right ones). Among the "
               f"{100 * cf['share_conf_above_0.8']:.0f}% of images above 0.8 the accuracy is {100 * cf['accuracy_when_conf_above_0.8']:.1f}%, "
               f"only {100 * (cf['accuracy_when_conf_above_0.8'] - 0.9075):.1f} points above the overall figure, and only "
               f"{cf['n_conf_below_0.6']} images fall below 0.6, so the low-confidence warning of the interface rarely fires. "
               f"The confidence is useful for ranking, but it is not a reliable guard against a wrong label. These plots reuse the "
               f"test predictions after the model was fixed and tuned nothing.")
    with Wide(doc):
        fig(doc, "sevconf", "fig4_severity_confidence.png", "Left: measured severity by true class (dotted lines mark 5, 15 and "
            "35%). Right: confidence of correct and wrong test predictions.", 5.6)
    H.add_heading2(doc, "4.8 Graphical interface")
    H.add_text(doc, "Fig. {fig:gui} shows the interface on a test leaf, and Table {tab:severity} lists its output for four test leaves, one per "
               "class. Fig. {fig:gui_err} shows the messages for an unreadable file and for a photograph without a leaf. "
               "Severity has no reference value for these leaves, so Table {tab:severity} demonstrates the output and does not "
               "validate it. A known weak case is the early-blight leaf used in Chapter 5 "
               "(file RS_Erly.B 7852), whose visible spots are measured at only 3.2%, i.e. “Trace / healthy”.")
    with Wide(doc):
        fig(doc, "gui", "fig4_gui_result.png", "Graphical interface after analysing an early-blight test leaf: label, "
            "confidence, severity ruler and the six processing stages.", 4.5)
    table(doc, "severity", ["Test leaf", "Predicted", "Conf.", "Severity", "Level"],
                [["Healthy", "Healthy", "90%", "0.2%", "Trace"],
                 ["Early blight", "Early blight", "80%", "11.2%", "Mild"],
                 ["Late blight", "Late blight", "82%", "7.5%", "Mild"],
                 ["Septoria leaf spot", "Septoria leaf spot", "88%", "16.2%", "Moderate"]],
                [0.95, 1.0, 0.33, 0.5, 0.57], "Interface output for four test-set leaves", align=["L", "L", "C", "C", "L"])
    fig(doc, "gui_err", "fig4_gui_errors.png", "Messages for a text file renamed as an image (left) and a "
        "photograph without a leaf (right).", COLUMN_WIDTH - 0.45)


def chapter5(doc):
    H.add_chapter(doc, "CHAPTER 5: Lab-1 to Lab-5 Functions Utilized")
    H.add_text(doc, "This chapter covers the library functions of the laboratory topics that the project uses: colour spaces, "
               "resizing, smoothing and other filters, contrast enhancement, thresholding, morphology, gradients and edge detection, "
               "contours, clustering and texture. For each group it gives the parameters chosen, how the function works, and a "
               "snapshot of its effect on one real photograph (an early-blight leaf, 256\u00d7256 pixels, whose lesions are "
               "measured at 3.2%). Several sections also compare the chosen function with its alternatives on the same photo. "
               "**Alternatives marked \u201cnot used\u201d were tried only on the single pictures shown, not inside the complete system, so "
               "the report draws no conclusion about their effect on accuracy.** Table {tab:functions} summarises the functions "
               "of the final system.")
    with Wide(doc):
        table(doc, "functions", ["Function", "Library", "Parameters used", "Purpose in this project"],
                    [["cvtColor, split, merge", "OpenCV", "BGR \u2194 HSV, Lab, grey", "Hue band, chroma distances, L for CLAHE"],
                     ["resize", "OpenCV", "INTER_AREA, longest side \u2264 256", "Same pixel scale for every photo"],
                     ["GaussianBlur", "OpenCV", "5\u00d75 kernel, \u03c3 from kernel", "Remove sensor noise before measuring"],
                     ["createCLAHE", "OpenCV", "clip 2.0, 8\u00d78 tiles, on L only", "Local contrast without changing colour"],
                     ["threshold", "OpenCV", "THRESH_OTSU on rescaled 8-bit map", "Automatic leaf/background cut"],
                     ["apply_hysteresis_threshold", "scikit-image", "low = 0.7 \u00d7 Otsu, high = Otsu", "Keep pale leaf tissue, drop noise"],
                     ["getStructuringElement, morphologyEx, erode", "OpenCV", "ellipse 3\u00d73 (open, close), 7\u00d77 (rim)", "Clean masks, trim blurred leaf rim"],
                     ["connectedComponentsWithStats", "OpenCV", "8-connectivity", "Largest leaf blob, remove blobs < 8 px"],
                     ["kmeans", "OpenCV", "k = 3, PP centres, 3 attempts", "Lesion method B (rejected)"],
                     ["Canny", "OpenCV", "auto thresholds, \u03c3 = 0.33", "Edge picture and edge-density feature"],
                     ["findContours, contourArea, arcLength, convexHull, boundingRect", "OpenCV", "RETR_EXTERNAL", "Area, perimeter, circularity, solidity, aspect"],
                     ["moments, HuMoments", "OpenCV", "binary lesion mask, log-scaled", "Seven shape-invariant features"],
                     ["graycomatrix, graycoprops", "scikit-image", "32 levels, d = 1, 3, four angles", "Ten GLCM texture features"],
                     ["local_binary_pattern", "scikit-image", "P = 8, R = 1, uniform", "Ten LBP texture features"],
                     ["drawContours, rectangle, addWeighted", "OpenCV", "overlays", "Stage pictures and GUI"]],
                    [2.3, 0.85, 2.05, 1.8], "Library functions used in the project",
                    align=["L", "L", "L", "L"])

    # 5.1 -----------------------------------------------------------------------------------
    H.add_heading2(doc, "5.1 Colour-space conversion")
    H.add_text(doc, "`cvtColor` converts between colour spaces. HSV separates the colour itself (hue, stored by OpenCV in 0\u2013179 "
               "for 8-bit images) from its purity (saturation) and brightness (value), which suits the green-hue band and "
               "the hue histograms. Lab separates lightness *L* from two colour axes, *a* (green\u2013red) and *b* "
               "(blue\u2013yellow), and Euclidean distance in (*a*, *b*) approximates how different two colours look. That is "
               "why every colour distance in the project, background versus leaf and healthy versus diseased, is measured "
               "on (*a*, *b*) and not on RGB (Fig. {fig:colour}). In Fig. {fig:colour} the hue channel is nearly uniform over the green tissue, whereas *a* "
               "brightens at the brown lesions, and the cast shadow, which is dark in *L* and *V*, almost vanishes in *a* and *b*.")
    with Wide(doc):
        fig(doc, "colour", "fig5_colour_spaces.png", "One photo in the channels used by the project.", FULL_WIDTH)

    # 5.2 -----------------------------------------------------------------------------------
    H.add_heading2(doc, "5.2 Resizing")
    H.add_text(doc, "`resize` changes the pixel grid. The project only ever shrinks, and only photographs larger than 256 pixels, "
               "with `INTER_AREA`, which averages the source pixels that fall into each output pixel and so avoids the jagged "
               "aliasing that nearest-neighbour sampling produces; the PlantVillage images are already 256\u00d7256 and pass "
               "through unchanged. For enlarging, which the project does not do, the choice matters more: Fig. {fig:resize} shows a "
               "64\u00d764 version of the photo enlarged four times. Nearest neighbour keeps hard blocks, bilinear averages the four "
               "nearest pixels and is smooth but soft, and bicubic and Lanczos use 16 and 64 neighbours and keep slightly sharper edges.")
    with Wide(doc):
        fig(doc, "resize", "fig5_resize.png", "A 100\u00d7100 patch of a 64\u00d764 photo enlarged to 256\u00d7256 with four "
            "interpolation methods.", 5.0)

    # 5.3 -----------------------------------------------------------------------------------
    H.add_heading2(doc, "5.3 Smoothing filters")
    H.add_text(doc, "`GaussianBlur(image, (5, 5), 0)` replaces each pixel by a weighted average of its neighbours, with weights "
               "given by Eq. (1). Passing a standard deviation of 0 lets OpenCV derive it from the kernel size (about 1.1 "
               "for 5\u00d75), and the filter is applied as two one-dimensional passes, which is fast. A larger kernel removes "
               "more noise but also erases small lesions, so the kernel is kept small (Fig. {fig:gauss} shows the three sizes "
               "for comparison): the 9\u00d79 result has lost the fine texture and the faint spots, while 3\u00d73 keeps most of the "
               "noise.")
    with Wide(doc):
        fig(doc, "gauss", "fig5_gaussian.png", "Effect of the Gaussian kernel size on an 80\u00d780 pixel patch.", 5.2)
    f = {(x["noise"], x["filter"]): x for x in NUM["filters"]}
    gn, sp = "Gaussian noise (sigma 15)", "salt-and-pepper noise (4%)"
    names = ["none", "box 5x5 (blur)", "Gaussian 5x5", "median 5", "bilateral (9, 50, 50)"]
    H.add_text(doc, "OpenCV offers other smoothing filters, which were compared on the same photo after adding synthetic noise "
               "(Fig. {fig:filters}). `blur` is a plain box average. `medianBlur` replaces each pixel by the median of its "
               "neighbourhood, which discards extreme values. `bilateralFilter` weights neighbours by both distance and colour "
               "similarity, so it averages inside a region but not across a strong edge. The quality measure is the peak "
               "signal-to-noise ratio against the unfiltered photo,")
    H.add_equation(doc, "PSNR = 10 log_{10} [ 255^{2} / MSE ]   [dB]")
    H.add_text(doc, "with MSE the mean squared difference of the pixel values; higher is closer to the photo. The photo is "
               "itself not noise-free, so the reference is only a yardstick. Table {tab:filters_t} shows three results. For Gaussian "
               f"noise only the bilateral filter improves on the noisy image ({f[(gn, names[4])]['psnr']} against {f[(gn, names[0])]['psnr']} dB): "
               f"the box, Gaussian and median filters remove the noise but also the real leaf texture, so they score lower "
               f"({f[(gn, names[1])]['psnr']}, {f[(gn, names[2])]['psnr']} and {f[(gn, names[3])]['psnr']} dB). For salt-and-pepper "
               f"noise the median and Gaussian filters both reach {f[(sp, names[3])]['psnr']} dB, while the bilateral filter does "
               f"not help ({f[(sp, names[4])]['psnr']} dB), because an extreme impulse has no similar neighbours to be averaged with. "
               f"The bilateral filter is also far slower ({f[(gn, names[4])]['ms']} ms against {f[(gn, names[2])]['ms']} ms for the Gaussian "
               f"on this computer). The Gaussian is used because PlantVillage photographs contain little sensor noise, it is "
               f"fast and has one parameter, and its blur also suppresses JPEG speckle before colour thresholding. The bilateral "
               f"filter is the more interesting alternative, but it was not tested inside the pipeline.")
    with Wide(doc):
        fig(doc, "filters", "fig5_filters.png", "Smoothing filters on a 72\u00d772 patch with added Gaussian noise (top) and "
            "salt-and-pepper noise (bottom); PSNR of the whole filtered image is in each title.", 5.2)
        table(doc, "filters_t", ["Filter", "Function", "PSNR, Gaussian noise (dB)", "PSNR, salt and pepper (dB)", "Time (ms)"],
              [["Noisy image", "\u2013", f[(gn, names[0])]["psnr"], f[(sp, names[0])]["psnr"], "\u2013"],
               ["Box average 5\u00d75", "blur", f[(gn, names[1])]["psnr"], f[(sp, names[1])]["psnr"], f[(gn, names[1])]["ms"]],
               ["Gaussian 5\u00d75 (used)", "GaussianBlur", f[(gn, names[2])]["psnr"], f[(sp, names[2])]["psnr"], f[(gn, names[2])]["ms"]],
               ["Median, size 5", "medianBlur", f[(gn, names[3])]["psnr"], f[(sp, names[3])]["psnr"], f[(gn, names[3])]["ms"]],
               ["Bilateral (9, 50, 50)", "bilateralFilter", f[(gn, names[4])]["psnr"], f[(sp, names[4])]["psnr"], f[(gn, names[4])]["ms"]]],
              [1.7, 1.2, 1.3, 1.3, 0.7], "Smoothing filters on a 256\u00d7256 colour image with added noise (time measured on the "
              "build computer, mean of 50 runs)", align=["L", "L", "C", "C", "C"])

    # 5.4 -----------------------------------------------------------------------------------
    H.add_heading2(doc, "5.4 Convolution with a custom kernel and sharpening")
    H.add_text(doc, "Every linear filter above is a convolution, and `filter2D` applies any kernel the user writes. A 3\u00d73 kernel "
               "with a centre of 5 and \u22121 in the four neighbours adds the image's high-pass detail (its negative Laplacian) to itself and so sharpens it; "
               "unsharp masking subtracts a blurred copy with `GaussianBlur` and `addWeighted` (here 1.8 \u00d7 image \u2212 0.8 "
               "\u00d7 blur, \u03c3 = 2). Fig. {fig:sharpen} shows that both make the veins and spots crisper but also "
               "amplify the leaf's fine grain. Because the project measures colour distances and areas, noise matters more "
               "than crispness, so **no sharpening is used**; the figure is included to show the technique.")
    with Wide(doc):
        fig(doc, "sharpen", "fig5_sharpen.png", "Sharpening a blurred patch: a 3\u00d73 kernel with filter2D and an unsharp mask "
            "(not used in the pipeline).", 4.4)

    # 5.5 -----------------------------------------------------------------------------------
    eq = NUM["equalise"]
    H.add_heading2(doc, "5.5 Contrast enhancement (histogram equalisation and CLAHE)")
    H.add_text(doc, "`createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))` divides the image into 8\u00d78 tiles, builds a histogram "
               "of each tile, clips every bin at the limit so that noise is not amplified, redistributes the clipped "
               "counts, equalises each tile and blends neighbouring tiles by bilinear interpolation to hide the seams. "
               "It is applied to *L* only, so colours are untouched. In Fig. {fig:clahe} the lightness histogram of the leaf "
               "spreads over a wider range after CLAHE, and the veins and spots become easier to see.")
    with Wide(doc):
        fig(doc, "clahe", "fig5_clahe.png", "Lightness before and after CLAHE, with the histogram of the leaf pixels.", FULL_WIDTH)
    H.add_text(doc, "`equalizeHist` equalises the histogram of the whole image at once. Fig. {fig:equalise} compares it with CLAHE at "
               "four clip limits, and Table {tab:equalise_t} measures the spread (standard deviation) of *L* over the leaf pixels, which "
               f"is {eq[0]['std']} before any enhancement. Global equalisation raises it to {eq[1]['std']} but also lifts the "
               f"grey background and the shadow, so the leaf and its surroundings both change. CLAHE with a clip limit of 1, 2, 4 "
               f"and 8 gives {eq[2]['std']}, {eq[3]['std']}, {eq[4]['std']} and {eq[5]['std']}: the higher the limit, the stronger the "
               f"contrast, and at 8 the grain of the background is amplified and {eq[5]['clipped']}% of the leaf pixels are pushed "
               f"to the ends of the scale. The clip limit of 2.0 is a moderate choice; it was not tuned on accuracy, "
               f"so it is a default, not an optimum.")
    with Wide(doc):
        fig(doc, "equalise", "fig5_equalise.png", "Global equalisation and CLAHE with four clip limits, applied to L only; the "
            "spread of L on the leaf is in each title.", FULL_WIDTH)
        table(doc, "equalise_t", ["Method", "Spread of L on the leaf", "Leaf pixels at 0\u20132 or 253\u2013255 (%)"],
              [[x["method"], x["std"], x["clipped"]] for x in eq], [2.4, 1.6, 2.3],
              "Effect of the contrast method on the lightness of the leaf pixels", align=["L", "C", "C"])

    # 5.6 -----------------------------------------------------------------------------------
    tc = {x["method"]: x["agreement"] for x in NUM["threshold_compare"]}
    H.add_heading2(doc, "5.6 Thresholding")
    H.add_text(doc, "`threshold` with `THRESH_OTSU` chooses the cut that best separates the histogram into two groups, "
               "Eq. (3); it needs an 8-bit image, so the colour-distance map is first rescaled to 0\u2013255 and the "
               "threshold is scaled back. For this photo Otsu selected a distance of 16.6. "
               "`apply_hysteresis_threshold` then takes every pixel above 16.6 as sure leaf, adds pixels above 70% of it "
               "(11.6) only if they connect to sure leaf, and so recovers pale tissue at the leaf edge without "
               "admitting isolated background noise (Fig. {fig:thresh}). Opening and keeping the largest component finish the mask.")
    with Wide(doc):
        fig(doc, "thresh", "fig5_threshold.png", "Colour distance to the background, Otsu and hysteresis thresholds and "
            "the resulting leaf mask.", FULL_WIDTH)
    H.add_text(doc, "The more familiar way to threshold is on the grey image, and Fig. {fig:threshcmp} shows why it was not used. A fixed "
               "cut at 127, Otsu on the grey image and two adaptive thresholds (`adaptiveThreshold`, mean and Gaussian "
               "weighted, 31\u00d731 neighbourhood) all fail on this photo, because the leaf, its cast shadow and the background have "
               "overlapping grey levels: the shadow joins the leaf or the veins split it. Their overlap with the mask that "
               f"is used (intersection over union) is only {tc['fixed 127 on grey']:.2f}, {tc['Otsu on grey (93)']:.2f}, "
               f"{tc['adaptive mean, 31x31']:.2f} and {tc['adaptive Gaussian, 31x31']:.2f}. The used mask is an automatic result and not "
               "a ground truth, but it agrees with PlantVillage's own masks at 0.924 on average (Chapter 4), so these numbers "
               "show how far the grey-level methods are from a good leaf mask. Colour distance in (*a*, *b*) ignores lightness "
               "and therefore ignores the shadow.")
    with Wide(doc):
        fig(doc, "threshcmp", "fig5_threshold_compare.png", "Grey-level thresholding methods compared with the chroma-distance leaf "
            "mask that the project uses (not used: the four on the left).", 4.6)

    # 5.7 -----------------------------------------------------------------------------------
    mt = {(x["shape"], x["size"]): x for x in NUM["morph_table"]}
    H.add_heading2(doc, "5.7 Morphology and connected components")
    H.add_text(doc, "Erosion shrinks a white region and dilation grows it. Opening is erosion followed by dilation and "
               "removes objects smaller than the structuring element; closing is dilation followed by erosion and "
               "fills small gaps:")
    H.add_equation(doc, "*A* \u2218 *B* = (*A* \u2296 *B*) \u2295 *B*,   *A* \u2022 *B* = (*A* \u2295 *B*) \u2296 *B*")
    H.add_text(doc, "`morphologyEx` with `MORPH_OPEN` and `MORPH_CLOSE` applies them with a 3\u00d73 elliptical element from "
               "`getStructuringElement`, and `erode` with a 7\u00d77 element trims the blurred leaf rim. "
               "`connectedComponentsWithStats` labels each white blob with 8-connectivity and returns its area, which is "
               "used to keep the largest leaf blob and to delete lesion blobs under 8 pixels. On this leaf the raw lesion "
               "mask holds 784 pixels in 49 blobs; opening leaves 648 pixels in 15 blobs, closing changes nothing here, and "
               "the area filter leaves 633 pixels in 12 blobs (Fig. {fig:morph}).", style="R Body Flush")
    with Wide(doc):
        fig(doc, "morph", "fig5_morphology.png", "Morphological clean-up of the lesion mask, and the rim removed by erosion.", FULL_WIDTH)
    H.add_text(doc, "`morphologyEx` offers more operations than the two used (Fig. {fig:morphops}). The morphological gradient "
               "(dilation minus erosion) outlines each blob. The top-hat (image minus its opening) keeps what opening removed, "
               "the small bright specks, and the black-hat (closing minus image) keeps what closing filled; the black-hat of this "
               "mask is almost empty, which agrees with the observation that closing changes nothing here. The size and "
               "shape of the element decide how much is lost (Table {tab:morph_t}, which applies opening, closing and the "
               f"8-pixel filter to the same raw mask of {NUM['raw_pixels']} pixels). With a 3\u00d73 element the "
               f"result keeps {mt[('ellipse', 3)]['pixels']} pixels in {mt[('ellipse', 3)]['blobs']} blobs; with 5\u00d75 it keeps "
               f"{mt[('ellipse', 5)]['pixels']} pixels in {mt[('ellipse', 5)]['blobs']} blobs and with 7\u00d77 only "
               f"{mt[('ellipse', 7)]['pixels']} in {mt[('ellipse', 7)]['blobs']}. A rectangular 7\u00d77 element removes every lesion, "
               f"because the lesions of this leaf are thin. OpenCV's 3\u00d73 ellipse is the same plus-shaped element as the 3\u00d73 "
               f"cross, so the two give identical results. The 3\u00d73 size is the one that removes speckle without deleting real, "
               f"small lesions.")
    with Wide(doc):
        fig(doc, "morphops", "fig5_morph_ops.png", "Other morphological operations on the raw lesion mask.", 4.6)
        table(doc, "morph_t", ["Element", "3\u00d73: pixels / blobs", "5\u00d75: pixels / blobs", "7\u00d77: pixels / blobs"],
              [[name.capitalize()] + [f"{mt[(shape, n)]['pixels']} / {mt[(shape, n)]['blobs']}" for n in (3, 5, 7)]
               for name, shape in (("rectangle", "rectangle"), ("ellipse (used, 3\u00d73)", "ellipse"), ("cross", "cross"))],
              [2.0, 1.4, 1.4, 1.4], "Lesion pixels and blobs left after opening, closing and the 8-pixel filter, by structuring element",
              align=["L", "C", "C", "C"])

    # 5.8 -----------------------------------------------------------------------------------
    cs = {x["setting"]: x for x in NUM["canny_sweep"]}
    H.add_heading2(doc, "5.8 Gradients and edge detection")
    H.add_text(doc, "An edge is a place where brightness changes quickly, so edge detectors estimate the gradient. `Sobel` convolves "
               "the grey image with two 3\u00d73 kernels, one for the horizontal and one for the vertical change, and the gradient "
               "magnitude and direction follow from the two results:")
    H.add_equation(doc, "|\u2207*I*| = \u221A( *G*_{x}^{2} + *G*_{y}^{2} ),   *\u03b8* = atan2( *G*_{y}, *G*_{x} )")
    H.add_text(doc, "`Scharr` uses a kernel with better rotational symmetry, and `Laplacian` takes the second derivative, which "
               "responds to both edges and noise and is therefore applied after a Gaussian blur (Laplacian of Gaussian). Fig. {fig:derivs} "
               "shows all of them on the CLAHE grey image. The horizontal and vertical Sobel images each show only one direction of "
               "veins; their magnitude shows the veins, the leaf outline and the shadow edge together. None of these produces a "
               "thin, closed edge: that is the job of Canny.", style="R Body Flush")
    with Wide(doc):
        fig(doc, "derivs", "fig5_derivatives.png", "Gradient and Laplacian filters on the CLAHE grey image (absolute values, "
            "scaled for display; not used in the pipeline).", FULL_WIDTH)
    H.add_text(doc, "`Canny` follows the classical five steps [4]: Gaussian smoothing, gradient magnitude and direction (Sobel "
               "filters), non-maximum suppression to thin edges to one pixel, a double threshold, and edge tracking by "
               "hysteresis. The two thresholds come from Eq. (6) with the leaf's median grey level, which was 121 here (low "
               "81, high 160), and edges outside the leaf core are discarded (Fig. {fig:canny}). The result contains 5,813 edge pixels, an "
               "edge density of 0.29 of the leaf area, because the veins and the fine surface texture produce edges as "
               "well as the lesions. This is the reason Canny is kept as an inspection view and one feature, and is not used to correct "
               "the lesion mask: edges cannot tell a vein from a lesion boundary.")
    with Wide(doc):
        fig(doc, "canny", "fig5_canny.png", "Canny edges over the whole image and as used (leaf core only).", 5.0)
    H.add_text(doc, "The thresholds matter less than one might expect (Fig. {fig:cannysweep}). Changing *\u03c3* in the automatic rule from 0.1 "
               f"to 0.66 moves the edge density only from {cs['auto, sigma 0.1']['density']:.2f} to {cs['auto, sigma 0.66']['density']:.2f}, "
               f"because the strong vein edges pass any of these settings. Fixed thresholds show the two extremes: 40/80 "
               f"gives {cs['fixed 40/80']['density']:.2f} and lets in the fine grain of the leaf and the background, while 160/255 "
               f"gives {cs['fixed 160/255']['density']:.2f} and loses the faint veins. No setting isolates the lesions, which is "
               f"consistent with the decision above. A median-based rule has the practical advantage that a dark or bright photo "
               f"is handled without a new setting.")
    with Wide(doc):
        fig(doc, "cannysweep", "fig5_canny_sweep.png", "Canny with automatic thresholds for three values of sigma (top) and with "
            "three fixed pairs of thresholds (bottom).", 4.4)

    # 5.9 -----------------------------------------------------------------------------------
    H.add_heading2(doc, "5.9 Contours and shape measures")
    H.add_text(doc, "`findContours` with `RETR_EXTERNAL` traces the outer border of every white blob (a border-following "
               "algorithm). `contourArea` and `arcLength` give *A* and *P* for Eq. (5); `convexHull` gives the hull area for "
               "solidity; `boundingRect` gives the box for the aspect ratio. `moments` computes the raw and central moments "
               "of the whole lesion mask and `HuMoments` combines the normalised moments into seven values that do not "
               "change with position, size or rotation. In Fig. {fig:contours} the largest lesion has area 118 px, perimeter 77.4 px, "
               "circularity 0.25, solidity 0.65 and aspect ratio 3.8: an elongated, ragged blob, as the "
               "measurements say.")
    with Wide(doc):
        fig(doc, "contours", "fig5_contours.png", "Lesion contours and the measurements of the largest lesion.", FULL_WIDTH)

    # 5.10 ----------------------------------------------------------------------------------
    H.add_heading2(doc, "5.10 Clustering and labelled components")
    H.add_text(doc, "`kmeans` repeats two steps, assigning every (*a*, *b*) colour to the nearest of *k* = 3 centres and moving "
               "each centre to the mean of its colours, until 30 iterations or a movement below 0.5. "
               "`KMEANS_PP_CENTERS` chooses well-spread starting centres and three attempts keep the most compact result. "
               "It is lesion method B. Because it must always return three groups, it splits a healthy leaf into darker and "
               "lighter green and calls one of them diseased (Fig. {fig:kmeans}), which is the reason it was rejected.")
    with Wide(doc):
        fig(doc, "kmeans", "fig5_components_kmeans.png", "Labelled lesion blobs, and K-means clusters of the leaf "
            "colours with the cluster taken as lesion.", 5.0)

    # 5.11 ----------------------------------------------------------------------------------
    H.add_heading2(doc, "5.11 Histograms and texture")
    H.add_text(doc, "The hue and saturation histograms among the features use eight bins computed with NumPy; "
               "`cv2.calcHist` is used in the split-leakage audit, with a 3-D colour histogram. `graycomatrix` counts how "
               "often grey level *i* has a neighbour of level *j* at a given distance and direction, and `graycoprops` "
               "reduces that matrix to numbers such as")
    H.add_equation(doc, "contrast = \u03a3 *P*(*i*, *j*) (*i* \u2212 *j*)^{2}")
    H.add_equation(doc, "homogeneity = \u03a3 *P*(*i*, *j*) / (1 + (*i* \u2212 *j*)^{2})")
    H.add_text(doc, "where *P* is the normalised matrix. A smooth leaf concentrates *P* on the diagonal (Fig. {fig:texture}, middle); "
               "spots spread it out. `local_binary_pattern` compares each pixel with its eight neighbours on a circle of "
               "radius 1 and codes the result; \u201cuniform\u201d patterns (at most two light/dark transitions around the circle) "
               "are counted in ten bins, which gives a rotation-tolerant description of local texture. As Fig. {fig:features_pca} "
               "shows, these texture features are the most class-separating of the 90.", style="R Body Flush")
    with Wide(doc):
        fig(doc, "texture", "fig5_texture.png", "Hue histogram of the leaf, its GLCM, and the LBP histogram.", FULL_WIDTH)


def chapter6(doc):
    H.add_chapter(doc, "CHAPTER 6: CONCLUSION, LIMITATIONS & FUTURE WORK")
    H.add_heading2(doc, "6.1 Conclusion")
    H.add_text(doc, "A transparent classical pipeline can classify six tomato leaf conditions with 90.8% accuracy (macro-F1 "
               "0.909) on 757 test images that were scored once, using 90 hand-designed features and an SVM. The leaf "
               "is segmented well (IoU 0.924). Lesion segmentation is the weaker part: the three colour-based methods "
               "reach Dice of only about 0.4, and the method chosen, which compares each pixel with the leaf's own "
               "healthy green, mainly lowers the severity error rather than raising overlap. Every stage is visible in "
               "the interface, which suits explanation and debugging.")
    table(doc, "summary", ["Quantity", "Result", "Data"],
          [["Classification accuracy / macro-F1", "90.8% / 0.909", "757 test images, scored once"],
           ["Accuracy with random folds / grouped folds", "92.7% / 88.7%", "training set, same SVM"],
           ["Leaf mask, mean IoU", "0.924", "300 training images"],
           ["Lesion mask, Dice (own-green)", "0.407", "12 hold-out annotated leaves"],
           ["Severity error (MAE)", "6.6 points", "12 hold-out annotated leaves"],
           ["Healthy-leaf area flagged as lesion", "0.6%", "2 healthy hold-out leaves"],
           ["Healthy test leaves above 5% severity", f"{100 * NUM['healthy_above_5']:.0f}%", "127 healthy test images"]],
          [1.7, 0.8, 1.4], "Key measured results", align=["L", "C", "L"])
    H.add_heading2(doc, "6.2 What worked and what did not")
    H.add_bullets(doc, [
        "**Worked:** measuring colour distance on (*a*, *b*) instead of RGB or grey level removed the cast shadow from the leaf "
        "mask (Section 5.6), and the grouped split showed that about four points of the apparent accuracy came from "
        "near-duplicate photographs (Section 4.5).",
        "**Worked:** texture features carry most of the class information: ten of the fifteen best features are LBP or "
        "GLCM values (Section 4.5), and colour plus texture alone already reach 0.874 validation macro-F1.",
        "**Did not work as hoped:** the lesion masks. All three colour methods reach a Dice of about 0.4, K-means "
        "invents lesions on healthy leaves, and Canny edges cannot separate lesions from veins.",
        "**Did not work as hoped:** the confidence value. It is almost as high for wrong labels as for right ones, so it "
        "ranks images but does not protect against errors (Section 4.7).",
    ])
    H.add_heading2(doc, "6.3 Limitations")
    H.add_bullets(doc, [
        "**Reference masks.** The 24 lesion annotations were drawn by an AI assistant, not by an expert, and are few; "
        "all segmentation scores carry wide uncertainty.",
        "**Severity** is colour-based and area-based. It misses faint symptoms (a hold-out late-blight leaf: 29.3% "
        "true, 7.4% measured) and can count shadows, veins or natural yellowing, so "
        f"{100 * NUM['healthy_above_5']:.0f}% of healthy test leaves exceed the 5% trace limit; levels are proposals; curled leaves "
        "and the leaf underside are not seen.",
        "**Confidence** is calibrated on training data but still high for most wrong labels, so it should not be "
        "read as the chance that a label is right.",
        "**Data.** PlantVillage is photographed in the laboratory. No field photographs were tested, and accuracy there would "
        "be lower. Near-duplicate leakage is reduced, not removed, so 90.8% is probably still somewhat optimistic.",
        "**Scope.** Six classes only and no “not a tomato leaf” class. The leaf-likeness check rejects noise and "
        "non-plant colours but not every green, yellow or brown object. Two dead dark-brown leaves cannot be analysed.",
        "**Software.** Tested with OpenCV 5.0.0 only.",
    ])
    H.add_heading2(doc, "6.4 Future work")
    H.add_bullets(doc, [
        "Annotate lesions with a plant pathologist and enlarge the set, then retune the lesion colour floor.",
        "Test on field photographs; add augmentation and an “unknown” class.",
        "Suppress veins and shadows before lesion detection and add texture cues for pale lesions.",
        "Calibrate severity levels against a recognised disease scale.",
    ])


REFERENCES = [
    'D. P. Hughes and M. Salathé, “An open access repository of images on plant health to enable the development '
    'of mobile disease diagnostics,” arXiv:1511.08060, 2015.',
    'S. P. Mohanty, D. P. Hughes, and M. Salathé, “Using deep learning for image-based plant disease detection,” '
    '*Front. Plant Sci.*, vol. 7, Art. no. 1419, 2016.',
    'N. Otsu, “A threshold selection method from gray-level histograms,” *IEEE Trans. Syst., Man, Cybern.*, '
    'vol. SMC-9, no. 1, pp. 62–66, Jan. 1979.',
    'J. Canny, “A computational approach to edge detection,” *IEEE Trans. Pattern Anal. Mach. Intell.*, '
    'vol. PAMI-8, no. 6, pp. 679–698, Nov. 1986.',
    'K. Zuiderveld, “Contrast limited adaptive histogram equalization,” in *Graphics Gems IV*, P. S. Heckbert, Ed. '
    'San Diego, CA, USA: Academic Press, 1994, pp. 474–485.',
    'R. M. Haralick, K. Shanmugam, and I. Dinstein, “Textural features for image classification,” *IEEE Trans. Syst., '
    'Man, Cybern.*, vol. SMC-3, no. 6, pp. 610–621, Nov. 1973.',
    'T. Ojala, M. Pietikäinen, and T. Mäenpää, “Multiresolution gray-scale and rotation invariant texture '
    'classification with local binary patterns,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 24, no. 7, '
    'pp. 971–987, Jul. 2002.',
    'M.-K. Hu, “Visual pattern recognition by moment invariants,” *IRE Trans. Inf. Theory*, vol. 8, no. 2, '
    'pp. 179–187, Feb. 1962.',
    'C. Cortes and V. Vapnik, “Support-vector networks,” *Mach. Learn.*, vol. 20, no. 3, pp. 273–297, 1995.',
    'F. Pedregosa *et al.*, “Scikit-learn: Machine learning in Python,” *J. Mach. Learn. Res.*, vol. 12, '
    'pp. 2825–2830, 2011.',
    'G. Bradski, “The OpenCV library,” *Dr. Dobb’s Journal of Software Tools*, 2000.',
    'S. van der Walt *et al.*, “scikit-image: Image processing in Python,” *PeerJ*, vol. 2, p. e453, 2014.',
    'J. C. Platt, “Probabilistic outputs for support vector machines and comparisons to regularized likelihood '
    'methods,” in *Advances in Large Margin Classifiers*, Cambridge, MA, USA: MIT Press, 1999, pp. 61–74.',
    'S. P. Lloyd, “Least squares quantization in PCM,” *IEEE Trans. Inf. Theory*, vol. 28, no. 2, pp. 129–137, Mar. 1982.',
    'L. R. Dice, “Measures of the amount of ecologic association between species,” *Ecology*, vol. 26, no. 3, '
    'pp. 297–302, 1945.',
    'R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed. New York, NY, USA: Pearson, 2018.',
    'J. G. A. Barbedo, “A review on the main challenges in automatic plant disease identification based on '
    'visible range images,” *Biosyst. Eng.*, vol. 144, pp. 52–60, 2016.',
    'J. G. A. Barbedo, “Factors influencing the use of deep learning for plant disease recognition,” '
    '*Biosyst. Eng.*, vol. 172, pp. 84–91, 2018.',
    'C. R. Harris *et al.*, “Array programming with NumPy,” *Nature*, vol. 585, pp. 357–362, 2020.',
    'Pallets Projects, “Flask documentation (3.1.x).” [Online]. Available: https://flask.palletsprojects.com',
]


def chapter7(doc):
    H.add_chapter(doc, "CHAPTER 7: REFERENCES")
    for number, text in enumerate(REFERENCES, start=1):
        paragraph = doc.add_paragraph(style="R Ref")
        paragraph.add_run(f"[{number}]\t")
        paragraph.paragraph_format.tab_stops.add_tab_stop(Inches(0.3))
        H.add_runs(paragraph, text, size=8)


# =============================================================================
def build(toc_pages=None):
    for source, target in (("results/stages/UF.GRC_BS_Lab_Leaf_9194/montage.png", "stages_montage.png"),
                           ("results/confusion_matrix_test.png", "confusion_matrix_test.png"),
                           ("results/test_misclassified.png", "test_misclassified.png"),
                           ("results/lesion_failures_own_green_holdout.png", "lesion_failures_own_green_holdout.png")):
        if (ROOT / source).exists():          # results/stages is regenerable; the copies in figures/ are enough
            shutil.copy(ROOT / source, FIG / target)
        elif not (FIG / target).exists():
            raise FileNotFoundError(f"{target} is missing: run `python -m src.visualize` and `python -m src.evaluate` first")

    doc = Document(str(TEMPLATE))
    H.define_styles(doc)
    fill_front_matter(doc)
    fill_toc(doc, toc_pages)

    # body: new page, margins like an IEEE paper, two columns, page numbers restart at 1
    section = H.new_section(doc, 2, kind=WD_SECTION.NEW_PAGE, restart_numbers=True)
    section.left_margin = section.right_margin = Inches(0.75)
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.9)
    section.footer_distance = Inches(0.4)
    H.add_page_number_footer(section)

    H.Counter.figures = H.Counter.tables = H.Counter.equations = 0
    abstract(doc)
    chapter1(doc)
    chapter2(doc)
    chapter3(doc)
    chapter4(doc)
    chapter5(doc)
    chapter6(doc)
    chapter7(doc)
    doc.save(str(OUTPUT))
    print("saved", OUTPUT, "figures", H.Counter.figures, "tables", H.Counter.tables, "equations", H.Counter.equations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--toc-pages", type=Path, default=None)
    args = parser.parse_args()
    pages = json.loads(args.toc_pages.read_text()) if args.toc_pages and args.toc_pages.exists() else None
    build(pages)
