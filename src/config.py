"""
config.py - every path, seed and tunable number in one place.

Why one file: if a threshold is scattered across modules, the GUI, the training
script and the notebooks can silently disagree about how an image is processed.
Anything you change here changes the whole project consistently.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths (all relative to the project root, so the project can be moved/zipped)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                  # PlantVillage subset, one folder per class
SPLITS_CSV = DATA_DIR / "splits" / "splits.csv"
ANNOTATION_DIR = DATA_DIR / "annotations"   # hand-drawn lesion masks
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# ----------------------------------------------------------------------------
# Dataset subset (see the project plan: six tomato classes)
# ----------------------------------------------------------------------------
# folder name in the original PlantVillage dataset -> short readable class name
CLASS_FOLDERS = {
    "Tomato___healthy": "Healthy",
    "Tomato___Bacterial_spot": "Bacterial_spot",
    "Tomato___Early_blight": "Early_blight",
    "Tomato___Late_blight": "Late_blight",
    "Tomato___Leaf_Mold": "Leaf_Mold",
    "Tomato___Septoria_leaf_spot": "Septoria_leaf_spot",
}
CLASS_NAMES = list(CLASS_FOLDERS.values())  # fixed order => stable class ids

IMAGES_PER_CLASS = 800        # cap so the classes are balanced
RANDOM_SEED = 42              # one seed for sampling, splitting and models

# Split fractions. Test is touched only once, at the very end.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# PlantVillage photographs the SAME leaf several times (different orientations)
# and numbers them consecutively. Neighbouring ids are near-duplicates, so we
# split by *blocks* of consecutive ids, never by single images. A pair of
# neighbours straddles a block border with probability (id gap / block size),
# so a bigger block leaks less; 20 keeps hundreds of groups for balanced splits.
# Measured with src/audit_split.py (share of test images whose nearest same-class
# train image is a very close colour match): random split ~12-15%, block 20 ~5-7%,
# blocks of 50-200 ~4-6%. So 20 gets nearly all of the benefit and bigger blocks
# do not help further; the leftover is mostly look-alike leaves, not twins.
ID_BLOCK_SIZE = 20

# ----------------------------------------------------------------------------
# Image handling and preprocessing
# ----------------------------------------------------------------------------
MAX_IMAGE_SIDE = 256        # PlantVillage is 256x256; larger photos are shrunk to this
MIN_IMAGE_SIDE = 64         # anything smaller has too few pixels to analyse

# Gaussian blur removes sensor noise before we threshold. Bigger = smoother but it
# also blurs small lesions, so this is the first thing to check if tiny spots vanish.
DENOISE_KERNEL = 5
# CLAHE = contrast-limited adaptive histogram equalisation, applied to lightness only.
CLAHE_CLIP_LIMIT = 2.0      # caps contrast gain so noise is not exaggerated
CLAHE_TILE_GRID = 8         # image is cut into 8x8 tiles that are equalised separately

# ----------------------------------------------------------------------------
# Leaf mask (separating the leaf from the background)
# ----------------------------------------------------------------------------
# We assume the outer border of the photo is background and measure its colour.
BACKGROUND_BORDER_PX = 12
# Hysteresis: pixels above the Otsu threshold are "sure leaf"; pixels above
# HYSTERESIS_LOW_FACTOR x Otsu are leaf only if they touch sure-leaf pixels.
# This recovers pale/dead tissue at the leaf edge. 0.7 won a comparison against
# the PlantVillage reference masks (mean IoU 0.915 vs 0.885 without hysteresis).
HYSTERESIS_LOW_FACTOR = 0.7
LEAF_OPEN_KERNEL = 3        # removes background specks the threshold let through
# Sanity limits: a "leaf" smaller/larger than this is almost certainly a failure.
MIN_LEAF_FRACTION = 0.05
MAX_LEAF_FRACTION = 0.97

# ----------------------------------------------------------------------------
# Lesion mask (which leaf pixels look diseased)
# ----------------------------------------------------------------------------
# The outermost pixels of the leaf are a blur of leaf and background, so they look
# "not green" and are wrongly flagged. We shrink the leaf mask by about 3 px before
# looking for lesions. Cost: a very thin lesion exactly on the leaf edge is partly lost.
LEAF_RIM_ERODE_KERNEL = 7

# Method A (baseline from the plan): "anything outside the green hue band is diseased".
# OpenCV stores hue as 0-179, so green is roughly 35-85.
GREEN_HUE_MIN = 35
GREEN_HUE_MAX = 85

# Method B (from the plan): K-means colour clustering inside the leaf.
KMEANS_CLUSTERS = 3

# Method C (our choice): compare every pixel with THIS leaf's own healthy green.
# The healthy reference colour = median colour of the greenest pixels of the leaf.
HEALTHY_REFERENCE_PERCENTILE = 25
# A pixel is a lesion candidate if its colour (Lab a,b) is farther than this from the
# healthy reference. It is an absolute floor so a perfectly healthy leaf is not forced
# to have lesions. A small grid on the 12 dev annotations (floor 8-18, percentile 25/40,
# lightness weight 0-0.75) gave at most +0.02 Dice, which is noise on 12 images, so the
# value was NOT retuned. Adding lightness made healthy-leaf false alarms jump from ~3% to
# ~18% (veins, folds and shading change lightness), which is why only colour is compared.
MIN_LESION_CHROMA_DISTANCE = 14.0

# ----------------------------------------------------------------------------
# Lesion mask clean-up (morphology) and contour analysis
# ----------------------------------------------------------------------------
LESION_OPEN_KERNEL = 3        # opening removes 1-2 pixel speckle noise
LESION_CLOSE_KERNEL = 3       # closing joins a lesion broken into near pieces
MIN_LESION_AREA_PIXELS = 8    # blobs smaller than this are treated as noise. PROVISIONAL.

# Canny edge detector. Thresholds are set from the image itself (see refine.py).
CANNY_SIGMA = 0.33

# ----------------------------------------------------------------------------
# Severity levels: percent of the leaf area covered by lesions
# ----------------------------------------------------------------------------
# These cut-offs are proposed starting points, not an official plant-pathology scale.
SEVERITY_LEVELS = [          # (upper limit in percent, name)
    (5.0, "Trace / healthy"),
    (15.0, "Mild"),
    (35.0, "Moderate"),
    (100.0, "Severe"),
]

# Which lesion method the pipeline uses by default: "hue_rule", "kmeans" or "own_green".
# Evidence (results/lesion_method_comparison_*.csv, results/false_alarm_check_train.csv):
#   * kmeans is rejected: it flags ~1/3 of every healthy leaf (healthy-vs-diseased AUC 0.26).
#   * hue_rule and own_green TIE on Dice (0.406 vs 0.407 on the hold-out annotations).
#   * own_green is kept because it handles naturally yellow leaves much better (hue_rule
#     called 65% of a yellow Septoria leaf diseased, true 13%) and has about half the
#     severity error (6.6 vs 12.1 percentage points). It is NOT a large improvement.
DEFAULT_LESION_METHOD = "own_green"

# ----------------------------------------------------------------------------
# "Does this object look like a leaf?" sanity check (guards against random uploads)
# ----------------------------------------------------------------------------
# Measured on all 4,798 analysable dataset images: the share of leaf pixels with a plant
# colour (hue 5-95, saturation >= 40) was never below 0.34, and the leaf-mask solidity
# (area / convex-hull area) never below 0.42. The limits sit just under those minimums,
# so no real dataset image is rejected, while random noise (solidity 0.27), a blue or red
# object and smooth colour clouds (plant share 0) are. It is a coarse check: a skin-tone
# or any other green/yellow/brown object still passes.
PLANT_HUE_MIN = 5
PLANT_HUE_MAX = 95
PLANT_MIN_SATURATION = 40
MIN_PLANT_COLOUR_FRACTION = 0.30
MIN_LEAF_SOLIDITY = 0.35
