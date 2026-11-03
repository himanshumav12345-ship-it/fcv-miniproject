#!/usr/bin/env bash
# Download the six tomato classes of PlantVillage (colour + segmented versions) from GitHub
# and build the project's data/raw folder and data/splits/splits.csv from them.
#
# Usage (from the project root):   bash scripts/fetch_dataset.sh [download_folder]
# Default download folder: ../PlantVillage-source   (about 250 MB; it can be deleted afterwards,
# the project keeps its own copy of the 4,800 chosen images in data/raw/)
#
# Needs: git (any recent version) and internet access to github.com.
set -euo pipefail

DEST="${1:-../PlantVillage-source}"
CLASSES=(Tomato___healthy Tomato___Bacterial_spot Tomato___Early_blight
         Tomato___Late_blight Tomato___Leaf_Mold Tomato___Septoria_leaf_spot)

if [ -e "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ]; then
  echo "'$DEST' already exists and is not empty - choose another folder or delete it." >&2
  exit 1
fi

echo "Cloning repository metadata (no image files yet) ..."
git clone --filter=blob:none --no-checkout --depth 1 \
    https://github.com/spMohanty/PlantVillage-Dataset.git "$DEST"

cd "$DEST"
git sparse-checkout init --no-cone
PATTERNS=()
for c in "${CLASSES[@]}"; do
  PATTERNS+=("/raw/color/$c/" "/raw/segmented/$c/")
done
git sparse-checkout set "${PATTERNS[@]}"

echo "Downloading the six tomato class folders ..."
git checkout

cd - > /dev/null
echo "Building data/raw and data/splits/splits.csv ..."
python3 -m src.dataset --source "$DEST/raw/color"

echo
echo "Done. The segmented reference images (only needed by src.evaluate_leaf_mask) are in:"
echo "  $DEST/raw/segmented"
