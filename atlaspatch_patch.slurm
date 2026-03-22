#!/bin/bash
#SBATCH --job-name=atlaspatch-patch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/atlaspatch-patch-%j.out
#SBATCH --error=logs/atlaspatch-patch-%j.err
#SBATCH --mail-user=s_markan@hotmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
mkdir -p logs

###################################
# Usage check
###################################
if [ "$#" -lt 2 ]; then
  echo "Usage:"
  echo "sbatch atlaspatch_patch.slurm.sh <WSI_ROOT> <OUTPUT_ROOT> [PATCH_SIZE] [TARGET_MAG] [SEG_BATCH]"
  exit 1
fi

###################################
# Required parameters
###################################
WSI_ROOT="$1"
OUTPUT_ROOT="$2"

###################################
# Optional parameters (defaults)
###################################
PATCH_SIZE="${3:-256}"
TARGET_MAG="${4:-20}"
SEG_BATCH="${5:-32}"

###################################
# Environment
###################################
cd "$SLURM_SUBMIT_DIR"

# Activate environment
source "$SLURM_SUBMIT_DIR/atlas_env/bin/activate"

###################################
# Derived settings
###################################
PATCH_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
MAX_OPEN_SLIDES=200

###################################
# Diagnostics
###################################
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "WSI_ROOT: $WSI_ROOT"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "PATCH_SIZE: $PATCH_SIZE"
echo "TARGET_MAG: $TARGET_MAG"
echo "SEG_BATCH: $SEG_BATCH"
echo "PATCH_WORKERS: $PATCH_WORKERS"

###################################
# Run AtlasPatch
###################################
atlaspatch segment-and-get-coords "$WSI_ROOT" \
    --patch-size "$PATCH_SIZE" \
    --target-mag "$TARGET_MAG" \
    --output "$OUTPUT_ROOT" \
    --tissue-thresh 0 \
    --fast-mode \
    --recursive \
    --seg-batch-size "$SEG_BATCH" \
    --patch-workers "$PATCH_WORKERS" \
    --max-open-slides "$MAX_OPEN_SLIDES"