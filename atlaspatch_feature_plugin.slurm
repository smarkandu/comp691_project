#!/bin/bash
#SBATCH --job-name=atlaspatch-features
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/atlaspatch-features-%j.out
#SBATCH --error=logs/atlaspatch-features-%j.err
#SBATCH --mail-user=s_markan@hotmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

mkdir -p logs

###################################
# Usage
###################################
if [ "$#" -lt 4 ]; then
  echo "Usage:"
  echo "  sbatch $0 <WSI_ROOT> <OUTPUT_ROOT> <FEATURES> <FEATURE_PLUGIN> [PATCH_SIZE] [TARGET_MAG] [FEATURE_BATCH]"
  echo "example: sbatch $0 atlaspatch_features_plugin.slurm ./data/Camelyon16/training vit_b_16_finetuned p2q1.py 256 20"
  exit 1
fi

###################################
# Required args
###################################
WSI_ROOT="$SLURM_SUBMIT_DIR/$1"
OUTPUT_ROOT="$SLURM_SUBMIT_DIR/$2"
FEATURES="$3"
FEATURE_PLUGIN="$SLURM_SUBMIT_DIR/$4"

###################################
# Optional args with defaults
###################################
PATCH_SIZE="${5:-256}"
TARGET_MAG="${6:-20}"
FEATURE_BATCH="${7:-64}"

###################################
# Fixed/default settings
###################################
FEATURE_DEVICE="cuda"
DEVICE="cuda"
FEATURE_PRECISION="float32"
FEATURE_WORKERS="${SLURM_CPUS_PER_TASK:-8}"

###################################
# Environment activation
###################################
cd "$SLURM_SUBMIT_DIR"

# Activate environment
source "/home/smarkan/scratch/comp691_pathology/A2/atlas_env/bin/activate"

###################################
# Diagnostics
###################################
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "Submit dir: $SLURM_SUBMIT_DIR"

echo "WSI_ROOT: $WSI_ROOT"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "FEATURES: $FEATURES"
echo "FEATURE_PLUGIN: $FEATURE_PLUGIN"

echo "PATCH_SIZE: $PATCH_SIZE"
echo "TARGET_MAG: $TARGET_MAG"
echo "FEATURE_BATCH: $FEATURE_BATCH"
echo "FEATURE_WORKERS: $FEATURE_WORKERS"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found."
  exit 1
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: GPU not accessible."
  exit 1
fi

###################################
# Run AtlasPatch
###################################
atlaspatch process "$WSI_ROOT" \
  --output "$OUTPUT_ROOT" \
  --patch-size "$PATCH_SIZE" \
  --target-mag "$TARGET_MAG" \
  --feature-plugin "$FEATURE_PLUGIN" \
  --feature-extractors "$FEATURES" \
  --feature-device "$FEATURE_DEVICE" \
  --feature-batch-size "$FEATURE_BATCH" \
  --feature-num-workers "$FEATURE_WORKERS" \
  --feature-precision "$FEATURE_PRECISION" \
  --device "$DEVICE" \
  --skip-existing \
  --recursive