#!/usr/bin/env bash
set -euo pipefail

# Run the full Tartan test pipeline:
#   1. Extract random multi-scene Tartan samples and simulate sparse lidar depth.
#   2. Run Marigold-DC completion and evaluate MAE/RMSE.
#
# Example:
#   bash script/run_tartan_pipeline.sh
#
# Common optional parameters can be overridden with environment variables:
#
# Dataset/extraction:
#   TARTAN_ROOT                 default: tartan
#   DATASET_DIR                 default: data/tartan_test/dataset
#   SCENE                       default: empty, sample all scenes
#   DIFFICULTY                  default: empty, sample Easy and Hard
#   TRAJECTORY                  default: empty, sample all trajectories
#   CAMERA                      default: left
#   MAX_SAMPLES                 default: 100
#   MAX_SAMPLES_PER_TRAJECTORY  default: 1
#   START_INDEX                 default: 0
#   SAMPLE_STRIDE               default: 1
#   NUM_BEAMS                   default: 64
#   POINTS_PER_BEAM             default: 160
#   MIN_DEPTH                   default: 0.1
#   MAX_DEPTH                   default: 120.0
#   VIS_MIN_DEPTH               default: empty, use MIN_DEPTH
#   VIS_MAX_DEPTH               default: empty, use MAX_DEPTH
#   SEED                        default: 2024
#
# Inference/evaluation:
#   RES_DIR                     default: data/tartan_test/res
#   NUM_INFERENCE_STEPS         default: 50
#   ENSEMBLE_SIZE               default: 1
#   PROCESSING_RESOLUTION       default: 768
#   CHECKPOINT                  default: prs-eth/marigold-depth-v1-0
#   EVAL_MAX_DEPTH              default: 60.0, exclude GT depth beyond this from metrics
#   EVAL_ONLY                   default: 0, set to 1 to skip Marigold-DC inference
#   USE_FULL_PRECISION          default: 0, set to 1 to enable
#   USE_TINY_VAE                default: 0, set to 1 to enable
#
# Safety:
#   The Python scripts skip existing outputs and never delete or overwrite data.

TARTAN_ROOT=${TARTAN_ROOT:-tartan}
DATASET_DIR=${DATASET_DIR:-data/tartan_test/dataset}
RES_DIR=${RES_DIR:-data/tartan_test/res}

SCENE=${SCENE:-}
DIFFICULTY=${DIFFICULTY:-}
TRAJECTORY=${TRAJECTORY:-}
CAMERA=${CAMERA:-left}
MAX_SAMPLES=${MAX_SAMPLES:-100}
MAX_SAMPLES_PER_TRAJECTORY=${MAX_SAMPLES_PER_TRAJECTORY:-1}
START_INDEX=${START_INDEX:-0}
SAMPLE_STRIDE=${SAMPLE_STRIDE:-1}
NUM_BEAMS=${NUM_BEAMS:-64}
POINTS_PER_BEAM=${POINTS_PER_BEAM:-160}
MIN_DEPTH=${MIN_DEPTH:-0.1}
MAX_DEPTH=${MAX_DEPTH:-120.0}
VIS_MIN_DEPTH=${VIS_MIN_DEPTH:-}
VIS_MAX_DEPTH=${VIS_MAX_DEPTH:-}
SEED=${SEED:-2024}

NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
ENSEMBLE_SIZE=${ENSEMBLE_SIZE:-1}
PROCESSING_RESOLUTION=${PROCESSING_RESOLUTION:-768}
CHECKPOINT=${CHECKPOINT:-prs-eth/marigold-depth-v1-0}
EVAL_MAX_DEPTH=${EVAL_MAX_DEPTH:-60.0}
EVAL_ONLY=${EVAL_ONLY:-0}
USE_FULL_PRECISION=${USE_FULL_PRECISION:-0}
USE_TINY_VAE=${USE_TINY_VAE:-0}

extract_args=(
  --tartan_root "${TARTAN_ROOT}"
  --output_dir "${DATASET_DIR}"
  --camera "${CAMERA}"
  --max_samples "${MAX_SAMPLES}"
  --max_samples_per_trajectory "${MAX_SAMPLES_PER_TRAJECTORY}"
  --start_index "${START_INDEX}"
  --sample_stride "${SAMPLE_STRIDE}"
  --num_beams "${NUM_BEAMS}"
  --points_per_beam "${POINTS_PER_BEAM}"
  --min_depth "${MIN_DEPTH}"
  --max_depth "${MAX_DEPTH}"
  --seed "${SEED}"
)

if [[ -n "${SCENE}" ]]; then
  extract_args+=(--scene "${SCENE}")
fi
if [[ -n "${DIFFICULTY}" ]]; then
  extract_args+=(--difficulty "${DIFFICULTY}")
fi
if [[ -n "${TRAJECTORY}" ]]; then
  extract_args+=(--trajectory "${TRAJECTORY}")
fi
if [[ -n "${VIS_MIN_DEPTH}" ]]; then
  extract_args+=(--vis_min_depth "${VIS_MIN_DEPTH}")
fi
if [[ -n "${VIS_MAX_DEPTH}" ]]; then
  extract_args+=(--vis_max_depth "${VIS_MAX_DEPTH}")
fi

run_args=(
  --input_dir "${DATASET_DIR}"
  --output_dir "${RES_DIR}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --ensemble_size "${ENSEMBLE_SIZE}"
  --processing_resolution "${PROCESSING_RESOLUTION}"
  --checkpoint "${CHECKPOINT}"
  --seed "${SEED}"
  --eval_max_depth "${EVAL_MAX_DEPTH}"
)

if [[ "${USE_FULL_PRECISION}" == "1" ]]; then
  run_args+=(--use_full_precision)
fi
if [[ "${USE_TINY_VAE}" == "1" ]]; then
  run_args+=(--use_tiny_vae)
fi
if [[ "${EVAL_ONLY}" == "1" ]]; then
  run_args+=(--eval_only)
fi

echo "Extracting Tartan test data..."
python script/extract_tartan_test.py "${extract_args[@]}"

echo "Running Marigold-DC and evaluating..."
python script/run_tartan_test.py "${run_args[@]}"
