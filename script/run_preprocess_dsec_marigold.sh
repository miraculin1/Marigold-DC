#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# DSEC + Marigold preprocessing launcher
# -----------------------------
#
# Main parameters (edit these):
#   SEQ_ROOT                 DSEC sequence root
#   OUTPUT_DIR               Output directory
#   CAM_TO_CAM_YAML          DSEC calibration yaml
#   EVENTS_H5                Event stream h5
#   RGB_DIR                  Rectified left RGB dir
#   DISP_DIR                 Disparity image dir
#   RGB_TS                   RGB timestamp file (ns/us as your data provides)
#   DISP_TS                  Disparity timestamp file (same unit as RGB_TS)
#   EVENT_WIDTH/HEIGHT       Event camera resolution
#
# Optional runtime parameters:
#   NUM_BINS                 Voxel bins (default 5, aligned with convert_tartan.py)
#   DISP_SCALE               Disparity png scale (default 256.0 for DSEC)
#   CHECKPOINT               Marigold checkpoint
#   STEPS / ENSEMBLE / PROC_RES
#   VIZ_EVERY / VIZ_ALPHA / VIZ_PERCENTILE (used by overlay visualization step)
#   USE_FULL_PRECISION       0 or 1
#
# Usage:
#   bash script/run_preprocess_dsec_marigold.sh

source ~/miniconda3/bin/activate
conda activate marigold

SEQ_ROOT="${SEQ_ROOT:-/root/workspace/testrange/download/dsec/interlaken_00_c}"
OUTPUT_DIR="${OUTPUT_DIR:-${SEQ_ROOT}/marigold_preprocess_out}"

CAM_TO_CAM_YAML="${CAM_TO_CAM_YAML:-${SEQ_ROOT}/calibration/cam_to_cam.yaml}"
EVENTS_H5="${EVENTS_H5:-${SEQ_ROOT}/events_left/events.h5}"
RGB_DIR="${RGB_DIR:-${SEQ_ROOT}/images_rectified_left}"
DISP_DIR="${DISP_DIR:-${SEQ_ROOT}/disparity_image}"
RGB_TS="${RGB_TS:-${SEQ_ROOT}/interlaken_00_c_image_timestamps.txt}"
DISP_TS="${DISP_TS:-${SEQ_ROOT}/interlaken_00_c_disparity_timestamps.txt}"

EVENT_WIDTH="${EVENT_WIDTH:-640}"
EVENT_HEIGHT="${EVENT_HEIGHT:-480}"

NUM_BINS="${NUM_BINS:-5}"
DISP_SCALE="${DISP_SCALE:-256.0}"
CHECKPOINT="${CHECKPOINT:-prs-eth/marigold-depth-v1-0}"
STEPS="${STEPS:-50}"
ENSEMBLE="${ENSEMBLE:-1}"
PROC_RES="${PROC_RES:-768}"
VIZ_EVERY="${VIZ_EVERY:-5}"
VIZ_ALPHA="${VIZ_ALPHA:-0.45}"
VIZ_PERCENTILE="${VIZ_PERCENTILE:-99.0}"
USE_FULL_PRECISION="${USE_FULL_PRECISION:-0}"

cmd=(
  python script/preprocess_dsec_marigold.py
  --left_rgb_dir "${RGB_DIR}"
  --disparity_dir "${DISP_DIR}"
  --events_h5 "${EVENTS_H5}"
  --rgb_timestamps_ns "${RGB_TS}"
  --disparity_timestamps_ns "${DISP_TS}"
  --cam_to_cam_yaml "${CAM_TO_CAM_YAML}"
  --event_width "${EVENT_WIDTH}"
  --event_height "${EVENT_HEIGHT}"
  --output_dir "${OUTPUT_DIR}"
  --checkpoint "${CHECKPOINT}"
  --num_inference_steps "${STEPS}"
  --ensemble_size "${ENSEMBLE}"
  --processing_resolution "${PROC_RES}"
  --num_bins "${NUM_BINS}"
  --disparity_scale "${DISP_SCALE}"
)

if [[ "${USE_FULL_PRECISION}" == "1" ]]; then
  cmd+=(--use_full_precision)
fi

# echo "Running:"
# printf ' %q' "${cmd[@]}"
# echo
# "${cmd[@]}"

viz_cmd=(
  python script/generate_dsec_overlay_viz.py
  --output_dir "${OUTPUT_DIR}"
  --viz_every "${VIZ_EVERY}"
  --viz_alpha "${VIZ_ALPHA}"
  --viz_percentile "${VIZ_PERCENTILE}"
)

echo "Running visualization:"
printf ' %q' "${viz_cmd[@]}"
echo
"${viz_cmd[@]}"
