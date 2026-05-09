#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# DSEC + Marigold preprocessing launcher
# -----------------------------
#
# Main parameters (edit these):
#   DSEC_ROOT                Root containing all scene folders; if set, iterate all scene folders under it
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

DSEC_ROOT="${DSEC_ROOT:-/root/workspace/testrange/download/dsec}"
SEQ_ROOT="${SEQ_ROOT:-/root/workspace/testrange/download/dsec/interlaken_00_c}"

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

run_one_scene() {
  local seq_root="$1"
  local scene_name
  scene_name="$(basename "${seq_root}")"

  local output_dir="${OUTPUT_DIR:-${seq_root}/marigold_preprocess_out}"
  local cam_to_cam_yaml="${CAM_TO_CAM_YAML:-${seq_root}/calibration/cam_to_cam.yaml}"
  local events_h5="${EVENTS_H5:-${seq_root}/events_left/events.h5}"
  local rgb_dir="${RGB_DIR:-${seq_root}/images_rectified_left}"
  local disp_dir="${DISP_DIR:-${seq_root}/disparity_image}"
  local rgb_ts="${RGB_TS:-${seq_root}/${scene_name}_image_timestamps.txt}"
  local disp_ts="${DISP_TS:-${seq_root}/${scene_name}_disparity_timestamps.txt}"
  local rectify_map="${RECTIFY_MAP:-${seq_root}/events_left/rectify_map.h5}"

  echo "==== Scene: ${scene_name} ===="

  cmd=(
    python script/preprocess_dsec_marigold_new.py
    --left_rgb_dir "${rgb_dir}"
    --disparity_dir "${disp_dir}"
    --events_h5 "${events_h5}"
    --rgb_timestamps_ns "${rgb_ts}"
    --disparity_timestamps_ns "${disp_ts}"
    --cam_to_cam_yaml "${cam_to_cam_yaml}"
    --event_width "${EVENT_WIDTH}"
    --event_height "${EVENT_HEIGHT}"
    --output_dir "${output_dir}"
    --checkpoint "${CHECKPOINT}"
    --num_inference_steps "${STEPS}"
    --ensemble_size "${ENSEMBLE}"
    --processing_resolution "${PROC_RES}"
    --num_bins "${NUM_BINS}"
    --disparity_scale "${DISP_SCALE}"
    --rectify_map_h5 "${rectify_map}"
  )

  if [[ "${USE_FULL_PRECISION}" == "1" ]]; then
    cmd+=(--use_full_precision)
  fi

  echo "Running:"
  printf ' %q' "${cmd[@]}"
  echo
  "${cmd[@]}"

  viz_cmd=(
    python script/generate_dsec_overlay_viz.py
    --output_dir "${output_dir}"
    --viz_every "${VIZ_EVERY}"
    --viz_alpha "${VIZ_ALPHA}"
    --viz_percentile "${VIZ_PERCENTILE}"
  )

  echo "Running visualization:"
  printf ' %q' "${viz_cmd[@]}"
  echo
  "${viz_cmd[@]}"
}

if [[ -n "${DSEC_ROOT}" ]]; then
  shopt -s nullglob
  scenes=( "${DSEC_ROOT}"/* )
  shopt -u nullglob

  if [[ "${#scenes[@]}" -eq 0 ]]; then
    echo "No scene directories found under DSEC_ROOT: ${DSEC_ROOT}" >&2
    exit 1
  fi

  for scene_dir in "${scenes[@]}"; do
    [[ -d "${scene_dir}" ]] || continue
    echo "${scene_dir}"
    # run_one_scene "${scene_dir}"
  done
else
  echo "${SEQ_ROOT}"
  # run_one_scene "${SEQ_ROOT}"
fi
