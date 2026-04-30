"""
Run Marigold-DC on the extracted Tartan test set and evaluate MAE/RMSE.

Default paths:
  input:  data/tartan_test/dataset
  output: data/tartan_test/res

Safety policy: this script never deletes files and never overwrites existing
outputs. Existing predictions/visualizations are skipped, and metric CSV files
receive a timestamped suffix when the default name already exists.
"""

import argparse
import csv
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import diffusers
import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from marigold_dc import MarigoldDepthCompletionPipeline
from script.utils import validate_rgb_sparse_structure


warnings.simplefilter(action="ignore", category=FutureWarning)
diffusers.utils.logging.disable_progress_bar()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Marigold-DC on extracted Tartan test data")
    parser.add_argument("--input_dir", type=Path, default=Path("data/tartan_test/dataset"), help="Input dataset directory")
    parser.add_argument("--output_dir", type=Path, default=Path("data/tartan_test/res"), help="Output directory")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--ensemble_size", type=int, default=1, help="Number of predictions to ensemble")
    parser.add_argument("--processing_resolution", type=int, default=768, help="Denoising resolution")
    parser.add_argument("--checkpoint", type=str, default="prs-eth/marigold-depth-v1-0", help="Model checkpoint")
    parser.add_argument("--use_full_precision", action="store_true", help="Use float32 inference")
    parser.add_argument("--use_tiny_vae", action="store_true", help="Use tiny VAE")
    parser.add_argument("--seed", type=int, default=2024, help="Inference seed")
    parser.add_argument("--eval_max_depth", type=float, default=60.0, help="Maximum GT depth included in MAE/RMSE")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate existing predictions; do not load or run Marigold-DC")
    return parser.parse_args()


def safe_np_save(path: Path, array: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, array)


def safe_image_save(path: Path, image: Image.Image, image_format: str) -> None:
    with path.open("xb") as handle:
        image.save(handle, format=image_format)


def unique_csv_path(metrics_dir: Path) -> Path:
    base = metrics_dir / "evaluation_results.csv"
    if not base.exists():
        return base
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = metrics_dir / f"evaluation_results_{timestamp}.csv"
    suffix = 1
    while candidate.exists():
        candidate = metrics_dir / f"evaluation_results_{timestamp}_{suffix}.csv"
        suffix += 1
    return candidate


def find_rgb_files(input_dir: Path) -> List[Path]:
    rgb_dir = input_dir / "rgb"
    files: List[Path] = []
    for extension in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(rgb_dir.glob(extension))
    return sorted(files)


def validate_input(input_dir: Path) -> None:
    success, error_msg = validate_rgb_sparse_structure(str(input_dir))
    if not success:
        raise RuntimeError(error_msg)
    gt_dir = input_dir / "gt"
    if not gt_dir.is_dir():
        raise RuntimeError(f"GT directory not found: {gt_dir}")

    rgb_stems = {path.stem for path in find_rgb_files(input_dir)}
    gt_stems = {path.stem for path in gt_dir.glob("*.npy")}
    if rgb_stems != gt_stems:
        missing_gt = sorted(rgb_stems - gt_stems)
        missing_rgb = sorted(gt_stems - rgb_stems)
        raise RuntimeError(f"Filename mismatch between rgb and gt. missing_gt={missing_gt} missing_rgb={missing_rgb}")


def resolve_runtime(args: argparse.Namespace) -> Tuple[torch.device, torch.dtype, int, int, int]:
    num_inference_steps = args.num_inference_steps
    ensemble_size = args.ensemble_size
    processing_resolution = args.processing_resolution

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        if processing_resolution > 512:
            logging.warning("CUDA not found: reducing processing_resolution to 512")
            processing_resolution = 512
        if num_inference_steps > 10:
            logging.warning("CUDA not found: reducing num_inference_steps to 10")
            num_inference_steps = 10
        if ensemble_size > 1:
            logging.warning("CUDA not found: reducing ensemble_size to 1")
            ensemble_size = 1

    torch_dtype = torch.float32 if args.use_full_precision else torch.bfloat16
    return device, torch_dtype, num_inference_steps, ensemble_size, processing_resolution


def load_pipeline(args: argparse.Namespace, device: torch.device, torch_dtype: torch.dtype) -> MarigoldDepthCompletionPipeline:
    pipe = MarigoldDepthCompletionPipeline.from_pretrained(args.checkpoint, prediction_type="depth").to(device, dtype=torch_dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    if args.use_tiny_vae:
        logging.info("Using tiny VAE")
        del pipe.vae
        pipe.vae = diffusers.AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device, dtype=torch_dtype)
    return pipe


def compute_metrics(pred: np.ndarray, gt: np.ndarray, eval_max_depth: float) -> Dict[str, float]:
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape} gt={gt.shape}")
    mask = np.isfinite(pred) & np.isfinite(gt) & (gt > 0) & (gt <= eval_max_depth)
    if not np.any(mask):
        return {"mae": float("nan"), "rmse": float("nan")}
    diff = pred[mask].astype(np.float64) - gt[mask].astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
    }


def write_metrics(csv_path: Path, rows: List[Dict[str, object]]) -> None:
    with csv_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "mae", "rmse"])
        writer.writeheader()
        writer.writerows(rows)


def evaluate_predictions(input_dir: Path, prediction_dir: Path, metrics_dir: Path, eval_max_depth: float) -> Optional[Path]:
    rows: List[Dict[str, object]] = []
    maes: List[float] = []
    rmses: List[float] = []

    for pred_path in sorted(prediction_dir.glob("*.npy")):
        gt_path = input_dir / "gt" / pred_path.name
        if not gt_path.exists():
            logging.warning("Skipping metric for prediction without GT: %s", pred_path)
            continue
        try:
            metrics = compute_metrics(np.load(pred_path), np.load(gt_path), eval_max_depth)
            rows.append({"filename": pred_path.name, "mae": metrics["mae"], "rmse": metrics["rmse"]})
            if np.isfinite(metrics["mae"]):
                maes.append(metrics["mae"])
            if np.isfinite(metrics["rmse"]):
                rmses.append(metrics["rmse"])
        except Exception as exc:
            logging.error("Failed to evaluate %s: %s", pred_path, exc)
            rows.append({"filename": pred_path.name, "mae": float("nan"), "rmse": float("nan")})

    if not rows:
        logging.warning("No predictions available for evaluation")
        return None

    global_row = {
        "filename": "GLOBAL_AVERAGE",
        "mae": float(np.mean(maes)) if maes else float("nan"),
        "rmse": float(np.mean(rmses)) if rmses else float("nan"),
    }
    csv_path = unique_csv_path(metrics_dir)
    write_metrics(csv_path, [global_row] + rows)
    return csv_path


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Safety policy: existing files are skipped; no files are deleted or overwritten.")

    validate_input(args.input_dir)

    prediction_dir = args.output_dir / "prediction"
    visuals_dir = args.output_dir / "visuals"
    metrics_dir = args.output_dir / "metrics"
    for directory in (prediction_dir, visuals_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    successful = 0
    skipped_existing = 0
    failed = 0

    if args.eval_only:
        logging.info("Eval-only mode: skipping Marigold-DC loading and inference.")
    else:
        device, torch_dtype, num_inference_steps, ensemble_size, processing_resolution = resolve_runtime(args)
        logging.info("Using device=%s dtype=%s", device, torch_dtype)
        pipe = load_pipeline(args, device, torch_dtype)

        for rgb_path in tqdm(find_rgb_files(args.input_dir), desc="Running inference"):
            basename = rgb_path.stem
            sparse_path = args.input_dir / "sparse" / f"{basename}.npy"
            pred_path = prediction_dir / f"{basename}.npy"
            vis_path = visuals_dir / f"{basename}_vis.jpg"

            if pred_path.exists() and vis_path.exists():
                skipped_existing += 1
                continue
            if pred_path.exists() or vis_path.exists():
                logging.warning("Skipping %s because partial output already exists", basename)
                skipped_existing += 1
                continue

            try:
                pred = pipe(
                    image=Image.open(rgb_path).convert("RGB"),
                    sparse_depth=np.load(sparse_path),
                    num_inference_steps=num_inference_steps,
                    ensemble_size=ensemble_size,
                    processing_resolution=processing_resolution,
                    seed=args.seed,
                )
                pred = np.asarray(pred, dtype=np.float32)
                safe_np_save(pred_path, pred)
                vis = pipe.image_processor.visualize_depth(pred, val_min=float(np.nanmin(pred)), val_max=float(np.nanmax(pred)))[0]
                safe_image_save(vis_path, vis, "JPEG")
                successful += 1
            except Exception as exc:
                failed += 1
                logging.error("Failed to process %s: %s", basename, exc)

    csv_path = evaluate_predictions(args.input_dir, prediction_dir, metrics_dir, args.eval_max_depth)

    logging.info(
        "Tartan test complete: successful=%d skipped_existing=%d failed=%d output=%s metrics=%s",
        successful,
        skipped_existing,
        failed,
        args.output_dir,
        csv_path,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
