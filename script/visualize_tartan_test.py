"""
Interactively visualize Tartan depth completion results.

Panels:
  RGB | Ground-truth depth
  Prediction | Absolute error

This script only reads dataset/prediction files. It never deletes or overwrites
data. Saved screenshots use a unique filename if the default target exists.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Tartan depth result viewer")
    parser.add_argument("--data_dir", type=Path, default=Path("data/tartan_test/dataset"), help="Dataset directory")
    parser.add_argument("--prediction_dir", type=Path, default=Path("data/tartan_test/res/prediction"), help="Prediction .npy directory")
    parser.add_argument("--save_dir", type=Path, default=Path("data/tartan_test/res/viewer"), help="Screenshot output directory")
    parser.add_argument("--start_index", type=int, default=0, help="Initial sample index")
    parser.add_argument("--min_depth", type=float, default=None, help="Depth visualization minimum; defaults to current GT/pred min")
    parser.add_argument("--max_depth", type=float, default=None, help="Depth visualization maximum; defaults to current GT/pred max")
    parser.add_argument("--valid_max_depth", type=float, default=60.0, help="Maximum depth shown/evaluated; filters sky and far points")
    parser.add_argument("--error_max", type=float, default=None, help="Error visualization maximum; defaults to current 95th percentile")
    parser.add_argument("--window_width", type=int, default=1600, help="Displayed window width")
    parser.add_argument("--window_height", type=int, default=900, help="Displayed window height")
    return parser.parse_args()


def find_rgb_path(rgb_dir: Path, stem: str) -> Optional[Path]:
    for suffix in (".png", ".jpg", ".jpeg"):
        path = rgb_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def collect_samples(data_dir: Path, prediction_dir: Path) -> Tuple[List[str], int]:
    rgb_dir = data_dir / "rgb"
    gt_dir = data_dir / "gt"
    rgb_stems = {path.stem for ext in ("*.png", "*.jpg", "*.jpeg") for path in rgb_dir.glob(ext)}
    gt_stems = {path.stem for path in gt_dir.glob("*.npy")}
    pred_stems = {path.stem for path in prediction_dir.glob("*.npy")}
    samples = sorted(rgb_stems & gt_stems & pred_stems)
    missing_predictions = len((rgb_stems & gt_stems) - pred_stems)
    return samples, missing_predictions


def load_sample(data_dir: Path, prediction_dir: Path, stem: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_path = find_rgb_path(data_dir / "rgb", stem)
    if rgb_path is None:
        raise FileNotFoundError(f"RGB file not found for {stem}")
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    gt = np.asarray(np.load(data_dir / "gt" / f"{stem}.npy"), dtype=np.float32).squeeze()
    pred = np.asarray(np.load(prediction_dir / f"{stem}.npy"), dtype=np.float32).squeeze()
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch for {stem}: gt={gt.shape}, pred={pred.shape}")
    return rgb, gt, pred


def valid_depth(depth: np.ndarray, max_depth: float) -> np.ndarray:
    return np.isfinite(depth) & (depth > 0) & (depth <= max_depth)


def finite_min_max(values: np.ndarray, fallback: Tuple[float, float] = (0.0, 1.0)) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def colorize(
    values: np.ndarray,
    mask: np.ndarray,
    vmin: float,
    vmax: float,
    colormap: int = cv2.COLORMAP_INFERNO,
) -> np.ndarray:
    if vmax <= vmin:
        vmax = vmin + 1e-6
    normalized = ((values.astype(np.float32) - vmin) / (vmax - vmin) * 255.0).clip(0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, colormap)
    colored[~mask] = 0
    return colored


def resize_panel(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(output, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def compute_metrics(gt: np.ndarray, pred: np.ndarray, valid_max_depth: float) -> Tuple[float, float, np.ndarray, np.ndarray]:
    mask = valid_depth(gt, valid_max_depth) & np.isfinite(pred)
    error = np.zeros_like(gt, dtype=np.float32)
    if not np.any(mask):
        return float("nan"), float("nan"), error, mask
    diff = pred[mask].astype(np.float64) - gt[mask].astype(np.float64)
    error[mask] = np.abs(diff).astype(np.float32)
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return mae, rmse, error, mask


def make_canvas(
    stem: str,
    index: int,
    total: int,
    rgb: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    mae, rmse, error, metric_mask = compute_metrics(gt, pred, args.valid_max_depth)
    depth_mask = valid_depth(gt, args.valid_max_depth) | valid_depth(pred, args.valid_max_depth)

    if args.min_depth is None or args.max_depth is None:
        dmin, dmax = finite_min_max(np.concatenate([gt[depth_mask], pred[depth_mask]]) if np.any(depth_mask) else np.array([]))
    else:
        dmin, dmax = args.min_depth, args.max_depth
    dmax = min(dmax, args.valid_max_depth)

    if args.error_max is None:
        err_values = error[metric_mask]
        err_max = float(np.percentile(err_values, 95)) if err_values.size else 1.0
        if err_max <= 0:
            err_max = 1.0
    else:
        err_max = args.error_max

    panel_w = max(args.window_width // 2, 1)
    panel_h = max((args.window_height - 48) // 2, 1)

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gt_vis = colorize(gt, valid_depth(gt, args.valid_max_depth), dmin, dmax)
    pred_vis = colorize(pred, valid_depth(pred, args.valid_max_depth), dmin, dmax)
    err_vis = colorize(error, metric_mask, 0.0, err_max, cv2.COLORMAP_MAGMA)

    panels = [
        add_label(resize_panel(rgb_bgr, (panel_w, panel_h)), "RGB"),
        add_label(resize_panel(gt_vis, (panel_w, panel_h)), f"GT depth [{dmin:.2f}, {dmax:.2f}] m"),
        add_label(resize_panel(pred_vis, (panel_w, panel_h)), "Prediction depth"),
        add_label(resize_panel(err_vis, (panel_w, panel_h)), f"Abs error [0, {err_max:.2f}] m"),
    ]

    top = np.concatenate([panels[0], panels[1]], axis=1)
    bottom = np.concatenate([panels[2], panels[3]], axis=1)
    body = np.concatenate([top, bottom], axis=0)

    header = np.zeros((48, body.shape[1], 3), dtype=np.uint8)
    title = f"{index + 1}/{total}  {stem}  MAE={mae:.4f}m  RMSE={rmse:.4f}m"
    cv2.putText(header, title, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return np.concatenate([header, body], axis=0)


def unique_screenshot_path(save_dir: Path, stem: str) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    base = save_dir / f"{stem}_viewer.png"
    if not base.exists():
        return base
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = save_dir / f"{stem}_viewer_{timestamp}.png"
    suffix = 1
    while candidate.exists():
        candidate = save_dir / f"{stem}_viewer_{timestamp}_{suffix}.png"
        suffix += 1
    return candidate


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    samples, missing_predictions = collect_samples(args.data_dir, args.prediction_dir)
    if not samples:
        raise RuntimeError("No matched RGB/GT/prediction samples found")
    if missing_predictions:
        logging.info("Matched %d samples; %d RGB/GT samples do not have predictions", len(samples), missing_predictions)
    else:
        logging.info("Matched %d samples", len(samples))

    index = min(max(args.start_index, 0), len(samples) - 1)
    window_name = "Tartan depth completion viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_width, args.window_height)

    while True:
        stem = samples[index]
        rgb, gt, pred = load_sample(args.data_dir, args.prediction_dir, stem)
        canvas = make_canvas(stem, index, len(samples), rgb, gt, pred, args)
        cv2.imshow(window_name, canvas)

        key = cv2.waitKeyEx(0)
        if key in (ord("q"), 27):
            break
        if key in (ord("n"), ord("l"), ord("j"), 83, 2555904):
            index = (index + 1) % len(samples)
        elif key in (ord("p"), ord("h"), ord("k"), 81, 2424832):
            index = (index - 1) % len(samples)
        elif key == ord("s"):
            path = unique_screenshot_path(args.save_dir, stem)
            cv2.imwrite(str(path), canvas)
            logging.info("Saved screenshot: %s", path)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
