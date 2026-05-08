"""
Extract a TartanAir-style test split for Marigold-DC.

Output layout:
data/tartan_test/dataset/
├── rgb/
├── sparse/
├── gt/
└── visuals/

Safety policy: this script never deletes files and never overwrites existing
outputs. Existing files are skipped, and new files are written with exclusive
create semantics.
"""

import argparse
import csv
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Tartan test data for depth completion")
    parser.add_argument("--tartan_root", type=Path, default=Path("tartan"), help="Tartan source root")
    parser.add_argument("--output_dir", type=Path, default=Path("data/tartan_test/dataset"), help="Output dataset directory")
    parser.add_argument("--scene", type=str, default=None, help="Optional scene name, e.g. abandonedfactory_night")
    parser.add_argument("--difficulty", type=str, default=None, choices=["Easy", "Hard"], help="Optional difficulty")
    parser.add_argument("--trajectory", type=str, default=None, help="Optional trajectory, e.g. P006")
    parser.add_argument("--camera", type=str, default="left", choices=["left", "right"], help="Camera side")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to write")
    parser.add_argument("--max_samples_per_trajectory", type=int, default=1, help="Maximum samples drawn from one trajectory")
    parser.add_argument("--start_index", type=int, default=0, help="Start index after sorting local trajectory frames")
    parser.add_argument("--sample_stride", type=int, default=1, help="Keep every Nth local trajectory frame")
    parser.add_argument("--sparse_mode", type=str, default="dsec_density", choices=["dsec_density", "lidar_lines"], help="Sparse depth simulation mode")
    parser.add_argument("--dsec_disparity_dir", type=Path, default=Path("data/dsec"), help="DSEC root used to estimate disparity valid-pixel density")
    parser.add_argument("--dsec_disparity_source", type=str, default="image", choices=["image", "event"], help="DSEC disparity source used for density")
    parser.add_argument("--dsec_density_max_files", type=int, default=0, help="Maximum DSEC disparity files used for density; 0 means all")
    parser.add_argument("--num_beams", type=int, default=32, help="Number of simulated lidar scan lines")
    parser.add_argument("--points_per_beam", type=int, default=80, help="Maximum sampled points per scan line")
    parser.add_argument("--horizontal_keep_prob", type=float, default=0.45, help="Probability of keeping each candidate point on a scan line")
    parser.add_argument("--min_depth", type=float, default=0.1, help="Minimum valid depth in meters")
    parser.add_argument("--max_depth", type=float, default=120.0, help="Maximum valid depth in meters")
    parser.add_argument("--vis_min_depth", type=float, default=None, help="Minimum depth for visualization color scaling")
    parser.add_argument("--vis_max_depth", type=float, default=None, help="Maximum depth for visualization color scaling")
    parser.add_argument("--seed", type=int, default=2024, help="Deterministic random seed")
    return parser.parse_args()


def frame_id(path: Path) -> Optional[str]:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        return None
    return match.group(1).lstrip("0") or "0"


def selected_roots(args: argparse.Namespace) -> List[Path]:
    root = args.tartan_root
    if not root.exists():
        raise FileNotFoundError(f"Tartan root does not exist: {root}")

    if args.scene and args.difficulty and args.trajectory:
        roots = [root / args.scene / args.difficulty / args.trajectory]
    elif args.scene and args.difficulty:
        roots = sorted([p for p in (root / args.scene / args.difficulty).glob("P*") if p.is_dir()])
    elif args.scene:
        roots = sorted([p for p in (root / args.scene).glob("*/P*") if p.is_dir()])
    elif args.difficulty:
        roots = sorted([p for p in root.glob(f"*/{args.difficulty}/P*") if p.is_dir()])
    else:
        roots = sorted([p for p in root.glob("*/*/P*") if p.is_dir()])

    if args.trajectory:
        roots = [p for p in roots if p.name == args.trajectory]

    return [
        p for p in roots
        if (p / f"image_{args.camera}").is_dir() and (p / f"depth_{args.camera}").is_dir()
    ]


def pair_frames(traj_root: Path, camera: str) -> List[Tuple[Path, Path]]:
    image_dir = traj_root / f"image_{camera}"
    depth_dir = traj_root / f"depth_{camera}"
    if not image_dir.is_dir() or not depth_dir.is_dir():
        return []

    depth_by_frame: Dict[str, Path] = {}
    for depth_path in sorted(depth_dir.glob(f"*_{camera}_depth.npy")):
        fid = frame_id(depth_path)
        if fid is not None and fid not in depth_by_frame:
            depth_by_frame[fid] = depth_path

    pairs: List[Tuple[Path, Path]] = []
    for image_path in sorted(image_dir.glob(f"*_{camera}.png")):
        fid = frame_id(image_path)
        if fid is not None and fid in depth_by_frame:
            pairs.append((image_path, depth_by_frame[fid]))
    return pairs


def collect_pairs(args: argparse.Namespace) -> List[Tuple[Path, Path]]:
    if args.sample_stride < 1:
        raise ValueError("--sample_stride must be >= 1")
    if args.max_samples < 1:
        raise ValueError("--max_samples must be >= 1")
    if args.max_samples_per_trajectory < 1:
        raise ValueError("--max_samples_per_trajectory must be >= 1")

    rng = np.random.default_rng(args.seed)
    roots = list(selected_roots(args))
    rng.shuffle(roots)

    pairs: List[Tuple[Path, Path]] = []
    for traj_root in roots:
        local_pairs = pair_frames(traj_root, args.camera)
        local_pairs = local_pairs[args.start_index :: args.sample_stride]
        if not local_pairs:
            continue
        rng.shuffle(local_pairs)
        pairs.extend(local_pairs[: args.max_samples_per_trajectory])
        if len(pairs) >= args.max_samples:
            break

    return pairs[: args.max_samples]


def safe_np_save(path: Path, array: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, array)


def safe_image_save(path: Path, image: Image.Image) -> None:
    with path.open("xb") as handle:
        image.save(handle, format="PNG")


def output_exists(paths: Iterable[Path]) -> bool:
    return any(path.exists() for path in paths)


def unique_manifest_path(output_dir: Path) -> Path:
    base = output_dir / "manifest.csv"
    if not base.exists():
        return base
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / f"manifest_{timestamp}.csv"
    suffix = 1
    while candidate.exists():
        candidate = output_dir / f"manifest_{timestamp}_{suffix}.csv"
        suffix += 1
    return candidate


def write_manifest(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "basename",
        "source_rgb",
        "source_depth",
        "scene",
        "difficulty",
        "trajectory",
        "frame",
        "sparse_points",
        "sparse_mode",
        "target_density",
        "status",
    ]
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def estimate_dsec_disparity_density(dsec_root: Path, source: str, max_files: int) -> float:
    if not dsec_root.exists():
        raise FileNotFoundError(f"DSEC directory does not exist: {dsec_root}")
    if max_files < 0:
        raise ValueError("--dsec_density_max_files must be >= 0")

    files = sorted(dsec_root.glob(f"**/disparity/{source}/*.png"))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"No DSEC disparity PNG files found under {dsec_root}/**/disparity/{source}")

    valid_pixels = 0
    total_pixels = 0
    for path in files:
        disparity = np.asarray(Image.open(path))
        valid_pixels += int(np.count_nonzero(disparity > 0))
        total_pixels += int(disparity.size)
    if total_pixels == 0:
        raise RuntimeError("DSEC disparity files contain no pixels")
    density = valid_pixels / total_pixels
    if density <= 0:
        raise RuntimeError("Estimated DSEC disparity density is zero")
    logging.info(
        "Estimated DSEC %s disparity density: %.6f from %d files",
        source,
        density,
        len(files),
    )
    return density


def simulate_density_sparse(
    depth: np.ndarray,
    target_density: float,
    min_depth: float,
    max_depth: float,
    rng: np.random.Generator,
) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Depth must be 2D after squeeze, got shape {depth.shape}")
    if not 0 < target_density <= 1:
        raise ValueError("target_density must be in (0, 1]")

    valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    sparse = np.zeros_like(depth, dtype=np.float32)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return sparse

    target_points = int(round(depth.size * target_density))
    target_points = min(max(target_points, 1), valid_indices.size)
    selected = rng.choice(valid_indices, size=target_points, replace=False)
    sparse.flat[selected] = depth.flat[selected]
    return sparse


def simulate_lidar_sparse(
    depth: np.ndarray,
    num_beams: int,
    points_per_beam: int,
    horizontal_keep_prob: float,
    min_depth: float,
    max_depth: float,
    rng: np.random.Generator,
) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Depth must be 2D after squeeze, got shape {depth.shape}")

    valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    sparse = np.zeros_like(depth, dtype=np.float32)
    if not np.any(valid):
        return sparse
    if not 0 < horizontal_keep_prob <= 1:
        raise ValueError("--horizontal_keep_prob must be in (0, 1]")

    height, width = depth.shape
    beam_count = min(max(num_beams, 1), height)
    rows = np.linspace(0, height - 1, beam_count).round().astype(np.int64)
    row_jitter = height // max(beam_count * 8, 1)
    if row_jitter > 0:
        rows = np.clip(rows + rng.integers(-row_jitter, row_jitter + 1, size=rows.shape), 0, height - 1)

    per_row = min(max(points_per_beam, 1), width)
    for row in np.unique(rows):
        valid_cols = np.flatnonzero(valid[row])
        if valid_cols.size == 0:
            continue
        if valid_cols.size <= per_row:
            cols = valid_cols
        else:
            base_cols = np.linspace(valid_cols.min(), valid_cols.max(), per_row).round().astype(np.int64)
            jitter = max(width // max(per_row * 6, 1), 1)
            cols = np.clip(base_cols + rng.integers(-jitter, jitter + 1, size=base_cols.shape), 0, width - 1)
            cols = cols[valid[row, cols]]
            if cols.size == 0:
                continue
        keep = rng.random(cols.shape[0]) < horizontal_keep_prob
        cols = np.unique(cols[keep])
        if cols.size == 0:
            cols = np.array([rng.choice(valid_cols)], dtype=np.int64)
        sparse[row, cols] = depth[row, cols]
    return sparse


def turbo_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    red = 34.61 + values * (1172.33 + values * (-10793.56 + values * (33300.12 + values * (-38394.49 + values * 14825.05))))
    green = 23.31 + values * (557.33 + values * (1225.33 + values * (-3574.96 + values * (1073.77 + values * 707.56))))
    blue = 27.2 + values * (3211.1 + values * (-15327.97 + values * (27814.0 + values * (-22569.18 + values * 6838.66))))
    return np.stack([red, green, blue], axis=-1).clip(0, 255).astype(np.uint8)


def colorize_depth(depth: np.ndarray, valid_mask: np.ndarray, vis_min: float, vis_max: float) -> Image.Image:
    image = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if vis_max <= vis_min:
        vis_max = vis_min + 1e-6
    normalized = (depth.astype(np.float32) - vis_min) / (vis_max - vis_min)
    image[valid_mask] = turbo_colormap(normalized[valid_mask])
    return Image.fromarray(image, mode="RGB")


def sample_name(image_path: Path, tartan_root: Path) -> str:
    rel = image_path.relative_to(tartan_root).with_suffix("")
    parts = [re.sub(r"[^A-Za-z0-9]+", "_", part).strip("_") for part in rel.parts]
    return "_".join(part for part in parts if part)


def parse_metadata(image_path: Path, tartan_root: Path) -> Dict[str, str]:
    rel = image_path.relative_to(tartan_root)
    parts = rel.parts
    return {
        "scene": parts[0] if len(parts) > 0 else "",
        "difficulty": parts[1] if len(parts) > 1 else "",
        "trajectory": parts[2] if len(parts) > 2 else "",
        "frame": image_path.stem,
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Safety policy: existing files are skipped; no files are deleted or overwritten.")

    pairs = collect_pairs(args)
    if not pairs:
        raise RuntimeError(
            "No RGB/depth pairs found. Expected paths like "
            "tartan/<scene>/<Easy|Hard>/<Pxxx>/image_left/*.png and depth_left/*.npy"
        )

    rgb_dir = args.output_dir / "rgb"
    sparse_dir = args.output_dir / "sparse"
    gt_dir = args.output_dir / "gt"
    gt_visuals_dir = args.output_dir / "visuals" / "gt"
    sparse_visuals_dir = args.output_dir / "visuals" / "sparse"
    for directory in (rgb_dir, sparse_dir, gt_dir, gt_visuals_dir, sparse_visuals_dir):
        directory.mkdir(parents=True, exist_ok=True)

    written = 0
    visualized_existing = 0
    skipped_existing = 0
    failed = 0
    manifest_rows: List[Dict[str, object]] = []

    vis_min = args.vis_min_depth if args.vis_min_depth is not None else args.min_depth
    vis_max = args.vis_max_depth if args.vis_max_depth is not None else args.max_depth
    target_density = None
    if args.sparse_mode == "dsec_density":
        target_density = estimate_dsec_disparity_density(
            args.dsec_disparity_dir,
            args.dsec_disparity_source,
            args.dsec_density_max_files,
        )

    for index, (image_path, depth_path) in tqdm(list(enumerate(pairs)), desc="Extracting"):
        basename = sample_name(image_path, args.tartan_root)
        rgb_out = rgb_dir / f"{basename}.png"
        sparse_out = sparse_dir / f"{basename}.npy"
        gt_out = gt_dir / f"{basename}.npy"
        gt_vis_out = gt_visuals_dir / f"{basename}_gt.png"
        sparse_vis_out = sparse_visuals_dir / f"{basename}_sparse.png"
        core_outputs = (rgb_out, sparse_out, gt_out)
        all_outputs = (rgb_out, sparse_out, gt_out, gt_vis_out, sparse_vis_out)
        metadata = parse_metadata(image_path, args.tartan_root)

        if output_exists(all_outputs) and not all(path.exists() for path in core_outputs):
            skipped_existing += 1
            continue

        try:
            if all(path.exists() for path in core_outputs):
                gt = np.asarray(np.load(gt_out), dtype=np.float32).squeeze()
                sparse = np.asarray(np.load(sparse_out), dtype=np.float32).squeeze()
                status = "existing"
            else:
                image = Image.open(image_path).convert("RGB")
                gt = np.asarray(np.load(depth_path), dtype=np.float32).squeeze()
                if gt.ndim != 2:
                    raise ValueError(f"Depth shape is not 2D: {gt.shape}")
                if gt.shape != (image.height, image.width):
                    raise ValueError(f"Image/depth shape mismatch: image={(image.height, image.width)} depth={gt.shape}")

                gt[~np.isfinite(gt)] = 0.0
                rng = np.random.default_rng(args.seed + index)
                if args.sparse_mode == "dsec_density":
                    if target_density is None:
                        raise RuntimeError("DSEC target density was not estimated")
                    sparse = simulate_density_sparse(
                        gt,
                        target_density,
                        args.min_depth,
                        args.max_depth,
                        rng,
                    )
                else:
                    sparse = simulate_lidar_sparse(
                        gt,
                        args.num_beams,
                        args.points_per_beam,
                        args.horizontal_keep_prob,
                        args.min_depth,
                        args.max_depth,
                        rng,
                    )
                if not np.any(sparse > 0):
                    raise ValueError("Sparse simulation produced no valid points")

                safe_image_save(rgb_out, image)
                safe_np_save(gt_out, gt)
                safe_np_save(sparse_out, sparse)
                written += 1
                status = "written"

            sparse_points = int(np.count_nonzero(sparse > 0))
            gt_valid = np.isfinite(gt) & (gt >= args.min_depth) & (gt <= args.max_depth)
            sparse_valid = sparse > 0
            if not gt_vis_out.exists():
                safe_image_save(gt_vis_out, colorize_depth(gt, gt_valid, vis_min, vis_max))
            if not sparse_vis_out.exists():
                safe_image_save(sparse_vis_out, colorize_depth(sparse, sparse_valid, vis_min, vis_max))
            if status == "existing":
                visualized_existing += 1

            manifest_rows.append({
                "basename": basename,
                "source_rgb": str(image_path),
                "source_depth": str(depth_path),
                "scene": metadata["scene"],
                "difficulty": metadata["difficulty"],
                "trajectory": metadata["trajectory"],
                "frame": metadata["frame"],
                "sparse_points": sparse_points,
                "sparse_mode": args.sparse_mode,
                "target_density": target_density if target_density is not None else "",
                "status": status,
            })
        except Exception as exc:
            failed += 1
            logging.error("Failed to extract %s with %s: %s", image_path, depth_path, exc)

    manifest_path = None
    if manifest_rows:
        manifest_path = unique_manifest_path(args.output_dir)
        write_manifest(manifest_path, manifest_rows)

    logging.info(
        "Extraction complete: written=%d visualized_existing=%d skipped_existing=%d failed=%d output=%s manifest=%s",
        written,
        visualized_existing,
        skipped_existing,
        failed,
        args.output_dir,
        manifest_path,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
