import argparse
from pathlib import Path

import cv2
import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm


def colorize_depth(depth_event: np.ndarray) -> np.ndarray:
    valid = depth_event > 0
    out = np.zeros_like(depth_event, dtype=np.float32)
    if np.any(valid):
        vals = depth_event[valid]
        vmin = float(np.percentile(vals, 1.0))
        vmax = float(np.percentile(vals, 99.0))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        out[valid] = np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0)
    out_u8 = (out * 255.0).astype(np.uint8)
    return cv2.applyColorMap(out_u8, cv2.COLORMAP_TURBO)


def colorize_voxel_energy(voxel: np.ndarray, percentile: float) -> np.ndarray:
    energy = np.abs(voxel).sum(axis=0)
    vmax = float(np.percentile(energy, percentile))
    if vmax <= 0:
        vmax = 1.0
    norm = np.clip(energy / vmax, 0.0, 1.0)
    norm_u8 = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(norm_u8, cv2.COLORMAP_HOT)


def save_overlay_viz(
    depth_event: np.ndarray,
    voxel: np.ndarray,
    out_path: Path,
    alpha: float,
    percentile: float,
    frame_id: str,
    t0: int,
    t1: int,
    event_count: int,
) -> None:
    depth_color = colorize_depth(depth_event)
    voxel_color = colorize_voxel_energy(voxel, percentile=percentile)
    blend = cv2.addWeighted(depth_color, 1.0 - alpha, voxel_color, alpha, 0.0)
    cv2.putText(
        blend,
        f"frame={frame_id} t0={t0} t1={t1} events={event_count}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        blend,
        f"frame={frame_id} t0={t0} t1={t1} events={event_count}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), blend)


def main():
    parser = argparse.ArgumentParser("Generate DSEC overlay visualization from preprocessing outputs")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--viz_every", type=int, default=5, help="Save overlay every N voxel frames")
    parser.add_argument("--viz_alpha", type=float, default=0.45, help="Voxel overlay alpha")
    parser.add_argument("--viz_percentile", type=float, default=99.0, help="Voxel normalization percentile")
    args = parser.parse_args()

    depth_dir = args.output_dir / "depth_event"
    voxel_dir = args.output_dir / "voxel"
    viz_dir = args.output_dir / "viz_overlay"
    viz_dir.mkdir(parents=True, exist_ok=True)

    depth_files = sorted(depth_dir.glob("*.npy"))
    pbar = tqdm(depth_files, desc="Generating viz", dynamic_ncols=True)
    for i, depth_path in enumerate(pbar):
        stem = depth_path.stem
        pbar.set_postfix(frame=stem)

        if args.viz_every > 0 and (i % args.viz_every != 0):
            continue

        voxel_path = voxel_dir / f"{stem}.h5"
        if not voxel_path.exists():
            tqdm.write(f"[WARN] Missing voxel file for frame {stem}: {voxel_path}")
            continue

        depth_event = np.load(str(depth_path))
        with h5py.File(str(voxel_path), "r") as f:
            voxel = f["voxel"][:].astype(np.float32)
            t0 = int(f["t0_ns"][0]) if "t0_ns" in f else -1
            t1 = int(f["t1_ns"][0]) if "t1_ns" in f else -1
        event_count = -1

        save_overlay_viz(
            depth_event=depth_event,
            voxel=voxel,
            out_path=viz_dir / f"{stem}.png",
            alpha=float(args.viz_alpha),
            percentile=float(args.viz_percentile),
            frame_id=stem,
            t0=t0,
            t1=t1,
            event_count=event_count,
        )


if __name__ == "__main__":
    main()
