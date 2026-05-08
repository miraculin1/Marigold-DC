import argparse
import sys
from pathlib import Path
from typing import Tuple

import h5py
import hdf5plugin
import numpy as np
import torch
import yaml
from PIL import Image
from diffusers import DDIMScheduler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from marigold_dc import MarigoldDepthCompletionPipeline


def load_matrix(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    mat = np.loadtxt(str(path), dtype=np.float64)
    if mat.shape != shape:
        raise ValueError(f"Invalid shape for {path}, expected {shape}, got {mat.shape}")
    return mat


def parse_cam_matrix(camera_matrix_4: list) -> np.ndarray:
    fx, fy, cx, cy = [float(v) for v in camera_matrix_4]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_dsec_calibration(cam_to_cam_yaml: Path):
    with open(cam_to_cam_yaml, "r", encoding="utf-8") as f:
        calib = yaml.safe_load(f)

    k_left = parse_cam_matrix(calib["intrinsics"]["camRect1"]["camera_matrix"])
    k_event = parse_cam_matrix(calib["intrinsics"]["camRect0"]["camera_matrix"])

    t_10 = np.array(calib["extrinsics"]["T_10"], dtype=np.float64)
    t_event_from_left = np.linalg.inv(t_10)  # T_01: event-left <- rgb-left

    cams_12 = np.array(calib["disparity_to_depth"]["cams_12"], dtype=np.float64)
    denom = float(cams_12[3, 2])
    if abs(denom) < 1e-12:
        raise ValueError("Invalid cams_12 for baseline computation: denominator is zero")
    baseline_m = 1.0 / denom

    return k_left, k_event, t_event_from_left, baseline_m


def disparity_to_sparse_depth(
    disparity: np.ndarray, fx: float, baseline: float, disparity_scale: float
) -> np.ndarray:
    disparity = disparity.astype(np.float32) / disparity_scale
    sparse_depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > 0
    sparse_depth[valid] = (fx * baseline) / disparity[valid]
    return sparse_depth


def make_marigold_pipe(checkpoint: str, use_full_precision: bool) -> MarigoldDepthCompletionPipeline:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch_dtype = torch.float32 if use_full_precision else torch.bfloat16
    pipe = MarigoldDepthCompletionPipeline.from_pretrained(
        checkpoint,
        prediction_type="depth",
    ).to(device, dtype=torch_dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    return pipe


def dense_depth_to_event_depth(
    depth_left: np.ndarray,
    k_left: np.ndarray,
    k_event: np.ndarray,
    t_event_from_left: np.ndarray,
    event_h: int,
    event_w: int,
) -> np.ndarray:
    h, w = depth_left.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    z = depth_left.reshape(-1)
    x = xs.reshape(-1)
    y = ys.reshape(-1)

    valid = z > 0
    if not np.any(valid):
        return np.zeros((event_h, event_w), dtype=np.float32)

    z = z[valid]
    x = x[valid]
    y = y[valid]

    fx_l, fy_l = k_left[0, 0], k_left[1, 1]
    cx_l, cy_l = k_left[0, 2], k_left[1, 2]
    x_l = (x - cx_l) * z / fx_l
    y_l = (y - cy_l) * z / fy_l

    ones = np.ones_like(z)
    pts_left = np.stack([x_l, y_l, z, ones], axis=0)
    pts_event = (t_event_from_left @ pts_left)[:3, :]

    z_e = pts_event[2]
    valid_e = z_e > 0
    pts_event = pts_event[:, valid_e]
    z_e = z_e[valid_e]
    if pts_event.shape[1] == 0:
        return np.zeros((event_h, event_w), dtype=np.float32)

    uv = k_event @ pts_event
    u = uv[0] / uv[2]
    v = uv[1] / uv[2]
    u_i = np.round(u).astype(np.int32)
    v_i = np.round(v).astype(np.int32)

    in_img = (u_i >= 0) & (u_i < event_w) & (v_i >= 0) & (v_i < event_h)
    u_i = u_i[in_img]
    v_i = v_i[in_img]
    z_e = z_e[in_img]

    depth_event = np.zeros((event_h, event_w), dtype=np.float32)
    zbuf = np.full((event_h, event_w), np.inf, dtype=np.float32)
    for uu, vv, zz in zip(u_i, v_i, z_e):
        if zz < zbuf[vv, uu]:
            zbuf[vv, uu] = zz
            depth_event[vv, uu] = zz
    return depth_event


def to_voxel_grid_numpy(events: np.ndarray, num_bins: int, width: int, height: int) -> np.ndarray:
    assert events.shape[1] == 4
    events = events.astype(np.float32)
    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    first_stamp = events[0, 0]
    last_stamp = events[-1, 0]
    delta_t = last_stamp - first_stamp
    if delta_t == 0:
        delta_t = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / delta_t
    ts = events[:, 0]
    xs = events[:, 1]
    ys = events[:, 2]
    pols = events[:, 3]
    pols[pols == 0] = -1

    tis = ts.astype(np.int64)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid = tis < num_bins
    np.add.at(
        voxel_grid,
        (xs[valid] + ys[valid] * width + tis[valid] * width * height).astype(np.int64),
        vals_left[valid],
    )
    valid = (tis + 1) < num_bins
    np.add.at(
        voxel_grid,
        (xs[valid] + ys[valid] * width + (tis[valid] + 1) * width * height).astype(np.int64),
        vals_right[valid],
    )
    return voxel_grid.reshape(num_bins, height, width)


def load_events(events_h5: Path):
    with h5py.File(str(events_h5), "r") as f:
        if "events" in f:
            g = f["events"]
            x = g["x"][:]
            y = g["y"][:]
            t = g["t"][:]
            p = g["p"][:]
        else:
            x = f["x"][:]
            y = f["y"][:]
            t = f["t"][:]
            p = f["p"][:]

        if "t_offset" in f:
            t = t.astype(np.int64) + int(f["t_offset"][()])
    return x, y, t, p


def event_slice_to_voxel(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    t0: int,
    t1: int,
    num_bins: int,
    width: int,
    height: int,
) -> np.ndarray:
    i0 = np.searchsorted(t, t0, side="left")
    i1 = np.searchsorted(t, t1, side="right")
    if i1 <= i0:
        return np.zeros((num_bins, height, width), dtype=np.float32)

    ev = np.stack(
        [
            t[i0:i1].astype(np.float32),
            x[i0:i1].astype(np.float32),
            y[i0:i1].astype(np.float32),
            p[i0:i1].astype(np.float32),
        ],
        axis=1,
    )
    return to_voxel_grid_numpy(ev, num_bins=num_bins, width=width, height=height)


def main():
    parser = argparse.ArgumentParser("DSEC preprocess: Marigold depth + projection + event voxel")
    parser.add_argument("--left_rgb_dir", type=Path, required=True)
    parser.add_argument("--disparity_dir", type=Path, required=True)
    parser.add_argument("--events_h5", type=Path, required=True)
    parser.add_argument("--rgb_timestamps_ns", type=Path, required=True)
    parser.add_argument(
        "--disparity_timestamps_ns",
        type=Path,
        required=True,
        help="Disparity timestamps used as voxel anchors",
    )
    parser.add_argument(
        "--cam_to_cam_yaml",
        type=Path,
        default=None,
        help="DSEC calibration yaml for auto-loading intrinsics/extrinsics/baseline",
    )
    parser.add_argument("--k_left_txt", type=Path, default=None)
    parser.add_argument("--k_event_txt", type=Path, default=None)
    parser.add_argument("--t_event_from_left_txt", type=Path, default=None)
    parser.add_argument("--baseline_m", type=float, default=None)
    parser.add_argument("--disparity_scale", type=float, default=256.0, help="DSEC disparity PNG scale")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default="prs-eth/marigold-depth-v1-0")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--ensemble_size", type=int, default=1)
    parser.add_argument("--processing_resolution", type=int, default=768)
    parser.add_argument("--num_bins", type=int, default=5, help="Follow convert_tartan.py default NBINS=5")
    parser.add_argument("--event_width", type=int, required=True)
    parser.add_argument("--event_height", type=int, required=True)
    parser.add_argument("--use_full_precision", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "depth_event").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "voxel").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "depth_left_dense").mkdir(parents=True, exist_ok=True)

    if args.cam_to_cam_yaml is not None:
        k_left, k_event, t_event_from_left, baseline_m = load_dsec_calibration(args.cam_to_cam_yaml)
    else:
        if (
            args.k_left_txt is None
            or args.k_event_txt is None
            or args.t_event_from_left_txt is None
            or args.baseline_m is None
        ):
            raise ValueError(
                "Provide --cam_to_cam_yaml, or provide all manual parameters: "
                "--k_left_txt --k_event_txt --t_event_from_left_txt --baseline_m"
            )
        k_left = load_matrix(args.k_left_txt, (3, 3))
        k_event = load_matrix(args.k_event_txt, (3, 3))
        t_event_from_left = load_matrix(args.t_event_from_left_txt, (4, 4))
        baseline_m = float(args.baseline_m)

    ts_rgb = np.loadtxt(str(args.rgb_timestamps_ns), dtype=np.int64).reshape(-1)
    ts_disp = np.loadtxt(str(args.disparity_timestamps_ns), dtype=np.int64).reshape(-1)
    rgb_files = sorted(args.left_rgb_dir.glob("*.png"))
    disp_files = sorted(args.disparity_dir.glob("*.png"))
    if not (len(rgb_files) == len(ts_rgb)):
        raise ValueError(f"Count mismatch: rgb={len(rgb_files)}, rgb_timestamps={len(ts_rgb)}")
    if not (len(disp_files) == len(ts_disp)):
        raise ValueError(
            f"Count mismatch: disparity={len(disp_files)}, disparity_timestamps={len(ts_disp)}"
        )

    rgb_by_index = {int(p.stem): p for p in rgb_files}
    ts_rgb_by_index = {idx: int(ts) for idx, ts in enumerate(ts_rgb)}

    x, y, t, p = load_events(args.events_h5)
    pipe = make_marigold_pipe(args.checkpoint, args.use_full_precision)
    fx_left = float(k_left[0, 0])

    total_frames = len(disp_files)
    pbar = tqdm(zip(disp_files, ts_disp), total=total_frames, desc="Preprocessing", dynamic_ncols=True)
    for i, (disp_path, ts) in enumerate(pbar):
        disp_idx = int(disp_path.stem)
        if disp_idx not in rgb_by_index or disp_idx not in ts_rgb_by_index:
            raise ValueError(f"Disparity frame {disp_idx:06d} has no matching RGB frame/timestamp")

        rgb_path = rgb_by_index[disp_idx]
        rgb = Image.open(str(rgb_path)).convert("RGB")
        disparity = np.asarray(Image.open(str(disp_path)))
        sparse_depth = disparity_to_sparse_depth(disparity, fx_left, baseline_m, args.disparity_scale)

        dense_depth = pipe(
            image=rgb,
            sparse_depth=sparse_depth,
            num_inference_steps=args.num_inference_steps,
            ensemble_size=args.ensemble_size,
            processing_resolution=args.processing_resolution,
        ).astype(np.float32)

        depth_event = dense_depth_to_event_depth(
            dense_depth,
            k_left=k_left,
            k_event=k_event,
            t_event_from_left=t_event_from_left,
            event_h=args.event_height,
            event_w=args.event_width,
        )

        stem = disp_path.stem
        np.save(str(args.output_dir / "depth_left_dense" / f"{stem}.npy"), dense_depth)
        np.save(str(args.output_dir / "depth_event" / f"{stem}.npy"), depth_event)

        # Follow convert_tartan.py semantics: first anchor has no previous window.
        if i > 0:
            t0, t1 = int(ts_disp[i - 1]), int(ts_disp[i])
            i0 = np.searchsorted(t, t0, side="left")
            i1 = np.searchsorted(t, t1, side="right")
            event_count = int(max(0, i1 - i0))
            voxel = event_slice_to_voxel(
                x=x,
                y=y,
                t=t,
                p=p,
                t0=t0,
                t1=t1,
                num_bins=args.num_bins,
                width=args.event_width,
                height=args.event_height,
            )
            with h5py.File(str(args.output_dir / "voxel" / f"{stem}.h5"), "w") as f:
                f.create_dataset(
                    "voxel",
                    data=voxel.astype(np.float16),
                    **hdf5plugin.Blosc(cname="zstd", clevel=4, shuffle=hdf5plugin.Blosc.SHUFFLE),
                )
                f.create_dataset("t0_ns", data=np.array([t0], dtype=np.int64))
                f.create_dataset("t1_ns", data=np.array([t1], dtype=np.int64))
                f.create_dataset("t_disp_ns", data=np.array([int(ts)], dtype=np.int64))

        pbar.set_postfix(frame=stem)


if __name__ == "__main__":
    main()
