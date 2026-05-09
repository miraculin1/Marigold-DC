import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

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


def to_voxel_grid_numpy(events: np.ndarray, num_bins: int, width: int, height: int) -> np.ndarray:
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



def zbuffer_project_depth(
    pts_xyz: np.ndarray,
    k: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    z = pts_xyz[2]
    front = z > 0

    pts_xyz = pts_xyz[:, front]
    z = z[front]

    uv = k @ pts_xyz
    u = np.round(uv[0] / uv[2]).astype(np.int32)
    v = np.round(uv[1] / uv[2]).astype(np.int32)

    in_img = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
    u = u[in_img]
    v = v[in_img]
    z = z[in_img].astype(np.float32)

    depth = np.zeros((out_h, out_w), dtype=np.float32)
    zbuf = np.full((out_h, out_w), np.inf, dtype=np.float32)
    for uu, vv, zz in zip(u, v, z):
        if zz < zbuf[vv, uu]:
            zbuf[vv, uu] = zz
            depth[vv, uu] = zz
    return depth


def dense_depth_to_event_depth(
    depth_left: np.ndarray,
    k_left_rect: np.ndarray,
    k_event_rect: np.ndarray,
    t_01: np.ndarray,
    r_rect_event: np.ndarray,
    r_rect_image: np.ndarray,
    event_h: int,
    event_w: int,
) -> np.ndarray:
    h, w = depth_left.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    z = depth_left.reshape(-1)
    x = xs.reshape(-1)
    y = ys.reshape(-1)

    valid = z > 0
    z = z[valid]
    x = x[valid]
    y = y[valid]

    fx_l, fy_l = float(k_left_rect[0, 0]), float(k_left_rect[1, 1])
    cx_l, cy_l = float(k_left_rect[0, 2]), float(k_left_rect[1, 2])

    x_l = (x - cx_l) * z / fx_l
    y_l = (y - cy_l) * z / fy_l

    pts_left_rect = np.stack([x_l, y_l, z], axis=0)
    pts_left = r_rect_image.T @ pts_left_rect
    pts_left = np.vstack([pts_left, np.ones((1, z.shape[0]), dtype=pts_left.dtype)])

    pts_event_rect = r_rect_event @ (t_01 @ pts_left)[:3, :]

    return zbuffer_project_depth(pts_event_rect, k_event_rect, out_h=event_h, out_w=event_w)


def disparity_to_sparse_depth(
    disparity: np.ndarray, fx: float, baseline: float, disparity_scale: float
) -> np.ndarray:
    disparity = disparity.astype(np.float32) / disparity_scale
    sparse_depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > 0
    sparse_depth[valid] = (fx * baseline) / disparity[valid]
    return sparse_depth


def make_marigold_pipe(
    checkpoint: str, use_full_precision: bool
) -> MarigoldDepthCompletionPipeline:
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
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    return pipe


def rectify_events_xy(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    rectify_map: np.ndarray,
    width: int,
    height: int,
):
    x_raw_i = x_raw.astype(np.int64)
    y_raw_i = y_raw.astype(np.int64)

    in_raw = (
        (x_raw_i >= 0)
        & (x_raw_i < rectify_map.shape[1])
        & (y_raw_i >= 0)
        & (y_raw_i < rectify_map.shape[0])
    )

    x_raw_i = x_raw_i[in_raw]
    y_raw_i = y_raw_i[in_raw]
    t_sel = t[in_raw]
    p_sel = p[in_raw]

    mapped = rectify_map[y_raw_i, x_raw_i]
    x_rect = np.round(mapped[:, 0]).astype(np.int32)
    y_rect = np.round(mapped[:, 1]).astype(np.int32)

    in_rect = (x_rect >= 0) & (x_rect < width) & (y_rect >= 0) & (y_rect < height)
    return x_rect[in_rect], y_rect[in_rect], t_sel[in_rect], p_sel[in_rect]


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


def load_rectify_map(rectify_map_h5: Path) -> np.ndarray:
    with h5py.File(str(rectify_map_h5), "r") as f:
        return f["rectify_map"][:].astype(np.float32)


@dataclass
class CalibrationBundle:
    k_left_rect: np.ndarray
    k_event_rect: np.ndarray
    t_01: np.ndarray
    r_rect_event: np.ndarray
    r_rect_image: np.ndarray
    baseline_m: float


def parse_cam_matrix(camera_matrix_4: list) -> np.ndarray:
    fx, fy, cx, cy = [float(v) for v in camera_matrix_4]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_dsec_calibration(cam_to_cam_yaml: Path) -> CalibrationBundle:
    with open(cam_to_cam_yaml, "r", encoding="utf-8") as f:
        calib = yaml.safe_load(f)

    k_left_rect = parse_cam_matrix(calib["intrinsics"]["camRect1"]["camera_matrix"])
    k_event_rect = parse_cam_matrix(calib["intrinsics"]["camRect0"]["camera_matrix"])

    t_10 = np.array(calib["extrinsics"]["T_10"], dtype=np.float64)
    t_01 = np.linalg.inv(t_10)

    r_rect_event = np.array(calib["extrinsics"]["R_rect0"], dtype=np.float64)
    r_rect_image = np.array(calib["extrinsics"]["R_rect1"], dtype=np.float64)

    cams_03 = np.array(calib["disparity_to_depth"]["cams_03"], dtype=np.float64)
    baseline_m = 1.0 / float(cams_03[3, 2])

    return CalibrationBundle(
        k_left_rect=k_left_rect,
        k_event_rect=k_event_rect,
        t_01=t_01,
        r_rect_event=r_rect_event,
        r_rect_image=r_rect_image,
        baseline_m=baseline_m,
    )


def main():
    parser = argparse.ArgumentParser(
        "DSEC preprocess: Marigold depth + projection + rectified event voxel"
    )
    parser.add_argument("--left_rgb_dir", type=Path, required=True)
    parser.add_argument("--disparity_dir", type=Path, required=True)
    parser.add_argument("--events_h5", type=Path, required=True)
    parser.add_argument("--rectify_map_h5", type=Path, required=True)
    parser.add_argument("--rgb_timestamps_ns", type=Path, required=True)
    parser.add_argument("--disparity_timestamps_ns", type=Path, required=True)
    parser.add_argument("--cam_to_cam_yaml", type=Path, required=True)
    parser.add_argument("--disparity_scale", type=float, default=256.0)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default="prs-eth/marigold-depth-v1-0")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--ensemble_size", type=int, default=1)
    parser.add_argument("--processing_resolution", type=int, default=768)
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--event_width", type=int, required=True)
    parser.add_argument("--event_height", type=int, required=True)
    parser.add_argument("--use_full_precision", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "depth_event").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "voxel").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "depth_left_dense").mkdir(parents=True, exist_ok=True)

    # index data
    calib = load_dsec_calibration(args.cam_to_cam_yaml)

    ts_rgb = np.loadtxt(str(args.rgb_timestamps_ns), dtype=np.int64).reshape(-1)
    ts_disp = np.loadtxt(str(args.disparity_timestamps_ns), dtype=np.int64).reshape(-1)
    rgb_files = sorted(args.left_rgb_dir.glob("*.png"))
    disp_files = sorted(args.disparity_dir.glob("*.png"))

    rgb_by_index = {int(p.stem): p for p in rgb_files}
    ts_rgb_by_index = {idx: int(ts) for idx, ts in enumerate(ts_rgb)}

    # rectify event
    x_raw, y_raw, t_raw, p_raw = load_events(args.events_h5)
    rectify_map = load_rectify_map(args.rectify_map_h5)
    x_evt, y_evt, t_evt, p_evt = rectify_events_xy(
        x_raw=x_raw,
        y_raw=y_raw,
        t=t_raw,
        p=p_raw,
        rectify_map=rectify_map,
        width=args.event_width,
        height=args.event_height,
    )

    # prepare marigold
    pipe = make_marigold_pipe(args.checkpoint, args.use_full_precision)
    total_frames = len(disp_files)
    fx_left = float(calib.k_left_rect[0, 0])
    pbar = tqdm(
        zip(disp_files, ts_disp),
        total=total_frames,
        desc="Preprocessing",
        dynamic_ncols=True,
    )

    # main loop
    for i, (disp_path, ts) in enumerate(pbar):
        disp_idx = int(disp_path.stem)
        rgb_path = rgb_by_index[disp_idx]
        rgb = Image.open(str(rgb_path)).convert("RGB")
        disparity = np.asarray(Image.open(str(disp_path)))
        sparse_depth = disparity_to_sparse_depth(
            disparity, fx_left, calib.baseline_m, args.disparity_scale
        )

        dense_depth = pipe(
            image=rgb,
            sparse_depth=sparse_depth,
            num_inference_steps=args.num_inference_steps,
            ensemble_size=args.ensemble_size,
            processing_resolution=args.processing_resolution,
        ).astype(np.float32)

        depth_event = dense_depth_to_event_depth(
            dense_depth,
            k_left_rect=calib.k_left_rect,
            k_event_rect=calib.k_event_rect,
            t_01=calib.t_01,
            r_rect_event=calib.r_rect_event,
            r_rect_image=calib.r_rect_image,
            event_h=args.event_height,
            event_w=args.event_width,
        )

        stem = disp_path.stem
        np.save(str(args.output_dir / "depth_left_dense" / f"{stem}.npy"), dense_depth)
        np.save(str(args.output_dir / "depth_event" / f"{stem}.npy"), depth_event)

        t0, t1 = int(ts_disp[i]), int(ts_disp[i] + 100000)
        voxel = event_slice_to_voxel(
            x=x_evt,
            y=y_evt,
            t=t_evt,
            p=p_evt,
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
                **hdf5plugin.Blosc(
                    cname="zstd", clevel=4, shuffle=hdf5plugin.Blosc.SHUFFLE
                ),
            )
            f.create_dataset("t0_ns", data=np.array([t0], dtype=np.int64))
            f.create_dataset("t1_ns", data=np.array([t1], dtype=np.int64))
            f.create_dataset("t_disp_ns", data=np.array([int(ts)], dtype=np.int64))

        pbar.set_postfix(frame=stem, ts_rgb=ts_rgb_by_index.get(disp_idx, -1))


if __name__ == "__main__":
    main()
