import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import yaml
from tqdm import tqdm


def parse_cam_matrix(camera_matrix_4: list) -> np.ndarray:
    fx, fy, cx, cy = [float(v) for v in camera_matrix_4]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_dsec_calibration(cam_to_cam_yaml: Path):
    with open(cam_to_cam_yaml, "r", encoding="utf-8") as f:
        calib = yaml.safe_load(f)

    k_event = parse_cam_matrix(calib["intrinsics"]["camRect0"]["camera_matrix"])
    k_rgb = parse_cam_matrix(calib["intrinsics"]["camRect1"]["camera_matrix"])

    # left-frame camera <- left-event camera
    r_rect0 = np.array(calib["extrinsics"]["R_rect0"])
    r_rect1 = np.array(calib["extrinsics"]["R_rect1"])
    t_10 = np.array(calib["extrinsics"]["T_10"], dtype=np.float64)

    cams_03 = np.array(calib["disparity_to_depth"]["cams_03"], dtype=np.float64)
    denom_event = float(cams_03[3, 2])
    if abs(denom_event) < 1e-12:
        raise ValueError("Invalid cams_03 for baseline computation: denominator is zero")
    baseline_event_m = 1.0 / denom_event

    return k_event, k_rgb, t_10, r_rect0, r_rect1, baseline_event_m


def colorize_disparity(disparity: np.ndarray, percentile: float) -> np.ndarray:
    disp = disparity.astype(np.float32)
    valid = disp > 0
    norm = np.zeros_like(disp, dtype=np.float32)
    if np.any(valid):
        vals = disp[valid]
        vmax = float(np.percentile(vals, percentile))
        if vmax <= 0:
            vmax = 1.0
        norm[valid] = np.clip(vals / vmax, 0.0, 1.0)
    disp_u8 = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(disp_u8, cv2.COLORMAP_TURBO)


def rectify_disparity_from_raw(raw_disparity: np.ndarray, rectify_map: np.ndarray) -> np.ndarray:
    """Rectify raw event disparity using raw->rectified map (nearest splat)."""
    h_rect, w_rect = rectify_map.shape[:2]
    out = np.zeros((h_rect, w_rect), dtype=np.float32)

    ys, xs = np.where(raw_disparity > 0)
    if ys.size == 0:
        return out

    disp_vals = raw_disparity[ys, xs].astype(np.float32)
    map_xy = rectify_map[ys, xs]  # [x_rect, y_rect]
    u = np.round(map_xy[:, 0]).astype(np.int32)
    v = np.round(map_xy[:, 1]).astype(np.int32)

    valid = (u >= 0) & (u < w_rect) & (v >= 0) & (v < h_rect)
    u = u[valid]
    v = v[valid]
    disp_vals = disp_vals[valid]

    # Keep larger disparity when multiple raw pixels map to same rectified pixel.
    for uu, vv, dd in zip(u, v, disp_vals):
        if dd > out[vv, uu]:
            out[vv, uu] = dd
    return out


def project_event_disparity_to_rgb(
    disparity_event: np.ndarray,
    k_event: np.ndarray,
    k_rgb: np.ndarray,
    t_10: np.ndarray,
    r_rect0: np.ndarray,
    r_rect1: np.array,
    baseline_event_m: float,
    disparity_scale: float,
    rgb_width: int,
    rgb_height: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    disp_event = disparity_event.astype(np.float32) / float(disparity_scale)
    valid = disp_event > 0
    source_valid_px = int(np.count_nonzero(valid))
    if source_valid_px == 0:
        return np.zeros((rgb_height, rgb_width), dtype=np.float32), {
            "source_valid_px": 0,
            "in_front_px": 0,
            "in_image_px": 0,
            "proj_valid_px": 0,
        }

    ys, xs = np.where(valid)
    dvals = disp_event[ys, xs]

    # Step 1: disparity -> depth in event frame.
    fx_e = float(k_event[0, 0])
    fy_e = float(k_event[1, 1])
    cx_e = float(k_event[0, 2])
    cy_e = float(k_event[1, 2])
    z_event = (fx_e * float(baseline_event_m)) / dvals

    # Step 2: backproject to 3D event camera points.
    x_event = (xs.astype(np.float64) - cx_e) * z_event / fx_e
    y_event = (ys.astype(np.float64) - cy_e) * z_event / fy_e
    pts_event_rect = np.stack([x_event, y_event, z_event])
    pts_event = r_rect0.T @ pts_event_rect

    assert pts_event.ndim == 2 and pts_event.shape[0] == 3
    assert z_event.ndim == 1 and pts_event.shape[1] == z_event.shape[0]
    pts_event = np.vstack(
        [pts_event, np.ones((1, z_event.shape[0]), dtype=pts_event.dtype)]
    )

    # Step 3: transform to left-image camera and project.
    pts_left = (t_10 @ pts_event)[:3, :]
    pts_left_rect = r_rect1 @ pts_left
    z_left = pts_left_rect[2]
    front = z_left > 0
    in_front_px = int(np.count_nonzero(front))
    if in_front_px == 0:
        return np.zeros((rgb_height, rgb_width), dtype=np.float32), {
            "source_valid_px": source_valid_px,
            "in_front_px": 0,
            "in_image_px": 0,
            "proj_valid_px": 0,
        }

    pts_left_rect = pts_left_rect[:, front]
    z_left = z_left[front]

    uv = k_rgb @ pts_left_rect
    u = np.round(uv[0] / uv[2]).astype(np.int32)
    v = np.round(uv[1] / uv[2]).astype(np.int32)

    in_img = (u >= 0) & (u < rgb_width) & (v >= 0) & (v < rgb_height)
    in_image_px = int(np.count_nonzero(in_img))
    if in_image_px == 0:
        return np.zeros((rgb_height, rgb_width), dtype=np.float32), {
            "source_valid_px": source_valid_px,
            "in_front_px": in_front_px,
            "in_image_px": 0,
            "proj_valid_px": 0,
        }

    u = u[in_img]
    v = v[in_img]
    z_left = z_left[in_img].astype(np.float32)

    # z-buffer in RGB plane.
    depth_left = np.zeros((rgb_height, rgb_width), dtype=np.float32)
    zbuf = np.full((rgb_height, rgb_width), np.inf, dtype=np.float32)
    for uu, vv, zz in zip(u, v, z_left):
        if zz < zbuf[vv, uu]:
            zbuf[vv, uu] = zz
            depth_left[vv, uu] = zz

    disp_on_rgb = np.zeros_like(depth_left, dtype=np.float32)
    valid_depth = depth_left > 0
    proj_valid_px = int(np.count_nonzero(valid_depth))
    if proj_valid_px > 0:
        fx_rgb = float(k_rgb[0, 0])
        disp_on_rgb[valid_depth] = (fx_rgb * float(baseline_event_m)) / depth_left[valid_depth]

    return disp_on_rgb, {
        "source_valid_px": source_valid_px,
        "in_front_px": in_front_px,
        "in_image_px": in_image_px,
        "proj_valid_px": proj_valid_px,
    }


def infer_default_paths(
    sequence_dir: Path,
    disp_event_dir: Optional[Path],
    rgb_dir: Optional[Path],
    cam_to_cam_yaml: Optional[Path],
    disp_ts: Optional[Path],
    rgb_ts: Optional[Path],
    rectify_map_h5: Optional[Path],
):
    seq_name = sequence_dir.name
    disp_event_dir = disp_event_dir or (sequence_dir / "disparity_event")
    rgb_dir = rgb_dir or (sequence_dir / "images_rectified_left")
    cam_to_cam_yaml = cam_to_cam_yaml or (sequence_dir / "calibration" / "cam_to_cam.yaml")
    disp_ts = disp_ts or (sequence_dir / f"{seq_name}_disparity_timestamps.txt")
    rgb_ts = rgb_ts or (sequence_dir / f"{seq_name}_image_timestamps.txt")
    rectify_map_h5 = rectify_map_h5 or (sequence_dir / "events_left" / "rectify_map.h5")
    return disp_event_dir, rgb_dir, cam_to_cam_yaml, disp_ts, rgb_ts, rectify_map_h5


def build_stem_map(img_dir: Path) -> Dict[str, Path]:
    return {p.stem: p for p in sorted(img_dir.glob("*.png"))}


def build_timestamp_lookup(
    disp_stems: list,
    disp_ts: Optional[np.ndarray],
    rgb_ts: Optional[np.ndarray],
) -> Dict[str, Tuple[int, int, int]]:
    out: Dict[str, Tuple[int, int, int]] = {}
    if disp_ts is None or rgb_ts is None:
        return out

    n = min(len(disp_stems), len(disp_ts))
    for i in range(n):
        stem = disp_stems[i]
        rgb_idx = int(stem)
        if rgb_idx < 0 or rgb_idx >= len(rgb_ts):
            continue
        ts_disp = int(disp_ts[i])
        ts_rgb = int(rgb_ts[rgb_idx])
        out[stem] = (ts_disp, ts_rgb, ts_rgb - ts_disp)
    return out


def main():
    parser = argparse.ArgumentParser("Project DSEC disparity_event to RGB and save overlay")
    parser.add_argument("--sequence_dir", type=Path, required=True)
    parser.add_argument("--disp_event_dir", type=Path, default=None)
    parser.add_argument("--rgb_dir", type=Path, default=None)
    parser.add_argument("--cam_to_cam_yaml", type=Path, default=None)
    parser.add_argument("--disp_ts", type=Path, default=None)
    parser.add_argument("--rgb_ts", type=Path, default=None)
    parser.add_argument("--rectify_map_h5", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--disp_scale", type=float, default=256.0)
    parser.add_argument("--disp_percentile", type=float, default=99.0)
    parser.add_argument("--viz_every", type=int, default=1)
    parser.add_argument("--coord_mode", type=str, default="rectified", choices=["rectified", "raw"])
    args = parser.parse_args()

    disp_event_dir, rgb_dir, cam_to_cam_yaml, disp_ts_path, rgb_ts_path, rectify_map_h5 = infer_default_paths(
        sequence_dir=args.sequence_dir,
        disp_event_dir=args.disp_event_dir,
        rgb_dir=args.rgb_dir,
        cam_to_cam_yaml=args.cam_to_cam_yaml,
        disp_ts=args.disp_ts,
        rgb_ts=args.rgb_ts,
        rectify_map_h5=args.rectify_map_h5,
    )
    output_dir = args.output_dir or (args.sequence_dir / "viz_disp_event_on_rgb_overlay")
    output_dir.mkdir(parents=True, exist_ok=True)

    for path_, name in [
        (disp_event_dir, "Event disparity directory"),
        (rgb_dir, "RGB directory"),
        (cam_to_cam_yaml, "Calibration file"),
    ]:
        if not path_.exists():
            raise FileNotFoundError(f"{name} not found: {path_}")

    rectify_map = None
    if args.coord_mode == "raw":
        if not rectify_map_h5.exists():
            raise FileNotFoundError(f"Rectify map file not found for raw mode: {rectify_map_h5}")
        with h5py.File(str(rectify_map_h5), "r") as f:
            rectify_map = f["rectify_map"][:].astype(np.float32)

    k_event, k_rgb, t_10, r_rect0, r_rect1, baseline_event_m = load_dsec_calibration(cam_to_cam_yaml)

    disp_map = build_stem_map(disp_event_dir)
    rgb_map = build_stem_map(rgb_dir)
    common_stems = sorted(set(disp_map.keys()) & set(rgb_map.keys()))
    if not common_stems:
        raise RuntimeError("No matching stems between disparity_event and RGB images")

    disp_ts: Optional[np.ndarray] = None
    rgb_ts: Optional[np.ndarray] = None
    if disp_ts_path.exists() and rgb_ts_path.exists():
        disp_ts = np.loadtxt(str(disp_ts_path), dtype=np.int64).reshape(-1)
        rgb_ts = np.loadtxt(str(rgb_ts_path), dtype=np.int64).reshape(-1)
    else:
        print(
            f"[WARN] Timestamp file missing, skip timestamp delta display. "
            f"disp_ts={disp_ts_path.exists()} rgb_ts={rgb_ts_path.exists()}"
        )

    ts_lookup = build_timestamp_lookup(common_stems, disp_ts, rgb_ts)
    delta_list = []

    pbar = tqdm(common_stems, desc="Project+Overlay", dynamic_ncols=True)
    for i, stem in enumerate(pbar):
        if args.viz_every > 1 and (i % args.viz_every != 0):
            continue

        disp = cv2.imread(str(disp_map[stem]), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(rgb_map[stem]), cv2.IMREAD_COLOR)
        if disp is None or rgb is None:
            tqdm.write(f"[WARN] skip unreadable pair: {stem}")
            continue

        if args.coord_mode == "raw":
            disp_event = rectify_disparity_from_raw(disp.astype(np.float32), rectify_map)
            disp_event = (disp_event + 0.5).astype(np.uint16)
        else:
            disp_event = disp

        h_rgb, w_rgb = rgb.shape[:2]
        disp_on_rgb, diag = project_event_disparity_to_rgb(
            disparity_event=disp_event,
            k_event=k_event,
            k_rgb=k_rgb,
            t_10=t_10,
            r_rect0=r_rect0,
            r_rect1=r_rect1,
            baseline_event_m=baseline_event_m,
            disparity_scale=float(args.disp_scale),
            rgb_width=w_rgb,
            rgb_height=h_rgb,
        )

        disp_color = colorize_disparity(disp_on_rgb, percentile=float(args.disp_percentile))
        overlay = cv2.addWeighted(rgb, 1.0 - float(args.alpha), disp_color, float(args.alpha), 0.0)

        coverage = 0.0
        if diag["source_valid_px"] > 0:
            coverage = float(diag["proj_valid_px"]) / float(diag["source_valid_px"])

        ts_text = "delta_ns=NA"
        if stem in ts_lookup:
            ts_disp, ts_rgb, delta_ns = ts_lookup[stem]
            delta_list.append(delta_ns)
            ts_text = f"ts_disp={ts_disp} ts_rgb={ts_rgb} delta_ns={delta_ns}"

        text = (
            f"{stem} src={diag['source_valid_px']} front={diag['in_front_px']} "
            f"in_img={diag['in_image_px']} proj={diag['proj_valid_px']} cov={coverage:.3f} {ts_text}"
        )
        cv2.putText(overlay, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imwrite(str(output_dir / f"{stem}.png"), overlay)
        pbar.set_postfix(frame=stem, proj=diag["proj_valid_px"], cov=f"{coverage:.3f}")

    print(f"Done. Saved overlay images to: {output_dir}")
    if delta_list:
        delta_arr = np.asarray(delta_list, dtype=np.int64)
        abs_delta = np.abs(delta_arr)
        print(
            "[TS] delta_ns stats: "
            f"min={int(delta_arr.min())} max={int(delta_arr.max())} "
            f"mean={float(delta_arr.mean()):.3f} median={float(np.median(delta_arr)):.3f} "
            f"p90_abs={float(np.percentile(abs_delta, 90)):.3f} "
            f"p99_abs={float(np.percentile(abs_delta, 99)):.3f}"
        )


if __name__ == "__main__":
    main()
