import os
import math
import subprocess
from typing import List, Optional, Tuple, Deque
from collections import deque

import numpy as np
import imageio.v2 as imageio
import imageio_ffmpeg
from tqdm import tqdm

from painter.config.config import Config
from painter.io.files import (
    ensure_dir_for,
    unique_path,
    load_image_rgb01,
    save_image_rgb01,
    load_brushes,
)
from painter.imagemath.image_math import (
    hex_to_rgb01,
    rgb_to_gray01,
    simple_central_gradients,
    mse_per_pixel,
    mse_mean,
)
from painter.strokes.sizes import make_all_sizes
from painter.strokes.brush_ops import Brush, build_weight_mask
from painter.strokes.roi_selection import CooldownMap, select_roi_center


def overlap_mask_and_frame(cx: int, cy: int, mask_w: int, mask_h: int, W: int, H: int):
    """
    Compute overlap between a brush mask and the canvas frame.
    Returns indices for both canvas and mask, or None if fully outside.
    """
    mask_left = cx - mask_w // 2
    mask_top = cy - mask_h // 2
    mask_right = mask_left + mask_w
    mask_bottom = mask_top + mask_h

    X1 = max(0, mask_left)
    Y1 = max(0, mask_top)
    X2 = min(W, mask_right)
    Y2 = min(H, mask_bottom)
    if X1 >= X2 or Y1 >= Y2:
        return None

    MX1 = X1 - mask_left
    MY1 = Y1 - mask_top
    MX2 = MX1 + (X2 - X1)
    MY2 = MY1 + (Y2 - Y1)
    MX1 = max(0, min(MX1, mask_w))
    MX2 = max(0, min(MX2, mask_w))
    MY1 = max(0, min(MY1, mask_h))
    MY2 = max(0, min(MY2, mask_h))
    return (X1, X2, Y1, Y2, MX1, MX2, MY1, MY2)


def local_gradient_angle_deg(
    Ix: np.ndarray,
    Iy: np.ndarray,
    cx: int,
    cy: int,
    size: int,
    rng: np.random.Generator,
    min_strength: float,
    jitter_deg: float,
) -> float:
    """
    Estimate brush orientation from local image gradients.
    Uses sum of gradients within a size×size window; falls back to random if too weak.
    """
    H, W = Ix.shape
    half = max(1, size // 2)
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(W, cx + half)
    y2 = min(H, cy + half)

    sx = float(np.sum(Ix[y1:y2, x1:x2]))
    sy = float(np.sum(Iy[y1:y2, x1:x2]))
    strength = sx * sx + sy * sy

    if strength < min_strength:
        angle = float(rng.uniform(0.0, 180.0))
    else:
        phi = math.degrees(math.atan2(sy, sx))
        angle = phi + 90.0  # along isophotes

    if jitter_deg > 1e-6:
        angle += float(rng.normal(0.0, jitter_deg))
    while angle < 0.0:
        angle += 180.0
    while angle >= 180.0:
        angle -= 180.0
    return angle


def choose_color_from_target(roi_target: np.ndarray, weight_mask_2d: np.ndarray) -> np.ndarray:
    """
    Weighted average color inside ROI using the coverage mask.
    """
    s = float(np.sum(weight_mask_2d))
    if s <= 1e-8:
        return roi_target.mean(axis=(0, 1), dtype=np.float32)
    w3 = weight_mask_2d[..., None]
    num = (roi_target * w3).sum(axis=(0, 1))
    return (num / s).astype(np.float32)


def try_one_stroke(
    canvas: np.ndarray,
    target: np.ndarray,
    brush: Brush,
    x_center: int,
    y_center: int,
    block_size: int,
    angle_deg: float,
    use_soft_edges: bool,
    mask_threshold: float,
    use_alpha: bool,
    alpha_value: float,
) -> Tuple[bool, Optional[Tuple], float]:
    """
    Simulate and score a single stroke attempt at (x_center, y_center).
    Returns (accepted, payload, gain_per_painted_pixel).
    """
    H, W, _ = canvas.shape
    mask = brush.render(block_size, angle_deg)
    mh, mw = mask.shape
    if mh == 0 or mw == 0:
        return False, None, 0.0

    ov = overlap_mask_and_frame(x_center, y_center, mw, mh, W, H)
    if ov is None:
        return False, None, 0.0
    X1, X2, Y1, Y2, MX1, MX2, MY1, MY2 = ov

    roi_canvas = canvas[Y1:Y2, X1:X2]
    roi_target = target[Y1:Y2, X1:X2]
    mask_roi = mask[MY1:MY2, MX1:MX2]

    m, wmask = build_weight_mask(mask_roi, use_soft_edges, mask_threshold, use_alpha, alpha_value)
    if np.max(wmask) <= 0.0:
        return False, None, 0.0

    color = choose_color_from_target(roi_target, m)
    old = mse_mean(roi_canvas, roi_target)
    w3 = wmask[..., None]
    new_roi = w3 * color + (1.0 - w3) * roi_canvas
    new = mse_mean(new_roi, roi_target)

    area = (X2 - X1) * (Y2 - Y1)
    gain_total = max(0.0, (old - new) * area)
    gain_pp = gain_total / max(1, int(wmask.sum()))
    if new < old:
        payload = (X1, X2, Y1, Y2, new_roi, mask_roi)
        return True, payload, float(gain_pp)
    return False, None, 0.0


def commit_payload(canvas: np.ndarray, err_map: np.ndarray, target: np.ndarray, payload: Tuple) -> None:
    """
    Write accepted ROI into the canvas and update the error map accordingly.
    """
    X1, X2, Y1, Y2, new_roi, _mask_roi = payload
    roi_target = target[Y1:Y2, X1:X2]
    canvas[Y1:Y2, X1:X2] = new_roi
    err_map[Y1:Y2, X1:X2] = mse_per_pixel(new_roi, roi_target)


def _smoothstep(x: float) -> float:
    """S-curve easing in [0,1]."""
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def _get_duration_seconds(path: str) -> float:
    """Probe container metadata for duration (fallback: frames/fps)."""
    rdr = imageio.get_reader(path)
    try:
        meta = rdr.get_meta_data()
    finally:
        rdr.close()
    dur = meta.get("duration")
    if dur and dur > 0:
        return float(dur)
    fps = meta.get("fps")
    nframes = meta.get("nframes")
    if fps and nframes and fps > 0:
        return float(nframes) / float(fps)
    raise RuntimeError("Cannot determine video duration")


def _has_audio_track(path: str, ffmpeg_bin: str) -> bool:
    """Return True if the input video contains an audio stream."""
    proc = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-i", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return "Audio:" in proc.stderr


def _clamp_segments(total: float, slow_sec: float, fast_sec: float, margin: float = 0.02):
    """Ensure slow+fast <= total - margin; scale down if needed."""
    slow_sec = max(0.0, slow_sec)
    fast_sec = max(0.0, fast_sec)
    if slow_sec + fast_sec <= max(0.0, total - margin):
        return slow_sec, fast_sec
    if total <= margin:
        return 0.0, 0.0
    scale = (total - margin) / max(1e-9, (slow_sec + fast_sec))
    return slow_sec * scale, fast_sec * scale


def _atempo_chain_str(speed: float) -> str:
    """
    Build an ffmpeg atempo chain for an arbitrary speed factor by composing steps within [0.5, 2.0].
    """
    chain = []
    s = float(speed)
    if abs(s - 1.0) < 1e-6:
        return ""
    while s > 2.0 + 1e-9:
        chain.append("atempo=2.0")
        s /= 2.0
    while s < 0.5 - 1e-9:
        chain.append("atempo=0.5")
        s /= 0.5
    chain.append(f"atempo={s:.6f}")
    return ",".join(chain)


def _build_filter_complex(
    duration: float,
    slow_sec: float,
    fast_sec: float,
    steps: int,
    with_audio: bool,
    slow_from: float,
    fast_to: float,
):
    """
    Assemble ffmpeg filter_complex that ramps speed:
      slow_from→1.0 over the first slow_sec, 1.0 flat in the middle,
      1.0→fast_to over the last fast_sec.
    """
    vin, ain = "0:v", "0:a"
    parts_v, parts_a, lines = [], [], []

    def add_segment(t1, t2, speed, tag):
        v_label = f"[v{tag}]"
        lines.append(
            f"[{vin}]trim=start={t1:.6f}:end={t2:.6f},setpts=PTS-STARTPTS,setpts=PTS/{speed:.6f}{v_label}"
        )
        parts_v.append(v_label)
        if with_audio:
            a_label = f"[a{tag}]"
            chain = _atempo_chain_str(speed)
            if chain:
                lines.append(
                    f"[{ain}]atrim=start={t1:.6f}:end={t2:.6f},asetpts=PTS-STARTPTS,{chain}{a_label}"
                )
            else:
                lines.append(f"[{ain}]atrim=start={t1:.6f}:end={t2:.6f},asetpts=PTS-STARTPTS{a_label}")
            parts_a.append(a_label)

    if slow_sec > 1e-6:
        dt = slow_sec / steps
        for i in range(steps):
            t1 = i * dt
            t2 = (i + 1) * dt
            u = i / (steps - 1) if steps > 1 else 1.0
            s = slow_from + (1.0 - slow_from) * _smoothstep(u)
            add_segment(t1, t2, s, f"s{i}")

    mid_start, mid_end = slow_sec, max(slow_sec, duration - fast_sec)
    if mid_end - mid_start > 1e-6:
        v_mid = "[vMid]"
        lines.append(f"[{vin}]trim=start={mid_start:.6f}:end={mid_end:.6f},setpts=PTS-STARTPTS{v_mid}")
        parts_v.append(v_mid)
        if with_audio:
            a_mid = "[aMid]"
            lines.append(f"[{ain}]atrim=start={mid_start:.6f}:end={mid_end:.6f},asetpts=PTS-STARTPTS{a_mid}")
            parts_a.append(a_mid)

    if fast_sec > 1e-6:
        dt = fast_sec / steps
        base = duration - fast_sec
        for i in range(steps):
            t1 = base + i * dt
            t2 = base + (i + 1) * dt
            u = i / (steps - 1) if steps > 1 else 1.0
            s = 1.0 + (fast_to - 1.0) * _smoothstep(u)
            add_segment(t1, t2, s, f"f{i}")

    n = len(parts_v)
    if n == 0:
        raise RuntimeError("Empty segment set for concat")
    if with_audio:
        if len(parts_v) != len(parts_a):
            raise RuntimeError("Video/audio segments count mismatch")
        interleaved = []
        for v, a in zip(parts_v, parts_a):
            interleaved.extend([v, a])
        lines.append("".join(interleaved) + f"concat=n={n}:v=1:a=1[vout][aout]")
        return ";".join(lines), "[vout]", "[aout]"
    else:
        lines.append("".join(parts_v) + f"concat=n={n}:v=1:a=0[vout]")
        return ";".join(lines), "[vout]", None


def postprocess_video_speed_replace(
    input_path: str,
    slow_seconds: float,
    fast_seconds: float,
    steps: int,
    crf: int,
    preset: str,
    fps: Optional[float],
    slow_from: float,
    fast_to: float,
) -> str:
    """
    Apply a smooth speed ramp to the video in-place using a temporary file and rename.
    """
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    duration = _get_duration_seconds(input_path)
    slow_sec, fast_sec = _clamp_segments(duration, slow_seconds, fast_seconds)
    with_audio = _has_audio_track(input_path, ffmpeg_bin)

    filter_complex, vout, aout = _build_filter_complex(
        duration=duration,
        slow_sec=slow_sec,
        fast_sec=fast_sec,
        steps=steps,
        with_audio=with_audio,
        slow_from=slow_from,
        fast_to=fast_to,
    )

    base, ext = os.path.splitext(input_path)
    tmp_out = f"{base}.speedtmp{ext}"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-filter_complex",
        filter_complex,
        "-map",
        vout,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
    ]
    if fps:
        cmd += ["-r", str(fps)]
    if aout:
        cmd += ["-map", aout, "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", tmp_out]

    subprocess.run(cmd, check=True)

    os.replace(tmp_out, input_path)
    return input_path


def main(cfg: Config) -> None:
    """
    Orchestrate the painting pipeline:
    load inputs, iterate over size phases, emit optional video, postprocess speed.
    """
    rng = np.random.default_rng(cfg.seed)

    target = load_image_rgb01(cfg.input_image_path, cfg.max_size)
    H, W, _ = target.shape
    start_rgb = hex_to_rgb01(cfg.start_color_hex)
    canvas = np.ones_like(target, dtype=np.float32) * start_rgb
    brushes: List[Brush] = load_brushes(cfg.brush_paths)

    err_map = mse_per_pixel(canvas, target)
    cooldown = CooldownMap(
        shape=err_map.shape,
        enabled=cfg.use_cooldown,
        min_val=cfg.cooldown_min,
        max_val=cfg.cooldown_max,
        recover_factor=cfg.cooldown_recover,
    )

    sizes = make_all_sizes(cfg, W, H)
    current_level = 0
    current_size = sizes[current_level]

    if cfg.orientation_mode == "gradient":
        gray = rgb_to_gray01(target)
        Ix, Iy = simple_central_gradients(gray)
    else:
        Ix = Iy = None

    writer = None
    video_path_final = None
    if cfg.make_video:
        if cfg.save_every_n_strokes and cfg.save_every_n_strokes > 0:
            save_every = cfg.save_every_n_strokes
        else:
            total_frames = max(1, cfg.video_fps * (cfg.video_duration_sec or 10))
            save_every = max(1, math.ceil(cfg.total_strokes / total_frames))
        ensure_dir_for(cfg.output_video_path)
        video_path_final = unique_path(cfg.output_video_path)
        writer = imageio.get_writer(video_path_final, fps=cfg.video_fps)

    attempts_in_phase = 0
    accept_window: Deque[int] = deque(maxlen=cfg.phase_accept_window)

    def phase_max_attempts_for(size_px: int) -> int:
        approx = cfg.phase_max_attempts_factor * (H * W) / max(1, size_px * size_px)
        return max(cfg.phase_min_strokes, int(round(approx)))

    phase_max_attempts = phase_max_attempts_for(current_size)

    global_step = 0
    pbar = tqdm(total=cfg.total_strokes, desc="Painting", ncols=80)
    try:
        for _ in range(cfg.total_strokes):
            cooldown.recover()

            y, x = select_roi_center(
                err_map=err_map,
                sel_weight=cooldown.array,
                method=cfg.roi_sampling,
                topk=cfg.topk,
                rng=rng,
            )

            if cfg.orientation_mode == "gradient":
                angle = local_gradient_angle_deg(
                    Ix, Iy, x, y, current_size, rng, cfg.grad_min_strength, cfg.angle_jitter_deg
                )
            elif cfg.orientation_mode == "random":
                angle = float(rng.uniform(0.0, 180.0))
            else:
                angle = 0.0

            brush = brushes[int(rng.integers(0, len(brushes)))]
            accepted, payload, _gain_pp = try_one_stroke(
                canvas,
                target,
                brush,
                x,
                y,
                current_size,
                angle,
                cfg.use_soft_edges,
                cfg.mask_threshold,
                cfg.use_alpha,
                cfg.alpha_value,
            )
            attempts_in_phase += 1
            accept_window.append(1 if accepted else 0)

            if accepted and payload is not None:
                commit_payload(canvas, err_map, target, payload)
                cooldown.apply_after_payload(payload, cfg.cooldown_factor)

            global_step += 1
            pbar.update(1)

            if writer is not None and (global_step % save_every == 0):
                writer.append_data(np.clip(canvas * 255.0, 0, 255).astype(np.uint8))

            move_down = False
            if attempts_in_phase >= cfg.phase_min_strokes:
                if len(accept_window) >= min(cfg.phase_accept_window, cfg.phase_min_strokes):
                    accept_rate = sum(accept_window) / float(len(accept_window))
                    if accept_rate < cfg.phase_accept_threshold:
                        move_down = True
            if attempts_in_phase >= phase_max_attempts:
                move_down = True

            if move_down and current_level + 1 < len(sizes):
                current_level += 1
                current_size = sizes[current_level]
                attempts_in_phase = 0
                accept_window.clear()
                phase_max_attempts = phase_max_attempts_for(current_size)
                cooldown.reset()

            if cfg.target_mse is not None and float(np.mean(err_map)) <= cfg.target_mse:
                break
    finally:
        pbar.close()

    img_path = save_image_rgb01(cfg.output_final_image_path, canvas)

    if writer is not None and video_path_final:
        last = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
        for _ in range(max(0, cfg.record_last_hold_frames)):
            writer.append_data(last)
        writer.close()

        if cfg.postprocess_speed:
            try:
                postprocess_video_speed_replace(
                    input_path=video_path_final,
                    slow_seconds=cfg.speed_slow_seconds,
                    fast_seconds=cfg.speed_fast_seconds,
                    steps=cfg.speed_steps,
                    crf=cfg.speed_crf,
                    preset=cfg.speed_preset,
                    fps=cfg.speed_fps,
                    slow_from=cfg.speed_slow_from,
                    fast_to=cfg.speed_fast_to,
                )
            except Exception as e:
                print(f"[warn] speed postprocess failed: {e}")

    if writer is not None and video_path_final:
        print(f"Done. Image: {img_path} | Video: {video_path_final}")
    else:
        print(f"Done. Image: {img_path}")


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
