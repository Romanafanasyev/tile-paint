import os
import math
import subprocess
from typing import List, Optional, Tuple, Deque
from collections import deque

import numpy as np
from PIL import Image
import imageio.v2 as imageio
import imageio_ffmpeg
from tqdm import tqdm

from painter.config import Config


# -------------------------
# File/path utilities
# -------------------------
def ensure_dir_for(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def unique_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    candidate = path
    k = 1
    while os.path.exists(candidate):
        candidate = f"{base}_{k}{ext}"
        k += 1
    return candidate


# -------------------------
# Image I/O and conversions
# -------------------------
def hex_to_rgb01(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return np.array([r, g, b], dtype=np.float32)


def load_image_rgb01(path: str, max_size: int = 0) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if max_size and max(im.size) > max_size:
        w, h = im.size
        if w >= h:
            new_w = max_size
            new_h = int(round(h * (max_size / w)))
        else:
            new_h = max_size
            new_w = int(round(w * (max_size / h)))
        im = im.resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def save_image_rgb01(path: str, img: np.ndarray) -> str:
    ensure_dir_for(path)
    path2 = unique_path(path)
    img8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img8, mode="RGB").save(path2)
    return path2


def load_brushes_gray01(paths: List[str]) -> List[np.ndarray]:
    out = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Brush file not found: {p}")
        arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        out.append(arr)
    if not out:
        raise RuntimeError("No brush masks loaded. Check Config.brush_paths.")
    return out


# -------------------------
# Metrics / error map
# -------------------------
def mse_per_pixel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return np.mean(d * d, axis=2)


def mse_mean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


# -------------------------
# Gradients / orientation
# -------------------------
def rgb_to_gray01(img: np.ndarray) -> np.ndarray:
    # ITU-R BT.709 luma
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)


def simple_central_gradients(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Central-difference gradients with clamped edges.
    Returns (Ix, Iy) where Ix is horizontal gradient and Iy is vertical gradient.
    """
    H, W = gray.shape
    Ix = 0.5 * (np.roll(gray, -1, axis=1) - np.roll(gray, 1, axis=1))
    Iy = 0.5 * (np.roll(gray, -1, axis=0) - np.roll(gray, 1, axis=0))
    # edge handling (first/last rows/cols)
    Ix[:, 0] = gray[:, 1] - gray[:, 0]
    Ix[:, -1] = gray[:, -1] - gray[:, -2]
    Iy[0, :] = gray[1, :] - gray[0, :]
    Iy[-1, :] = gray[-1, :] - gray[-2, :]
    return Ix, Iy


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
    Estimate stroke orientation from gradients in a size×size window around (cx, cy).
    If gradients are weak, fall back to a random angle. Align strokes along isophotes.
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
        phi = math.degrees(math.atan2(sy, sx))   # gradient direction
        angle = phi + 90.0                       # along isophotes

    if jitter_deg > 1e-6:
        angle += float(rng.normal(0.0, jitter_deg))
    while angle < 0.0:
        angle += 180.0
    while angle >= 180.0:
        angle -= 180.0
    return angle


# -------------------------
# Brush mask: scale / rotate
# -------------------------
def scale_and_rotate_mask(mask01: np.ndarray, target_box_size: int, angle_deg: float) -> np.ndarray:
    Hb, Wb = mask01.shape
    scale = target_box_size / float(max(Hb, Wb))
    new_h = max(1, int(round(Hb * scale)))
    new_w = max(1, int(round(Wb * scale)))
    img = Image.fromarray((mask01 * 255.0).astype(np.uint8))
    img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    if abs(angle_deg) > 1e-6:
        img = img.rotate(angle_deg, resample=Image.BILINEAR, expand=True, fillcolor=0)
    return np.asarray(img, dtype=np.float32) / 255.0


# -------------------------
# Overlap of mask and frame
# -------------------------
def overlap_mask_and_frame(cx: int, cy: int, mask_w: int, mask_h: int, W: int, H: int):
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


# -------------------------
# Stroke color / blend / evaluation
# -------------------------
def choose_color_from_target(roi_target: np.ndarray, weight_mask_2d: np.ndarray) -> np.ndarray:
    s = float(np.sum(weight_mask_2d))
    if s <= 1e-8:
        return roi_target.mean(axis=(0, 1), dtype=np.float32)
    w3 = weight_mask_2d[..., None]
    num = (roi_target * w3).sum(axis=(0, 1))
    return (num / s).astype(np.float32)


def build_weight_mask(
    mask_roi: np.ndarray,
    use_soft_edges: bool,
    mask_threshold: float,
    use_alpha: bool,
    alpha_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if use_soft_edges:
        m = np.clip(mask_roi, 0.0, 1.0).astype(np.float32)
    else:
        m = (mask_roi >= mask_threshold).astype(np.float32)
    wmask = np.clip(alpha_value * m, 0.0, 1.0) if use_alpha else m
    return m, wmask


def try_one_stroke(
    canvas: np.ndarray,
    target: np.ndarray,
    brush: np.ndarray,
    x_center: int,
    y_center: int,
    block_size: int,
    angle_deg: float,
    use_soft_edges: bool,
    mask_threshold: float,
    use_alpha: bool,
    alpha_value: float,
) -> Tuple[bool, Optional[Tuple], float]:
    H, W, _ = canvas.shape
    mask = scale_and_rotate_mask(brush, block_size, angle_deg)
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


def commit_payload(canvas: np.ndarray, err_map: np.ndarray, target: np.ndarray, payload: Tuple):
    X1, X2, Y1, Y2, new_roi, _mask_roi = payload
    roi_target = target[Y1:Y2, X1:X2]
    canvas[Y1:Y2, X1:X2] = new_roi
    err_map[Y1:Y2, X1:X2] = mse_per_pixel(new_roi, roi_target)


# -------------------------
# ROI selection (+ cooldown)
# -------------------------
def select_roi_center(
    err_map: np.ndarray,
    sel_weight: Optional[np.ndarray],
    method: str,
    topk: int,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    weighted = err_map if sel_weight is None else err_map * sel_weight
    H, W = weighted.shape
    if method == "argmax":
        idx = int(np.argmax(weighted))
        y, x = np.unravel_index(idx, (H, W))
        return int(y), int(x)
    elif method == "topk_random":
        flat = weighted.ravel()
        k = min(topk, flat.size)
        part = np.argpartition(flat, -k)[-k:]
        idx = int(part[int(rng.integers(0, k))])
        y, x = np.unravel_index(idx, (H, W))
        return int(y), int(x)
    else:
        raise ValueError(f"Unknown roi_sampling: {method}")


def update_cooldown(sel_weight: Optional[np.ndarray], payload: Tuple, cfg: Config):
    if sel_weight is None or not cfg.use_cooldown:
        return
    X1, X2, Y1, Y2, _new_roi, mask_roi = payload
    m = (mask_roi > 0).astype(np.float32)
    block = sel_weight[Y1:Y2, X1:X2]
    block *= (1.0 - (1.0 - cfg.cooldown_factor) * m)
    np.clip(block, cfg.cooldown_min, cfg.cooldown_max, out=block)
    sel_weight[Y1:Y2, X1:X2] = block


def recover_cooldown(sel_weight: Optional[np.ndarray], cfg: Config):
    if sel_weight is None or not cfg.use_cooldown:
        return
    np.multiply(sel_weight, cfg.cooldown_recover, out=sel_weight)
    np.clip(sel_weight, cfg.cooldown_min, cfg.cooldown_max, out=sel_weight)


# -------------------------
# Size ladder (strict large→small)
# -------------------------
def make_all_sizes(cfg: Config, W: int, H: int) -> List[int]:
    s_max = max(cfg.smallest_px, int(round(min(W, H) * cfg.largest_frac)))
    s_min = max(1, int(cfg.smallest_px))
    L = max(2, int(cfg.levels))
    if cfg.size_scale_mode == "linear":
        step = (s_max - s_min) / float(L - 1)
        sizes = [max(1, int(round(s_max - i * step))) for i in range(L)]
    else:
        sizes = [max(1, int(round(s_max * (s_min / s_max) ** (i / (L - 1))))) for i in range(L)]
    sizes = sorted(set(sizes), reverse=True)
    return [max(s_min, s) for s in sizes]


# -------------------------
# Video postprocess: smooth speed ramp (in-place replace)
# -------------------------
def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def _get_duration_seconds(path: str) -> float:
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
    proc = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-i", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return "Audio:" in proc.stderr


def _clamp_segments(total: float, slow_sec: float, fast_sec: float, margin: float = 0.02):
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
    Build an atempo chain for arbitrary speed factor
    (ffmpeg supports 0.5..2.0 per filter, so we compose).
    """
    chain: List[str] = []
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

    # ramp-in slow_from → 1.0
    if slow_sec > 1e-6:
        dt = slow_sec / steps
        for i in range(steps):
            t1 = i * dt
            t2 = (i + 1) * dt
            u = i / (steps - 1) if steps > 1 else 1.0
            s = slow_from + (1.0 - slow_from) * _smoothstep(u)
            add_segment(t1, t2, s, f"s{i}")

    # mid 1.0x
    mid_start, mid_end = slow_sec, max(slow_sec, duration - fast_sec)
    if mid_end - mid_start > 1e-6:
        v_mid = "[vMid]"
        lines.append(f"[{vin}]trim=start={mid_start:.6f}:end={mid_end:.6f},setpts=PTS-STARTPTS{v_mid}")
        parts_v.append(v_mid)
        if with_audio:
            a_mid = "[aMid]"
            lines.append(f"[{ain}]atrim=start={mid_start:.6f}:end={mid_end:.6f},asetpts=PTS-STARTPTS{a_mid}")
            parts_a.append(a_mid)

    # ramp-out 1.0 → fast_to
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
    """Apply a smooth speed ramp and replace the original file in-place."""
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


# -------------------------
# Main pipeline
# -------------------------
def main(cfg: Config) -> None:
    rng = np.random.default_rng(cfg.seed)

    # Load target and initialize canvas
    target = load_image_rgb01(cfg.input_image_path, cfg.max_size)
    H, W, _ = target.shape
    start_rgb = hex_to_rgb01(cfg.start_color_hex)
    canvas = np.ones_like(target, dtype=np.float32) * start_rgb
    brushes = load_brushes_gray01(cfg.brush_paths)

    # Error map and selection weights
    err_map = mse_per_pixel(canvas, target)
    sel_weight = np.ones_like(err_map, dtype=np.float32) if cfg.use_cooldown else None

    # Size schedule (strict phases)
    sizes = make_all_sizes(cfg, W, H)
    current_level = 0
    current_size = sizes[current_level]

    # Orientation precomputation
    if cfg.orientation_mode == "gradient":
        gray = rgb_to_gray01(target)
        Ix, Iy = simple_central_gradients(gray)
    else:
        Ix = Iy = None

    # Video writer setup
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
            recover_cooldown(sel_weight, cfg)

            # Pick center
            y, x = select_roi_center(err_map, sel_weight, cfg.roi_sampling, cfg.topk, rng)

            # Angle at current size
            if cfg.orientation_mode == "gradient":
                angle = local_gradient_angle_deg(
                    Ix, Iy, x, y, current_size, rng, cfg.grad_min_strength, cfg.angle_jitter_deg
                )
            elif cfg.orientation_mode == "random":
                angle = float(rng.uniform(0.0, 180.0))
            else:
                angle = 0.0

            # Try a stroke
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
                update_cooldown(sel_weight, payload, cfg)

            global_step += 1
            pbar.update(1)

            # Write frame
            if writer is not None and (global_step % save_every == 0):
                writer.append_data(np.clip(canvas * 255.0, 0, 255).astype(np.uint8))

            # Phase transition
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
                if sel_weight is not None:
                    sel_weight.fill(1.0)

            # Early stop by target MSE
            if cfg.target_mse is not None and float(np.mean(err_map)) <= cfg.target_mse:
                break
    finally:
        pbar.close()

    # Save final image
    img_path = save_image_rgb01(cfg.output_final_image_path, canvas)

    # Finalize video and apply speed ramp (in-place)
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

    # Summary
    if writer is not None and video_path_final:
        print(f"Done. Image: {img_path} | Video: {video_path_final}")
    else:
        print(f"Done. Image: {img_path}")


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
