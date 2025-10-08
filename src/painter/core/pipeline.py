from __future__ import annotations

import math
from typing import Optional

import numpy as np
from tqdm import tqdm

from painter.config.config import Config
from painter.io.files import load_image_rgb01, save_image_rgb01, load_brushes
from painter.imagemath.image_math import hex_to_rgb01
from painter.strokes.brush_ops import Brush
from painter.strokes.roi_selection import CooldownMap
from painter.strokes.stroke_engine import PainterEngine
from painter.video.recorder import VideoRecorder
from painter.video.speed_ramp import apply_speed_ramp_inplace


def main(cfg: Config) -> None:
    rng = np.random.default_rng(cfg.seed)

    target = load_image_rgb01(cfg.input_image_path, cfg.max_size)
    H, W, _ = target.shape
    start_rgb = hex_to_rgb01(cfg.start_color_hex)
    canvas = np.ones_like(target, dtype=np.float32) * start_rgb

    brushes: list[Brush] = load_brushes(cfg.brush_paths)

    cooldown = CooldownMap(
        shape=(H, W),
        enabled=cfg.use_cooldown,
        min_val=cfg.cooldown_min,
        max_val=cfg.cooldown_max,
        recover_factor=cfg.cooldown_recover,
    )

    engine = PainterEngine(
        cfg=cfg,
        target=target,
        start_canvas=canvas,
        brushes=brushes,
        cooldown=cooldown,
        rng=rng,
    )

    recorder: Optional[VideoRecorder] = None
    video_path_final: Optional[str] = None
    if cfg.make_video:
        recorder = VideoRecorder(cfg.output_video_path, fps=cfg.video_fps)
        video_path_final = recorder.path
        if cfg.save_every_n_strokes and cfg.save_every_n_strokes > 0:
            save_every = cfg.save_every_n_strokes
        else:
            total_frames = max(1, cfg.video_fps * (cfg.video_duration_sec or 10))
            save_every = max(1, math.ceil(cfg.total_strokes / total_frames))

    pbar = tqdm(total=cfg.total_strokes, desc="Painting", ncols=80)
    try:
        for _ in range(cfg.total_strokes):
            engine.step()
            pbar.update(1)

            if recorder is not None and (engine.global_step % save_every == 0):
                recorder.append_canvas(engine.canvas)

            if engine.done_by_mse():
                break
    finally:
        pbar.close()

    img_path = save_image_rgb01(cfg.output_final_image_path, engine.canvas)

    if recorder is not None and video_path_final:
        last = np.clip(engine.canvas * 255.0, 0, 255).astype(np.uint8)
        recorder.hold(last, cfg.record_last_hold_frames)
        recorder.close()

        if cfg.postprocess_speed:
            try:
                apply_speed_ramp_inplace(
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

    if recorder is not None and video_path_final:
        print(f"Done. Image: {img_path} | Video: {video_path_final}")
    else:
        print(f"Done. Image: {img_path}")


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
