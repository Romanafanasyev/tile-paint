from __future__ import annotations

from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image

from painter.config.config import Config
from painter.core.pipeline import run_pipeline
from painter.imagemath.image_math import mse_mean


def _make_demo_inputs(tmpdir: Path) -> tuple[str, list[str]]:
    """Create a tiny target image and a simple brush mask in tmp."""
    # target: small gradient image (RGB)
    H, W = 96, 96
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.stack(
        [
            xx / (W - 1),
            yy / (H - 1),
            0.5 * np.ones_like(xx),
        ],
        axis=2,
    ).astype(np.float32)
    # save as PNG [0..255]
    tpath = tmpdir / "target.png"
    Image.fromarray((img * 255).astype(np.uint8)).save(tpath)

    # brush: soft elliptical mask grayscale
    bh, bw = 64, 32
    y, x = np.mgrid[0:bh, 0:bw].astype(np.float32)
    cy, cx = (bh - 1) / 2.0, (bw - 1) / 2.0
    ry, rx = bh / 2.2, bw / 2.2
    dist = ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2
    mask = np.clip(1.0 - dist, 0.0, 1.0)  # ellipse ~1 inside, 0 outside
    bpath = tmpdir / "brush.png"
    Image.fromarray((mask * 255).astype(np.uint8)).save(bpath)

    return str(tpath), [str(bpath)]


@pytest.mark.timeout(20)
def test_smoke_minimal_render(tmp_path: Path, monkeypatch):
    # inputs
    target_path, brush_paths = _make_demo_inputs(tmp_path)

    # outputs
    out_img_dir = tmp_path / "outputs" / "images"
    out_vid_dir = tmp_path / "outputs" / "video"
    out_img = out_img_dir / "final.png"
    out_vid = out_vid_dir / "out.mp4"

    # try ffmpeg availability
    try:
        _ = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_ok = True
    except Exception:
        ffmpeg_ok = False

    # config tuned for speed
    cfg = Config(
        input_image_path=str(target_path),
        output_final_image_path=str(out_img),
        make_video=ffmpeg_ok,  # if ffmpeg missing, render image only
        output_video_path=str(out_vid),
        video_fps=15,
        save_every_n_strokes=10,
        video_duration_sec=5,
        max_size=96,
        workload_scale=1,
        total_strokes=150,
        size_scale_mode="log",
        levels=3,
        largest_frac=0.5,
        smallest_px=8,
        roi_sampling="topk_random",
        topk=512,
        use_soft_edges=True,
        start_color_hex="#FFFFFF",
        brush_paths=brush_paths,
        use_alpha=False,
        seed=123,
        record_last_hold_frames=0,
        postprocess_speed=False,
    )

    # a tiny helper to get initial MSE for assertion
    from painter.io.files import load_image_rgb01

    start = np.ones_like(load_image_rgb01(cfg.input_image_path, cfg.max_size), dtype=np.float32)
    from painter.imagemath.image_math import hex_to_rgb01

    start[:] = hex_to_rgb01(cfg.start_color_hex)[None, None, :]

    target = load_image_rgb01(cfg.input_image_path, cfg.max_size)
    initial_mse = float(mse_mean(start, target))

    # run
    run_pipeline(cfg)

    # image exists
    imgs = list(out_img_dir.glob("final*.png"))
    assert imgs, "final image was not produced"

    # final MSE should be lower than initial
    produced = np.asarray(Image.open(imgs[-1]).convert("RGB"), dtype=np.float32) / 255.0
    assert float(mse_mean(produced, target)) < initial_mse

    # video exists if ffmpeg available
    vids = list(out_vid_dir.glob("out*.mp4"))
    if ffmpeg_ok and cfg.make_video:
        assert vids, "video was expected but not produced"
