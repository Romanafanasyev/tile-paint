from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from painter.config.config import Config
from painter.core import pipeline


# TODO: tune actual values for these presets
PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "workload_scale": 4,          # ~4k strokes
        "make_video": True,
        "max_size": 640,
    },
    "balanced": {
        "workload_scale": 25,         # ~25k strokes
        "make_video": True,
        "max_size": 0,                # full-res
    },
    "quality": {
        "workload_scale": 60,         # ~60k strokes
        "make_video": True,
        "max_size": 0,                # full-res
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="painter",
        description="Stroke-based image reconstruction (coarse->fine) with optional video recording.",
    )

    # Preset
    p.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        help="Quick preset (fast|balanced|quality). Flags after the preset override its values.",
        default=None,
    )

    # Paths (relative to src/ or absolute)
    p.add_argument("--input", dest="input_image_path", default=None, help="Input PNG/JPG path.")
    p.add_argument("--out-image", dest="output_final_image_path", default=None, help="Output image path (auto _1/_2 suffix if exists).")
    p.add_argument("--out-video", dest="output_video_path", default=None, help="Output video path (auto _1/_2 suffix if exists).")

    # Brushes (list)
    p.add_argument(
        "--brush",
        dest="brush_paths",
        action="append",
        default=None,
        help="Brush PNG (grayscale). Repeat flag or pass comma-separated list to add multiple.",
    )

    # Core workload
    p.add_argument(
        "--workload",
        dest="workload_scale",
        type=int,
        default=None,
        help="Thousands of strokes (1 -> ~1000). Also sets total_strokes, speed_fast_seconds, speed_fast_to.",
    )
    p.add_argument("--make-video", dest="make_video", action="store_true", help="Enable video recording.")
    p.add_argument("--no-video", dest="make_video", action="store_false", help="Disable video recording.")
    p.set_defaults(make_video=None)

    # Video cadence
    p.add_argument("--video-fps", dest="video_fps", type=int, default=None, help="Video FPS (frames per second).")
    p.add_argument(
        "--save-every",
        dest="save_every_n_strokes",
        type=int,
        default=None,
        help="Write one video frame every N accepted attempts (takes precedence over --video-duration).",
    )
    p.add_argument(
        "--video-duration",
        dest="video_duration_sec",
        type=int,
        default=None,
        help="Approximate video duration in seconds if --save-every is not set.",
    )

    # Input resize
    p.add_argument("--max-size", dest="max_size", type=int, default=None, help="Resize longer image side to N px (0 keeps original).")

    # Quality target
    p.add_argument("--target-mse", dest="target_mse", type=float, default=None, help="Early stop when global MSE <= value (disabled if omitted).")

    # Size schedule
    p.add_argument("--size-scale-mode", dest="size_scale_mode", choices=["log", "linear"], default=None, help="Brush size schedule: logarithmic or linear.")
    p.add_argument("--levels", dest="levels", type=int, default=None, help="Number of size phases (>= 2).")
    p.add_argument("--largest-frac", dest="largest_frac", type=float, default=None, help="Largest brush box = fraction of min(H, W).")
    p.add_argument("--smallest-px", dest="smallest_px", type=int, default=None, help="Smallest brush box in pixels (>= 1).")

    # ROI sampling
    p.add_argument(
        "--roi-sampling",
        dest="roi_sampling",
        choices=["argmax", "topk_random"],
        default=None,
        help="ROI center selection: 'argmax' (highest error) or 'topk_random' (diversity from top-K).",
    )

    # Canvas & brush behavior
    p.add_argument("--start-color-hex", dest="start_color_hex", default=None, help="Initial canvas color in HEX, e.g. #FFFFFF.")
    p.add_argument("--use-soft-edges", dest="use_soft_edges", action="store_true", help="Treat brush grayscale as soft alpha.")
    p.add_argument("--no-soft-edges", dest="use_soft_edges", action="store_false", help="Use hard edges (thresholded mask).")
    p.set_defaults(use_soft_edges=None)
    p.add_argument("--mask-threshold", dest="mask_threshold", type=float, default=None, help="Binary threshold for hard edges (0..1).")
    p.add_argument("--use-alpha", dest="use_alpha", action="store_true", help="Apply global alpha multiplier to brush.")
    p.add_argument("--no-alpha", dest="use_alpha", action="store_false", help="Disable global alpha multiplier.")
    p.set_defaults(use_alpha=None)
    p.add_argument("--alpha-value", dest="alpha_value", type=float, default=None, help="Global alpha value in [0..1] when --use-alpha is set.")

    # Determinism
    p.add_argument("--seed", dest="seed", type=int, default=None, help="Random seed for reproducibility.")

    return p


def _normalize_brush_list(v: Optional[List[str]]) -> Optional[List[str]]:
    if not v:
        return None
    out: List[str] = []
    for item in v:
        parts = [s.strip() for s in item.split(",")] if "," in item else [item.strip()]
        out.extend([p for p in parts if p])
    return out or None


def _apply_workload_derivatives(overrides: Dict[str, Any], ws: int) -> None:
    overrides["workload_scale"] = int(ws)
    overrides["total_strokes"] = int(1000 * ws)
    overrides["speed_fast_seconds"] = float(2.64 * ws)
    overrides["speed_fast_to"] = float(1.0 * ws)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    overrides: Dict[str, Any] = {}
    if args.preset:
        overrides.update(PRESETS[args.preset])

    for name in [
        "input_image_path",
        "output_final_image_path",
        "output_video_path",
        "video_fps",
        "save_every_n_strokes",
        "video_duration_sec",
        "max_size",
        "target_mse",
        "size_scale_mode",
        "levels",
        "largest_frac",
        "smallest_px",
        "roi_sampling",
        "start_color_hex",
        "mask_threshold",
        "alpha_value",
        "seed",
    ]:
        val = getattr(args, name, None)
        if val is not None:
            overrides[name] = val

    if args.make_video is not None:
        overrides["make_video"] = bool(args.make_video)
    if args.use_soft_edges is not None:
        overrides["use_soft_edges"] = bool(args.use_soft_edges)
    if args.use_alpha is not None:
        overrides["use_alpha"] = bool(args.use_alpha)

    brush_paths = _normalize_brush_list(args.brush_paths)
    if brush_paths is not None:
        overrides["brush_paths"] = brush_paths

    if args.workload_scale is not None:
        _apply_workload_derivatives(overrides, int(args.workload_scale))
    elif "workload_scale" in overrides:
        _apply_workload_derivatives(overrides, int(overrides["workload_scale"]))

    cfg = Config(**overrides)
    pipeline.main(cfg)


if __name__ == "__main__":
    main()
