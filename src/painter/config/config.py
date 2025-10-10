from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Global pipeline configuration. Contains values only; no business logic.
    Validation happens in __post_init__ to ensure sane ranges and consistency.
    """

    # --- I/O (paths are relative to the src/ folder by design) ---
    input_image_path: str = "../assets/input/sky.jpg"  # PNG/JPG
    output_final_image_path: str = "../outputs/images/final.png"
    make_video: bool = True
    output_video_path: str = "../outputs/video/out.mp4"
    video_fps: int = 30

    # --- Video frame writing frequency (pick one) ---
    save_every_n_strokes: int | None = 10  # takes precedence if set
    video_duration_sec: int | None = 15  # used if save_every_n_strokes is None

    # --- Input image scaling (0 = keep original) ---
    max_size: int = 0

    # --- High-level workload scaling knob (used to derive a few defaults) ---
    workload_scale: int = 1

    # --- Compute budget / quality ---
    total_strokes: int = 1000 * workload_scale
    target_mse: float | None = None  # e.g., 1e-3; None = disabled

    # --- Coarse->fine size schedule (strict phases; no size mixing) ---
    size_scale_mode: str = "log"  # "log" | "linear"
    levels: int = 5
    largest_frac: float = 0.35  # s_max = largest_frac * min(H, W)
    smallest_px: int = 10  # lower bound (never 1px noise)

    # --- Phase transition rules (strictly go from large -> small) ---
    phase_min_strokes: int = 100  # minimum attempts per phase
    phase_accept_window: int = 120  # rolling window length
    phase_accept_threshold: float = 0.05  # if accept-rate < threshold -> move down
    phase_max_attempts_factor: float = 6.0  # ~factor * (H*W/size^2) attempts per phase

    # --- Stroke center selection ---
    roi_sampling: str = "topk_random"  # "argmax" | "topk_random"
    topk: int = 4096

    # --- Anti-sticking (selection weights; does NOT change true error) ---
    use_cooldown: bool = True
    cooldown_factor: float = 0.6  # downweight after accepted stroke
    cooldown_recover: float = 1.02  # multiplicative recovery per step
    cooldown_min: float = 0.25
    cooldown_max: float = 1.0

    # --- Initial canvas color (HEX) ---
    start_color_hex: str = "#FFFFFF"

    # --- Brushes (grayscale masks; white=paint, black=skip) ---
    brush_paths: list[str] = field(default_factory=lambda: ["../assets/brushes/brush.png"])
    use_soft_edges: bool = True  # True: respect mask grayscale
    mask_threshold: float = 0.5  # used only if use_soft_edges=False

    # --- Orientation of strokes ---
    orientation_mode: str = "gradient"  # "gradient" | "none" | "random"
    grad_min_strength: float = 1e-6  # weak structure -> random angle
    angle_jitter_deg: float = 0.0  # Gaussian jitter

    # --- Transparency of strokes ---
    use_alpha: bool = False
    alpha_value: float = 0.25  # 0..1; applied if use_alpha=True

    # --- Determinism ---
    seed: int = 42

    # --- Trailing static frames in the output video ---
    record_last_hold_frames: int = 120

    # --- Postprocess video with smooth speed ramp (in-place replace) ---
    postprocess_speed: bool = True
    # ramp durations in seconds
    speed_slow_seconds: float = 4.0
    speed_fast_seconds: float = 2.64 * workload_scale
    # start/end playback speed factors (e.g., 0.5 = 2× slower; 3.0 = 3× faster)
    speed_slow_from: float = 0.2
    speed_fast_to: float = 1.0 * workload_scale
    # smoothness and codec parameters
    speed_steps: int = 10
    speed_crf: int = 18
    speed_preset: str = "fast"
    speed_fps: float | None = None  # optionally force FPS (e.g., 30)

    def __post_init__(self) -> None:
        # Basics
        if self.video_fps <= 0:
            raise ValueError("video_fps must be > 0")
        if self.max_size < 0:
            raise ValueError("max_size must be >= 0")
        if self.workload_scale <= 0:
            raise ValueError("workload_scale must be > 0")
        if self.total_strokes <= 0:
            raise ValueError("total_strokes must be > 0")
        if self.target_mse is not None and self.target_mse <= 0:
            raise ValueError("target_mse must be > 0 if set")

        # Size schedule
        if self.size_scale_mode not in {"log", "linear"}:
            raise ValueError("size_scale_mode must be 'log' or 'linear'")
        if self.levels < 2:
            raise ValueError("levels must be >= 2")
        if not (0.0 < self.largest_frac <= 1.0):
            raise ValueError("largest_frac must be in (0, 1]")
        if self.smallest_px < 1:
            raise ValueError("smallest_px must be >= 1")

        # Phases
        if self.phase_min_strokes <= 0:
            raise ValueError("phase_min_strokes must be > 0")
        if self.phase_accept_window <= 0:
            raise ValueError("phase_accept_window must be > 0")
        if not (0.0 <= self.phase_accept_threshold <= 1.0):
            raise ValueError("phase_accept_threshold must be in [0, 1]")
        if self.phase_max_attempts_factor <= 0:
            raise ValueError("phase_max_attempts_factor must be > 0")

        # ROI selection
        if self.roi_sampling not in {"argmax", "topk_random"}:
            raise ValueError("roi_sampling must be 'argmax' or 'topk_random'")
        if self.topk <= 0:
            raise ValueError("topk must be > 0")

        # Cooldown
        if not (0.0 < self.cooldown_factor <= 1.0):
            raise ValueError("cooldown_factor must be in (0, 1]")
        if self.cooldown_recover <= 0:
            raise ValueError("cooldown_recover must be > 0")
        if not (0.0 <= self.cooldown_min <= 1.0):
            raise ValueError("cooldown_min must be in [0, 1]")
        if not (0.0 <= self.cooldown_max <= 1.0):
            raise ValueError("cooldown_max must be in [0, 1]")
        if self.cooldown_min > self.cooldown_max:
            raise ValueError("cooldown_min must be <= cooldown_max")

        # Brushes
        if not self.brush_paths:
            raise ValueError("brush_paths must contain at least one path")
        if not (0.0 <= self.mask_threshold <= 1.0):
            raise ValueError("mask_threshold must be in [0, 1]")

        # Orientation
        if self.orientation_mode not in {"gradient", "none", "random"}:
            raise ValueError("orientation_mode must be 'gradient' | 'none' | 'random'")
        if self.grad_min_strength < 0.0:
            raise ValueError("grad_min_strength must be >= 0")
        if self.angle_jitter_deg < 0.0:
            raise ValueError("angle_jitter_deg must be >= 0")

        # Transparency
        if not (0.0 <= self.alpha_value <= 1.0):
            raise ValueError("alpha_value must be in [0, 1]")

        # Speed ramp
        if self.speed_slow_seconds < 0.0 or self.speed_fast_seconds < 0.0:
            raise ValueError("speed_*_seconds must be >= 0")
        if self.speed_steps < 1:
            raise ValueError("speed_steps must be >= 1")
        if self.speed_slow_from <= 0.0:
            raise ValueError("speed_slow_from must be > 0")
        if self.speed_fast_to <= 0.0:
            raise ValueError("speed_fast_to must be > 0")
        if self.speed_crf < 0:
            raise ValueError("speed_crf must be >= 0")
        if self.speed_preset not in {
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        }:
            raise ValueError("speed_preset must be a valid x264 preset")
