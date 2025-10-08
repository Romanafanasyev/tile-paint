from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from painter.config.config import Config
from painter.imagemath.image_math import (
    mse_mean,
    mse_per_pixel,
    rgb_to_gray01,
    simple_central_gradients,
)
from painter.strokes.brush_ops import Brush, build_weight_mask
from painter.strokes.geometry import overlap_mask_and_frame
from painter.strokes.roi_selection import CooldownMap, select_roi_center
from painter.strokes.sizes import make_all_sizes


def _local_gradient_angle_deg(
    Ix: np.ndarray,
    Iy: np.ndarray,
    cx: int,
    cy: int,
    size: int,
    rng: np.random.Generator,
    min_strength: float,
    jitter_deg: float,
) -> float:
    """Orientation from summed gradients in a size x size window; random if weak."""
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
        angle = phi + 90.0

    if jitter_deg > 1e-6:
        angle += float(rng.normal(0.0, jitter_deg))
    while angle < 0.0:
        angle += 180.0
    while angle >= 180.0:
        angle -= 180.0
    return angle


def _choose_color_from_target(roi_target: np.ndarray, cover_mask_2d: np.ndarray) -> np.ndarray:
    """Weighted average RGB using coverage mask."""
    s = float(np.sum(cover_mask_2d))
    if s <= 1e-8:
        return roi_target.mean(axis=(0, 1), dtype=np.float32)
    w3 = cover_mask_2d[..., None]
    num = (roi_target * w3).sum(axis=(0, 1))
    return (num / s).astype(np.float32)


class PainterEngine:
    """
    Encapsulates stroke simulation, acceptance, and coarse-to-fine phase control.
    Pipeline drives `step()` in a loop; the engine updates canvas and error map.
    """

    def __init__(
        self,
        cfg: Config,
        target: np.ndarray,
        start_canvas: np.ndarray,
        brushes: List[Brush],
        cooldown: CooldownMap,
        rng: np.random.Generator,
    ) -> None:
        self.cfg = cfg
        self.target = target
        self.canvas = start_canvas
        self.err_map = mse_per_pixel(start_canvas, target)

        self.brushes = brushes
        self.cooldown = cooldown
        self.rng = rng

        H, W, _ = target.shape
        self.sizes = make_all_sizes(cfg, W, H)
        self.level = 0
        self.size_px = self.sizes[self.level]

        if cfg.orientation_mode == "gradient":
            gray = rgb_to_gray01(target)
            self.Ix, self.Iy = simple_central_gradients(gray)
        else:
            self.Ix = self.Iy = None

        self.attempts_in_phase = 0
        self.accept_window: Deque[int] = deque(maxlen=cfg.phase_accept_window)
        self.phase_max_attempts = self._phase_max_attempts_for(self.size_px)
        self.global_step = 0

    def _phase_max_attempts_for(self, size_px: int) -> int:
        H, W, _ = self.target.shape
        approx = self.cfg.phase_max_attempts_factor * (H * W) / max(1, size_px * size_px)
        return max(self.cfg.phase_min_strokes, int(round(approx)))

    def _pick_angle(self, x: int, y: int) -> float:
        if self.cfg.orientation_mode == "gradient" and self.Ix is not None and self.Iy is not None:
            return _local_gradient_angle_deg(
                self.Ix,
                self.Iy,
                x,
                y,
                self.size_px,
                self.rng,
                self.cfg.grad_min_strength,
                self.cfg.angle_jitter_deg,
            )
        if self.cfg.orientation_mode == "random":
            return float(self.rng.uniform(0.0, 180.0))
        return 0.0

    def _try_one_stroke(
        self,
        x_center: int,
        y_center: int,
        brush: Brush,
    ) -> Tuple[bool, Optional[Tuple], float]:
        H, W, _ = self.canvas.shape
        mask = brush.render(self.size_px, self._pick_angle(x_center, y_center))
        mh, mw = mask.shape
        if mh == 0 or mw == 0:
            return False, None, 0.0

        ov = overlap_mask_and_frame(x_center, y_center, mw, mh, W, H)
        if ov is None:
            return False, None, 0.0
        X1, X2, Y1, Y2, MX1, MX2, MY1, MY2 = ov

        roi_canvas = self.canvas[Y1:Y2, X1:X2]
        roi_target = self.target[Y1:Y2, X1:X2]
        mask_roi = mask[MY1:MY2, MX1:MX2]

        m, wmask = build_weight_mask(
            mask_roi,
            use_soft_edges=self.cfg.use_soft_edges,
            mask_threshold=self.cfg.mask_threshold,
            use_alpha=self.cfg.use_alpha,
            alpha_value=self.cfg.alpha_value,
        )
        if np.max(wmask) <= 0.0:
            return False, None, 0.0

        color = _choose_color_from_target(roi_target, m)
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

    def _commit(self, payload: Tuple) -> None:
        X1, X2, Y1, Y2, new_roi, _mask_roi = payload
        roi_target = self.target[Y1:Y2, X1:X2]
        self.canvas[Y1:Y2, X1:X2] = new_roi
        self.err_map[Y1:Y2, X1:X2] = mse_per_pixel(new_roi, roi_target)

    def _maybe_switch_phase(self) -> None:
        move_down = False
        if self.attempts_in_phase >= self.cfg.phase_min_strokes:
            n = min(len(self.accept_window), self.cfg.phase_min_strokes)
            if n > 0:
                rate = sum(self.accept_window) / float(n)
                if rate < self.cfg.phase_accept_threshold:
                    move_down = True
        if self.attempts_in_phase >= self.phase_max_attempts:
            move_down = True

        if move_down and self.level + 1 < len(self.sizes):
            self.level += 1
            self.size_px = self.sizes[self.level]
            self.attempts_in_phase = 0
            self.accept_window.clear()
            self.phase_max_attempts = self._phase_max_attempts_for(self.size_px)
            self.cooldown.reset()

    def step(self) -> bool:
        """
        One attempt: recover cooldown, pick ROI, try stroke, commit if better,
        update acceptance stats and possibly switch to a smaller size.
        Returns True if a stroke was accepted.
        """
        self.cooldown.recover()

        y, x = select_roi_center(
            err_map=self.err_map,
            sel_weight=self.cooldown.array,
            method=self.cfg.roi_sampling,
            topk=self.cfg.topk,
            rng=self.rng,
        )

        brush = self.brushes[int(self.rng.integers(0, len(self.brushes)))]
        accepted, payload, _gain_pp = self._try_one_stroke(x, y, brush)

        self.attempts_in_phase += 1
        self.accept_window.append(1 if accepted else 0)

        if accepted and payload is not None:
            self._commit(payload)
            self.cooldown.apply_after_payload(payload, self.cfg.cooldown_factor)

        self.global_step += 1
        self._maybe_switch_phase()
        return bool(accepted)

    def done_by_mse(self) -> bool:
        """Early stop by target MSE if configured."""
        if self.cfg.target_mse is None:
            return False
        return float(np.mean(self.err_map)) <= float(self.cfg.target_mse)
