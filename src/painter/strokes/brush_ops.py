from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Brush:
    """
    Immutable brush: holds a grayscale mask in [0,1], where white=paint, black=skip.
    """

    mask01: np.ndarray  # 2D float32 array, values in [0,1]

    def render(self, target_box_size: int, angle_deg: float) -> np.ndarray:
        """
        Scale the brush so that max(H, W) == target_box_size, then rotate by angle_deg.
        Returns a new 2D float32 mask in [0,1].
        """
        mask = self.mask01
        hb, wb = mask.shape
        scale = target_box_size / float(max(hb, wb))
        new_h = max(1, int(round(hb * scale)))
        new_w = max(1, int(round(wb * scale)))

        img = Image.fromarray((mask * 255.0).astype(np.uint8))
        img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        if abs(angle_deg) > 1e-6:
            img = img.rotate(
                angle_deg, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=0
            )
        return np.asarray(img, dtype=np.float32) / 255.0


def build_weight_mask(
    mask_roi: np.ndarray,
    use_soft_edges: bool,
    mask_threshold: float,
    use_alpha: bool,
    alpha_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    From a cropped brush mask (ROI), build:
      - m:   binary/soft coverage mask in [0,1]
      - w:   blending weight mask in [0,1] (alpha applied if enabled)
    """
    if use_soft_edges:
        m = np.clip(mask_roi, 0.0, 1.0).astype(np.float32)
    else:
        m = (mask_roi >= mask_threshold).astype(np.float32)
    w = np.clip(alpha_value * m, 0.0, 1.0) if use_alpha else m
    return m, w
