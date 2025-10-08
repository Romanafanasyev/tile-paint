import os
from typing import List

import numpy as np
from PIL import Image


def ensure_dir_for(path: str) -> None:
    """Create parent directory for a file path if it does not exist."""
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def unique_path(path: str) -> str:
    """Return a non-clobbering path by appending _1, _2, ... if needed."""
    base, ext = os.path.splitext(path)
    candidate = path
    k = 1
    while os.path.exists(candidate):
        candidate = f"{base}_{k}{ext}"
        k += 1
    return candidate


def load_image_rgb01(path: str, max_size: int = 0) -> np.ndarray:
    """Load image as float32 RGB in [0,1]; optionally downscale to max_size."""
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
    """Save float32 RGB [0,1] image to a unique file path. Returns actual path."""
    ensure_dir_for(path)
    path2 = unique_path(path)
    img8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img8, mode="RGB").save(path2)
    return path2


def load_brushes_gray01(paths: List[str]) -> List[np.ndarray]:
    """Load grayscale brush masks in [0,1]. White=paint, black=skip."""
    out: List[np.ndarray] = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Brush file not found: {p}")
        arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        out.append(arr)
    if not out:
        raise RuntimeError("No brush masks loaded. Check Config.brush_paths.")
    return out
