from typing import cast

import numpy as np
from numpy.typing import NDArray


def hex_to_rgb01(hex_color: str) -> NDArray[np.float32]:
    """HEX color (#RRGGBB) → float32 RGB in [0,1]."""
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return np.array([r, g, b], dtype=np.float32)


def rgb_to_gray01(img: NDArray[np.float32]) -> NDArray[np.float32]:
    """RGB → luma (BT.709) in [0,1], float32."""
    gray = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    return cast(NDArray[np.float32], gray.astype(np.float32))


def simple_central_gradients(
    gray: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Central-difference gradients with clamped edges.
    Returns (Ix, Iy) where Ix is horizontal gradient and Iy is vertical gradient.
    """
    H, W = gray.shape
    Ix = 0.5 * (np.roll(gray, -1, axis=1) - np.roll(gray, 1, axis=1))
    Iy = 0.5 * (np.roll(gray, -1, axis=0) - np.roll(gray, 1, axis=0))
    # edge handling
    Ix[:, 0] = gray[:, 1] - gray[:, 0]
    Ix[:, -1] = gray[:, -1] - gray[:, -2]
    Iy[0, :] = gray[1, :] - gray[0, :]
    Iy[-1, :] = gray[-1, :] - gray[-2, :]
    return Ix, Iy


def mse_per_pixel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel MSE for two RGB images."""
    d = a - b
    out = np.mean(d * d, axis=2)
    return cast(NDArray[np.float32], out)


def mse_mean(a: np.ndarray, b: np.ndarray) -> float:
    """Mean MSE for two RGB images."""
    return float(np.mean((a - b) ** 2))
