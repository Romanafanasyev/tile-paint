from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from painter.imagemath.image_math import mse_per_pixel, rgb_to_gray01


def test_rgb_to_gray01_and_mse():
    # 2x2 synthetic RGB (float32, [0..1])
    a: NDArray[np.float32] = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]],
        dtype=np.float32,
    )
    b: NDArray[np.float32] = np.zeros_like(a)
    # gray is well-defined float32 in [0..1]
    g = rgb_to_gray01(a)
    assert g.dtype == np.float32
    assert g.min() >= 0.0 and g.max() <= 1.0

    # per-pixel MSE against zeros equals mean of squares per RGB channel
    m = mse_per_pixel(a, b)
    # Known values (sum of squares over 3 channels / 3)
    # top-left: 0; top-right: (1+1+1)/3 = 1
    # bottom-left: (0.5^2)/3 = 0.25/3; bottom-right: (0.5^2)/3 = 0.25/3
    expected = np.array([[0.0, 1.0], [0.25 / 3.0, 0.25 / 3.0]], dtype=np.float32)
    np.testing.assert_allclose(m, expected, rtol=1e-6, atol=1e-6)
