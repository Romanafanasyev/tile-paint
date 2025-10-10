from __future__ import annotations

import numpy as np

from painter.strokes.roi_selection import select_roi_center


def test_roi_topk_vs_argmax():
    rng = np.random.default_rng(0)
    H, W = 32, 40
    err = np.zeros((H, W), dtype=np.float32)
    # single clear maximum
    peak_y, peak_x = 10, 25
    err[peak_y, peak_x] = 10.0

    # argmax must pick the peak
    y1, x1 = select_roi_center(err_map=err, sel_weight=None, method="argmax", topk=1, rng=rng)
    assert (y1, x1) == (peak_y, peak_x)

    # topk_random with topk=1 degenerates to argmax
    y2, x2 = select_roi_center(err_map=err, sel_weight=None, method="topk_random", topk=1, rng=rng)
    assert (y2, x2) == (peak_y, peak_x)
