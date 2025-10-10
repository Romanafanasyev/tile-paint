from __future__ import annotations

from painter.config.config import Config
from painter.strokes.sizes import make_all_sizes


def test_make_all_sizes_log_and_linear():
    W, H = 200, 120
    cfg = Config()
    cfg.levels = 5
    cfg.largest_frac = 0.4
    cfg.smallest_px = 8

    # log schedule
    cfg.size_scale_mode = "log"
    sizes_log = make_all_sizes(cfg, W, H)
    assert len(sizes_log) >= 2
    assert sizes_log == sorted(sizes_log, reverse=True)
    assert sizes_log[-1] >= cfg.smallest_px

    # linear schedule
    cfg.size_scale_mode = "linear"
    sizes_lin = make_all_sizes(cfg, W, H)
    assert len(sizes_lin) >= 2
    assert sizes_lin == sorted(sizes_lin, reverse=True)
    assert sizes_lin[-1] >= cfg.smallest_px

    # both begin not bigger than s_max and end >= smallest_px
    s_max = int(round(min(W, H) * cfg.largest_frac))
    assert sizes_log[0] <= s_max and sizes_lin[0] <= s_max
