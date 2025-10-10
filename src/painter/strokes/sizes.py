from painter.config.config import Config


def make_all_sizes(cfg: Config, W: int, H: int) -> list[int]:
    """
    Build a strictly decreasing list of stroke box sizes (large → small),
    deduplicated and clamped to [smallest_px, ...].
    """
    s_max = max(cfg.smallest_px, int(round(min(W, H) * cfg.largest_frac)))
    s_min = max(1, int(cfg.smallest_px))
    L = max(2, int(cfg.levels))
    if cfg.size_scale_mode == "linear":
        step = (s_max - s_min) / float(L - 1)
        sizes = [max(1, int(round(s_max - i * step))) for i in range(L)]
    else:
        sizes = [max(1, int(round(s_max * (s_min / s_max) ** (i / (L - 1))))) for i in range(L)]
    sizes = sorted(set(sizes), reverse=True)
    return [max(s_min, s) for s in sizes]
