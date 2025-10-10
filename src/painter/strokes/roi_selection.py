from __future__ import annotations

import numpy as np


class CooldownMap:
    """
    Mutable multiplicative priority mask for ROI sampling.

    When enabled, the map stores per-pixel weights in [min_val, max_val] (default 1.0).
    After a stroke is accepted, weights under the stroke ROI are reduced to discourage
    immediate re-selection of the same area. On each iteration, `recover()` softly
    restores weights back towards 1.0.

    Parameters
    ----------
    shape : tuple[int, int]
        (H, W) of the error map.
    enabled : bool
        If False, all methods become no-ops and `.array` is None.
    min_val, max_val : float
        Bounds for the weights (inclusive).
    recover_factor : float
        Multiplicative recovery per step; e.g. 1.02 means +2% towards max per iteration.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        enabled: bool,
        min_val: float,
        max_val: float,
        recover_factor: float,
    ) -> None:
        self._enabled = bool(enabled)
        self._min = float(min_val)
        self._max = float(max_val)
        self._recover = float(recover_factor)
        self._arr: np.ndarray | None = np.ones(shape, dtype=np.float32) if self._enabled else None

    @property
    def array(self) -> np.ndarray | None:
        """Underlying weight map or None when disabled."""
        return self._arr

    def reset(self) -> None:
        """Set all weights back to 1.0 (only when enabled)."""
        if self._arr is not None:
            self._arr.fill(1.0)

    def recover(self) -> None:
        """
        Softly restore weights towards the upper bound.
        Implemented as multiply+clip to keep values within [min, max].
        """
        if self._arr is None:
            return
        np.multiply(self._arr, self._recover, out=self._arr)
        np.clip(self._arr, self._min, self._max, out=self._arr)

    def apply_after_payload(self, payload: tuple, cooldown_factor: float) -> None:
        """
        Reduce weights inside the accepted stroke ROI.

        Parameters
        ----------
        payload : tuple
            (X1, X2, Y1, Y2, new_roi, mask_roi) from the stroke engine.
            mask_roi > 0 indicates covered pixels.
        cooldown_factor : float
            Factor in (0..1]; 1.0 keeps weight unchanged, smaller values reduce it more.
            The effective update is: weight *= 1 - (1 - cooldown_factor) * mask_binary
        """
        if self._arr is None:
            return
        X1, X2, Y1, Y2, _new_roi, mask_roi = payload
        m = (mask_roi > 0).astype(np.float32)
        block = self._arr[Y1:Y2, X1:X2]
        block *= 1.0 - (1.0 - float(cooldown_factor)) * m
        np.clip(block, self._min, self._max, out=block)


def select_roi_center(
    err_map: np.ndarray,
    sel_weight: np.ndarray | None,
    method: str,
    topk: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """
    Pick ROI center (y, x) using the error map, optionally weighted by a cooldown mask.
    """
    weighted = err_map if sel_weight is None else err_map * sel_weight
    H, W = weighted.shape
    if method == "argmax":
        idx = int(np.argmax(weighted))
        y, x = np.unravel_index(idx, (H, W))
        return int(y), int(x)
    if method == "topk_random":
        flat = weighted.ravel()
        k = min(int(topk), flat.size)
        part = np.argpartition(flat, -k)[-k:]
        idx = int(part[int(rng.integers(0, k))])
        y, x = np.unravel_index(idx, (H, W))
        return int(y), int(x)
    raise ValueError(f"Unknown roi_sampling: {method}")
