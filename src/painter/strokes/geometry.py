from __future__ import annotations


def overlap_mask_and_frame(
    cx: int,
    cy: int,
    mask_w: int,
    mask_h: int,
    W: int,
    H: int,
) -> tuple[int, int, int, int, int, int, int, int] | None:
    """Canvas/mask overlap indices for placing a mask centered at (cx, cy)."""
    mask_left = cx - mask_w // 2
    mask_top = cy - mask_h // 2
    mask_right = mask_left + mask_w
    mask_bottom = mask_top + mask_h

    X1 = max(0, mask_left)
    Y1 = max(0, mask_top)
    X2 = min(W, mask_right)
    Y2 = min(H, mask_bottom)
    if X1 >= X2 or Y1 >= Y2:
        return None

    MX1 = X1 - mask_left
    MY1 = Y1 - mask_top
    MX2 = MX1 + (X2 - X1)
    MY2 = MY1 + (Y2 - Y1)

    MX1 = max(0, min(MX1, mask_w))
    MX2 = max(0, min(MX2, mask_w))
    MY1 = max(0, min(MY1, mask_h))
    MY2 = max(0, min(MY2, mask_h))
    return (X1, X2, Y1, Y2, MX1, MX2, MY1, MY2)
