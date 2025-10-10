from __future__ import annotations

from painter.video.speed_ramp import _atempo_chain_str


def test_atempo_chain_str_decomposition():
    # > 2.0 splits into 2.0 * ... * residual
    assert _atempo_chain_str(5.0) == "atempo=2.0,atempo=2.0,atempo=1.25"
    # < 0.5 splits into 0.5 * 0.5 * residual
    assert _atempo_chain_str(0.2) == "atempo=0.5,atempo=0.5,atempo=0.8"
    # ~1.0 returns empty (no atempo)
    assert _atempo_chain_str(1.0) == ""
