from __future__ import annotations

from .base import SwapBackend, SwapParams
from .minimax_h3 import MiniMaxH3Backend
from .wan_animate import WanAnimateBackend

BACKENDS: dict[str, SwapBackend] = {
    b.id: b for b in (WanAnimateBackend(), MiniMaxH3Backend())
}

__all__ = ["BACKENDS", "SwapBackend", "SwapParams"]
