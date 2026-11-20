from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

Log = Callable[[str], None]
Progress = Callable[[float, str], None]   # (fraccion 0..1, descripcion)


@dataclass
class SwapParams:
    video: Path
    image: Path
    prompt: str = ""
    max_seconds: float = 4.0
    fps: int = 16
    resolution_area: tuple[int, int] = (832, 480)
    steps: int = 20
    seed: int = -1
    weights: str = "gguf:Q4_K_M"          # "gguf:<QUANT>" o "nf4"
    offload: str = "model"                # "model" | "sequential"
    segment_len: int = 0                  # frames por bloque (4N+1); 0 = automatico segun tokens_budget
    tokens_budget: int = 16000            # tokens por bloque que entran en 16 GB con Q3_K_M (prueba real: ~15k)
    prev_frames: int = 1                  # frames de continuidad entre segmentos (1 o 5)
    mask_iterations: int = 3
    mask_k: int = 7
    mask_w_len: int = 1
    mask_h_len: int = 1
    keep_audio: bool = True


class SwapBackend(ABC):
    id: str
    label: str

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """(disponible, motivo si no lo esta)."""

    @abstractmethod
    def run(self, params: SwapParams, log: Log, progress: Progress) -> Path:
        """Ejecuta el swap y devuelve la ruta del video resultante."""
