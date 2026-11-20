from __future__ import annotations

from pathlib import Path

from .base import Log, Progress, SwapBackend, SwapParams

REASON = (
    "Pendiente. El swap con MiniMax H3 necesita el VIDEO como referencia (ref2va). "
    "Segun la ficha de Hyperflow H3 Flash, los runtimes GGUF/stable-diffusion.cpp solo aceptan "
    "imagenes de referencia; el video requiere el paquete Python INT8, que pide >= 69 GiB de RAM "
    "con offload. Ademas las LoRAs de character swap estan hechas para el H3 completo, no para "
    "la version podada de 14B. Con 32 GB de RAM no es viable hoy."
)


class MiniMaxH3Backend(SwapBackend):
    """Hueco reservado para H3.

    Para implementarlo cuando haya hardware/modelos compatibles:
      1. Cargar el pipeline ref2va (paquete `hyperflow-h3-flash` o diffusers).
      2. Pasar [imagen, video] como referencias con el prompt "<Picture 1> ... <Video 1>".
      3. Aplicar la LoRA de character swap si el modelo base la soporta.
      4. Devolver la ruta del mp4 generado.
    """

    id = "minimax-h3"
    label = "MiniMax H3"

    def available(self) -> tuple[bool, str]:
        return False, REASON

    def run(self, params: SwapParams, log: Log, progress: Progress) -> Path:
        raise NotImplementedError(REASON)
