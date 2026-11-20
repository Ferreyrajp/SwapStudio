from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HwInfo:
    gpu: str | None
    vram_gb: float | None
    ram_gb: float | None
    cuda: bool

    def describe(self) -> str:
        if not self.cuda:
            return "No se detecto GPU NVIDIA con CUDA (o PyTorch no esta instalado)."
        ram = f"{self.ram_gb:.0f} GB RAM" if self.ram_gb else "RAM desconocida"
        return f"{self.gpu} - {self.vram_gb:.1f} GB VRAM - {ram}"


def detect() -> HwInfo:
    gpu = vram = ram = None
    cuda = False
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            gpu, vram, cuda = p.name, p.total_memory / 1024**3, True
    except Exception:
        pass
    try:
        import psutil

        ram = psutil.virtual_memory().total / 1024**3
    except Exception:
        pass
    return HwInfo(gpu, vram, ram, cuda)


def recommend(vram_gb: float | None) -> tuple[str, str]:
    """(weights, preset de resolucion) razonables segun la VRAM.

    Con offload de modelo completo, el transformer tiene que caber entero en VRAM
    junto con las activaciones; por eso se deja margen.
    """
    if vram_gb is None:
        return "gguf:Q4_K_M", "Borrador (640x360)"
    if vram_gb < 10:
        return "gguf:Q2_K", "Borrador (640x360)"
    if vram_gb < 14:
        return "gguf:Q3_K_M", "Borrador (640x360)"
    if vram_gb < 20:
        # Q4_K_M (11.5 GB) + activaciones queda al limite en 16 GB; Q3_K_M deja margen.
        return "gguf:Q3_K_M", "Borrador (640x360)"
    return "gguf:Q6_K", "Estandar (832x480)"
