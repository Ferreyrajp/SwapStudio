"""Descarga de modelos desde Hugging Face. Uso:

    python -m swapstudio.models --quant Q4_K_M        # lo necesario para el modo GGUF
    python -m swapstudio.models --nf4                 # alternativa con bitsandbytes
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from . import config

Log = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def ensure_gguf(quant: str, log: Log = _noop) -> Path:
    if quant not in config.GGUF_SIZES_GB:
        raise ValueError(f"Cuantizacion desconocida: {quant}")
    filename = f"Wan2.2-Animate-14B-{quant}.gguf"
    target = config.MODELS_DIR / "gguf" / filename
    if target.exists():
        return target
    log(f"Descargando {filename} (~{config.GGUF_SIZES_GB[quant]} GB)...")
    path = hf_hub_download(config.GGUF_REPO, filename, local_dir=config.MODELS_DIR / "gguf")
    return Path(path)


def ensure_base_components(include_transformer: bool, log: Log = _noop) -> Path:
    """Descarga texto, CLIP, VAE, scheduler y config. El transformer solo si hace falta (NF4)."""
    local = config.MODELS_DIR / "wan2.2-animate-diffusers"
    ignore = ["*.md", "*.mp4", "*.png", "*.jpg", "*.jpeg", "*.gif"]
    if not include_transformer:
        ignore += ["transformer/*.safetensors", "transformer/*.bin", "transformer/*.index.json"]
    log("Descargando/verificando componentes base de Wan2.2-Animate (texto, VAE, CLIP)...")
    snapshot_download(config.DIFFUSERS_REPO, local_dir=local, ignore_patterns=ignore)
    return local


def ensure_process_checkpoint(log: Log = _noop) -> Path:
    """Modelos de preprocesado (deteccion, pose, SAM2). Devuelve el directorio `ckpt_path`.

    Ojo: en el repo oficial `pose2d/vitpose_h_wholebody.onnx` es una CARPETA (pesos ONNX en
    formato de datos externos, con un `end2end.onnx` dentro), no un archivo. Por eso se
    descargan con patrones que cubren tanto archivos como carpetas.
    """
    files = HfApi().list_repo_files(config.ORIGINAL_REPO)
    local_root = config.MODELS_DIR / "original"
    roots: set[str] = set()
    patterns: list[str] = []
    for sub, name in config.PROCESS_FILES.items():
        wanted = f"{sub}/{name}"
        prefix = None
        for f in files:
            if "process_checkpoint" not in f:
                continue
            idx = f.find(wanted)
            end = idx + len(wanted)
            if idx != -1 and (end == len(f) or f[end] == "/"):
                prefix = f[:idx]
                break
        if prefix is None:
            raise FileNotFoundError(
                f"No encontre {wanted} en {config.ORIGINAL_REPO}. "
                "Es posible que el repo haya cambiado de estructura; revisa su pestana Files."
            )
        roots.add(prefix)
        patterns += [prefix + wanted, prefix + wanted + "/*"]
    if len(roots) != 1:
        raise RuntimeError(f"Estructura inesperada de process_checkpoint: {roots}")
    root = roots.pop()

    log("Descargando/verificando modelos de preprocesado (deteccion, pose, SAM2)...")
    snapshot_download(config.ORIGINAL_REPO, local_dir=local_root, allow_patterns=patterns)

    ckpt = local_root / root.rstrip("/")
    for sub, name in config.PROCESS_FILES.items():
        if not (ckpt / sub / name).exists():
            raise FileNotFoundError(f"Falta {ckpt / sub / name} tras la descarga.")
    pose = ckpt / "pose2d" / config.PROCESS_FILES["pose2d"]
    if pose.is_dir() and not (pose / "end2end.onnx").exists():
        contenido = sorted(p.name for p in pose.iterdir())[:8]
        raise FileNotFoundError(
            f"{pose} es una carpeta pero no contiene end2end.onnx (hay: {contenido}...). "
            "El preprocesado oficial lo necesita."
        )
    return ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=list(config.GGUF_SIZES_GB), help="Descarga ese GGUF")
    ap.add_argument("--nf4", action="store_true", help="Descarga el transformer completo (para NF4)")
    ap.add_argument("--skip-process", action="store_true", help="No descargar modelos de preprocesado")
    args = ap.parse_args()
    if not args.quant and not args.nf4:
        ap.error("Indica --quant Q4_K_M (recomendado para 16 GB) o --nf4")
    if args.quant:
        ensure_gguf(args.quant, print)
    ensure_base_components(include_transformer=args.nf4, log=print)
    if not args.skip_process:
        print("Checkpoint de preprocesado en:", ensure_process_checkpoint(print))
    print("Listo.")


if __name__ == "__main__":
    main()
