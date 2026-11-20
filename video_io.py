from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg

_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def probe(video: str | Path) -> dict:
    """Devuelve width, height, fps, duration, has_audio leyendo la salida de ffmpeg."""
    r = _run([_FFMPEG, "-hide_banner", "-i", str(video)])
    text = r.stderr
    info = {"width": 0, "height": 0, "fps": 0.0, "duration": 0.0, "has_audio": "Audio:" in text}
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if m:
        h, mi, s = m.groups()
        info["duration"] = int(h) * 3600 + int(mi) * 60 + float(s)
    m = re.search(r"Video:.*?, (\d{2,5})x(\d{2,5})", text)
    if m:
        info["width"], info["height"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?) fps", text)
    if m:
        info["fps"] = float(m.group(1))
    if not info["width"]:
        raise ValueError(f"No pude leer el video: {video}")
    return info


def fit_area(width: int, height: int, area_w: int, area_h: int, divisor: int = 16) -> tuple[int, int]:
    """Escala (width,height) para que su area ~ area_w*area_h, lados multiplos de `divisor`."""
    scale = math.sqrt((area_w * area_h) / (width * height))
    w = max(divisor, round(width * scale / divisor) * divisor)
    h = max(divisor, round(height * scale / divisor) * divisor)
    return w, h


def prepare_input(src: str | Path, dst: str | Path, max_seconds: float, fps: int,
                  area: tuple[int, int]) -> dict:
    """Recorta, remuestrea el fps y reduce la resolucion del video fuente.

    Reducir aqui (antes del preprocesado) ahorra RAM y tiempo: el preprocesado oficial
    de todas formas reescala por area.
    """
    info = probe(src)
    w, h = fit_area(info["width"], info["height"], *area)
    cmd = [
        _FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-t", f"{max_seconds:.3f}",
        "-vf", f"fps={fps},scale={w}:{h}:flags=lanczos",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14", str(dst),
    ]
    r = _run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg fallo al preparar el video:\n{r.stderr}")
    out = probe(dst)
    out["source_has_audio"] = info["has_audio"]
    out["source_duration"] = info["duration"]
    return out


def mux_audio(video_no_audio: str | Path, source: str | Path, dst: str | Path,
              duration: float) -> bool:
    """Copia el video generado y le pone el audio original (recortado). True si habia audio."""
    if not probe(source)["has_audio"]:
        Path(dst).write_bytes(Path(video_no_audio).read_bytes())
        return False
    cmd = [
        _FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_no_audio), "-t", f"{duration:.3f}", "-i", str(source),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(dst),
    ]
    r = _run(cmd)
    if r.returncode != 0:
        Path(dst).write_bytes(Path(video_no_audio).read_bytes())
        return False
    return True


def read_frames(path: str | Path):
    """Lee un video completo como lista de PIL.Image (RGB)."""
    import imageio.v2 as imageio
    from PIL import Image

    reader = imageio.get_reader(str(path))
    try:
        return [Image.fromarray(f).convert("RGB") for f in reader]
    finally:
        reader.close()
