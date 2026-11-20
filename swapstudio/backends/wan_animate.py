"""Backend Wan 2.2 Animate en modo *replacement*.

Flujo:
  1. ffmpeg: recorta / remuestrea fps / reduce el video fuente.
  2. Preprocesado OFICIAL de Wan2.2 (subproceso): pose, cara, fondo y mascara del personaje.
  3. diffusers `WanAnimatePipeline` con el transformer cuantizado (GGUF o NF4) y offload a CPU.
  4. Exporta el mp4 y le vuelve a poner el audio original.
"""
from __future__ import annotations

import collections
import gc
import importlib.util
import math
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

from .. import config, models, video_io
from .base import Log, Progress, SwapBackend, SwapParams

PRE_OUTPUTS = ("src_pose.mp4", "src_face.mp4", "src_bg.mp4", "src_mask.mp4")

# Medicion real en RTX 5060 (16 GB), Q3_K_M, 640x352, 64 frames: ~15.1k tokens por bloque,
# 15.3 GB de VRAM en uso y ~30 s por paso. Con ~30k tokens (160 frames a 464x832 en un
# solo bloque de 77) la VRAM se llena al 100 % y no completa ni un paso.
REF_TOKENS = 15100
REF_STEP_S = 30.0


def plan_segments(n_frames: int, width: int, height: int, segment_len: int,
                  prev_frames: int, budget: int) -> dict:
    """Decide el tamano de bloque. Tokens por bloque = frames latentes x (ancho/16) x (alto/16)."""
    per_latent = (width // 16) * (height // 16)
    if segment_len and segment_len > 0:
        seg = max(5, (segment_len - 1) // 4 * 4 + 1)
    else:
        max_latent = max(2, budget // per_latent)
        seg = (max_latent - 1) * 4 + 1
    eff = min(seg, n_frames)
    latents = math.ceil((eff - 1) / 4) + 1
    tokens = latents * per_latent
    new_per_seg = max(1, seg - prev_frames)
    n_seg = 1 if n_frames <= seg else 1 + math.ceil((n_frames - seg) / new_per_seg)
    step_s = REF_STEP_S * (tokens / REF_TOKENS) ** 1.4   # estimacion gruesa
    return {"seg": seg, "tokens": tokens, "n_seg": n_seg, "step_s": step_s}


class WanAnimateBackend(SwapBackend):
    id = "wan-animate"
    label = "Wan 2.2 Animate (replacement)"

    def __init__(self) -> None:
        self._pipe = None
        self._pipe_key: tuple | None = None

    # ------------------------------------------------------------------ estado
    def available(self) -> tuple[bool, str]:
        try:
            import torch
            from diffusers import WanAnimatePipeline  # noqa: F401
        except ImportError as e:
            return False, f"Faltan dependencias ({e}). Ejecuta `python install.py`."
        if not torch.cuda.is_available():
            return False, "PyTorch no detecta ninguna GPU CUDA."
        if not config.PREPROCESS_SCRIPT.exists():
            return False, "Falta third_party/Wan2.2. Ejecuta `python install.py`."
        return True, ""

    # --------------------------------------------------------------------- run
    def run(self, p: SwapParams, log: Log, progress: Progress) -> Path:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(why)

        import torch
        from PIL import Image

        stamp = time.strftime("%Y%m%d-%H%M%S")
        work = config.WORK_DIR / stamp
        work.mkdir(parents=True, exist_ok=True)
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        # 1) video e imagen de entrada
        progress(0.02, "Preparando video")
        src = work / "input.mp4"
        info = video_io.prepare_input(p.video, src, p.max_seconds, p.fps, p.resolution_area)
        log(f"Video preparado: {info['width']}x{info['height']} a {info['fps']:.0f} fps, "
            f"{info['duration']:.1f} s")
        ref = work / "ref.png"
        Image.open(p.image).convert("RGB").save(ref)

        # 2) modelos de preprocesado + preprocesado
        progress(0.05, "Verificando modelos de preprocesado")
        ckpt = models.ensure_process_checkpoint(log)
        progress(0.08, "Preprocesando (pose, cara, mascara)")
        pre = work / "pre"
        self._preprocess(src, ref, ckpt, pre, p, log)

        # 3) pipeline
        progress(0.30, "Cargando modelo")
        pipe = self._get_pipe(p.weights, p.offload, log)

        # 4) frames condicionantes
        pose = video_io.read_frames(pre / "src_pose.mp4")
        face = video_io.read_frames(pre / "src_face.mp4")
        bg = video_io.read_frames(pre / "src_bg.mp4")
        mask = video_io.read_frames(pre / "src_mask.mp4")
        n = min(len(pose), len(face), len(bg), len(mask))
        w, h = pose[0].size
        w, h = w // 16 * 16, h // 16 * 16

        def fit(frames):
            return [f if f.size == (w, h) else f.resize((w, h), Image.LANCZOS) for f in frames[:n]]

        pose, bg, mask = fit(pose), fit(bg), fit(mask)
        face = face[:n]
        ref_path = pre / "src_ref.png"
        ref_img = Image.open(ref_path if ref_path.exists() else ref).convert("RGB")
        log(f"Generando {n} frames a {w}x{h} ({n / p.fps:.1f} s)")

        plan = plan_segments(n, w, h, p.segment_len, p.prev_frames, p.tokens_budget)
        seg, n_seg = plan["seg"], plan["n_seg"]
        total_steps = n_seg * p.steps
        est_min = plan["step_s"] * total_steps / 60
        log(f"Plan: {n_seg} bloque(s) de hasta {seg} frames, ~{plan['tokens'] / 1000:.1f}k tokens por "
            f"bloque (referencia que funciono en 16 GB con Q3_K_M: ~{REF_TOKENS / 1000:.0f}k).")
        log(f"Estimacion MUY aproximada: ~{plan['step_s']:.0f} s/paso x {total_steps} pasos = ~{est_min:.0f} min")
        if plan["tokens"] > 1.1 * p.tokens_budget:
            log("AVISO: este bloque es mas grande de lo que entra en 16 GB; lo mas probable es que se "
                "llene la VRAM y no avance. Baja los frames por bloque o la resolucion.")
        state = {"done": 0, "last": time.time()}

        def on_step(_pipe, _i, _t, kwargs):
            now = time.time()
            state["done"] += 1
            dt = now - state["last"]
            log(f"Paso {state['done']} completado en {dt:.0f} s")
            hist = state.setdefault("hist", [])
            if len(hist) >= 3 and dt > 2.5 * sorted(hist[1:] or hist)[len(hist[1:] or hist) // 2]:
                log("AVISO: este paso fue mucho mas lento que los anteriores. Puede ser "
                    "desbordamiento de VRAM; si se repite, baja resolucion, segundos o cuantizacion.")
            hist.append(dt)
            state["last"] = now
            frac = 0.32 + 0.63 * min(1.0, state["done"] / total_steps)
            progress(frac, f"Muestreo: paso {state['done']} (aprox. de {total_steps})")
            return kwargs

        seed = p.seed if p.seed >= 0 else random.randint(0, 2**31 - 1)
        log(f"Semilla: {seed}")
        t0 = time.time()
        state["last"] = t0
        stop = threading.Event()
        threading.Thread(target=self._monitor, args=(log, stop, t0, state), daemon=True).start()
        try:
            out = pipe(
                image=ref_img, pose_video=pose, face_video=face,
                background_video=bg, mask_video=mask,
                prompt=p.prompt or config.DEFAULT_PROMPT,
                height=h, width=w, segment_frame_length=seg,
                guidance_scale=1.0, num_inference_steps=p.steps,
                mode="replace", prev_segment_conditioning_frames=p.prev_frames,
                generator=torch.Generator("cpu").manual_seed(seed),
                callback_on_step_end=on_step,
            )
        except torch.cuda.OutOfMemoryError as e:
            self._free()
            raise RuntimeError(
                "Sin memoria de GPU. Prueba: resolucion mas baja, menos segundos, una "
                "cuantizacion mas chica (Q3_K_M/Q2_K) o offload 'sequential'."
            ) from e
        finally:
            stop.set()
        frames = out.frames[0]

        # 5) exportar
        progress(0.96, "Exportando video")
        from diffusers.utils import export_to_video

        raw = work / "result_noaudio.mp4"
        export_to_video(frames, str(raw), fps=p.fps)
        final = config.OUTPUTS_DIR / f"swap-{stamp}.mp4"
        if p.keep_audio:
            had = video_io.mux_audio(raw, p.video, final, duration=len(frames) / p.fps)
            log("Audio original agregado." if had else "El video fuente no tiene audio.")
        else:
            final.write_bytes(raw.read_bytes())
        progress(1.0, "Listo")
        log(f"Guardado en {final}")
        return final

    # ------------------------------------------------------------ preprocesado
    def _preprocess(self, video: Path, ref: Path, ckpt: Path, save: Path,
                    p: SwapParams, log: Log) -> None:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        paths = [str(config.WAN_REPO_DIR)]
        if importlib.util.find_spec("decord") is None:
            paths.append(str(config.SHIMS_DIR))
            log("decord no esta instalado: uso el reemplazo basado en OpenCV.")
        env["PYTHONPATH"] = os.pathsep.join(paths + [env["PYTHONPATH"]] if env.get("PYTHONPATH") else paths)

        cmd = [
            sys.executable, str(config.PREPROCESS_SCRIPT),
            "--ckpt_path", str(ckpt), "--video_path", str(video),
            "--refer_path", str(ref), "--save_path", str(save),
            "--resolution_area", str(p.resolution_area[0]), str(p.resolution_area[1]),
            "--fps", str(p.fps),
            "--iterations", str(p.mask_iterations), "--k", str(p.mask_k),
            "--w_len", str(p.mask_w_len), "--h_len", str(p.mask_h_len),
            "--replace_flag",
        ]
        tail: collections.deque[str] = collections.deque(maxlen=15)
        proc = subprocess.Popen(
            cmd, cwd=config.WAN_REPO_DIR, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if line:
                tail.append(line[:300])
                log("[preproceso] " + line[:300])
        code = proc.wait()
        missing = [f for f in PRE_OUTPUTS if not (save / f).exists()]
        if code != 0 or missing:
            raise RuntimeError(
                f"El preprocesado fallo (codigo {code}, faltan: {missing or 'nada'}).\n"
                "Ultimas lineas:\n" + "\n".join(tail) +
                "\n\nRecuerda: el modo replacement solo funciona bien con UNA persona en el video."
            )

    @staticmethod
    def _monitor(log: Log, stop: threading.Event, t0: float, state: dict) -> None:
        """Cada 30 s informa la VRAM usada (PyTorch guarda memoria en cache, asi que verla
        casi llena es normal; la senal de desbordamiento es que los pasos se vuelvan mucho
        mas lentos, y eso se avisa en on_step)."""
        import torch

        while not stop.wait(30):
            free, total = torch.cuda.mem_get_info()
            used = (total - free) / 1024**3
            msg = (f"[monitor] {time.time() - t0:.0f} s - VRAM en uso {used:.1f}/"
                   f"{total / 1024**3:.1f} GB - pasos completados: {state['done']}")
            log(msg)

    # ---------------------------------------------------------------- pipeline
    def _get_pipe(self, weights: str, offload: str, log: Log):
        key = (weights, offload)
        if self._pipe is not None and self._pipe_key == key:
            return self._pipe
        self._free()

        import torch
        from diffusers import WanAnimatePipeline, WanAnimateTransformer3DModel

        dtype = torch.bfloat16
        try:
            if weights == "nf4":
                from diffusers import BitsAndBytesConfig

                base = models.ensure_base_components(include_transformer=True, log=log)
                log("Cargando transformer en NF4 (bitsandbytes)...")
                transformer = WanAnimateTransformer3DModel.from_pretrained(
                    str(base), subfolder="transformer", torch_dtype=dtype,
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype),
                )
            else:
                from diffusers import GGUFQuantizationConfig

                quant = weights.split(":", 1)[1]
                base = models.ensure_base_components(include_transformer=False, log=log)
                gguf = models.ensure_gguf(quant, log)
                log(f"Cargando transformer GGUF {quant}...")
                transformer = WanAnimateTransformer3DModel.from_single_file(
                    str(gguf), config=str(base), subfolder="transformer", torch_dtype=dtype,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                )
        except Exception as e:
            raise RuntimeError(
                f"No pude cargar el transformer ({weights}): {e}\n"
                "Si el problema es el GGUF, prueba la opcion 'nf4' (descarga mas pesada)."
            ) from e

        log("Armando pipeline...")
        pipe = WanAnimatePipeline.from_pretrained(str(base), transformer=transformer, torch_dtype=dtype)
        pipe.vae.to(torch.float32)   # recomendado en la documentacion de diffusers
        if offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()
        for fn in ("enable_tiling", "enable_slicing"):
            try:
                getattr(pipe.vae, fn)()
            except Exception:
                pass
        self._pipe, self._pipe_key = pipe, key
        return pipe

    def _free(self) -> None:
        self._pipe = None
        self._pipe_key = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
