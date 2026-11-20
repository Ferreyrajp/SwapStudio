"""Interfaz Gradio de Swap Studio.  Ejecutar:  python app.py"""
from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path

import gradio as gr

from swapstudio import config, hw
from swapstudio.backends import BACKENDS, SwapParams

PREVIEW_H = 260      # alto (px) de las vistas previas de entrada
RESULT_H = 320       # alto (px) del resultado


def _fullscreen_js(elem_id: str) -> str:
    """JS que pone en pantalla completa el <video> de un componente (Gradio no trae el boton)."""
    return ("() => { const v = document.querySelector('#%s video'); "
            "if (v) { (v.requestFullscreen || v.webkitRequestFullscreen).call(v); } }" % elem_id)


HW = hw.detect()
DEF_WEIGHTS, DEF_PRESET = hw.recommend(HW.vram_gb)


def _backend_choices():
    out = []
    for b in BACKENDS.values():
        ok, why = b.available()
        out.append((b.label if ok else f"{b.label}  [no disponible]", b.id))
    return out


def run_swap(video, image, backend_id, prompt, preset, seconds, fps, steps, seed,
             weights, offload, mask_iter, mask_k, seg_choice, prev_choice, keep_audio,
             progress=gr.Progress()):
    if not video or not image:
        raise gr.Error("Sube un video y una imagen del personaje.")
    backend = BACKENDS[backend_id]
    ok, why = backend.available()
    if not ok:
        raise gr.Error(why)

    params = SwapParams(
        video=Path(video), image=Path(image), prompt=prompt,
        max_seconds=float(seconds), fps=int(fps),
        resolution_area=config.RESOLUTION_PRESETS[preset],
        steps=int(steps), seed=int(seed), weights=weights, offload=offload,
        mask_iterations=int(mask_iter), mask_k=int(mask_k), keep_audio=bool(keep_audio),
        segment_len=0 if seg_choice == "auto" else int(seg_choice), prev_frames=int(prev_choice),
    )

    q: queue.Queue = queue.Queue()
    result: dict = {}

    def worker():
        try:
            result["path"] = backend.run(
                params,
                log=lambda m: q.put(("log", m)),
                progress=lambda f, d: q.put(("prog", (f, d))),
            )
        except Exception as e:  # se muestra en la UI
            result["error"] = e
            q.put(("log", traceback.format_exc()))

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    lines: list[str] = []
    while t.is_alive() or not q.empty():
        try:
            kind, val = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if kind == "log":
            lines.append(val)
            yield "\n".join(lines[-300:]), None
        else:
            progress(val[0], desc=val[1])

    if "error" in result:
        raise gr.Error(str(result["error"])[:900])
    yield "\n".join(lines[-300:]), str(result["path"])


def build() -> gr.Blocks:
    with gr.Blocks(title="Swap Studio") as demo:
        gr.Markdown("# Swap Studio\nReemplaza al personaje de un video por el de una imagen. "
                    "100 % local, sin ComfyUI.")
        gr.Markdown(f"**Hardware detectado:** {HW.describe()}  \n"
                    f"**Sugerencia:** `{DEF_WEIGHTS}` + `{DEF_PRESET}`")

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column(min_width=200):
                        video = gr.Video(label="Video original (UNA sola persona)",
                                         height=PREVIEW_H, elem_id="in_video")
                        btn_fs_in = gr.Button("Video a pantalla completa", size="sm")
                    image = gr.Image(label="Personaje de reemplazo", type="filepath",
                                     height=PREVIEW_H, buttons=["fullscreen"], min_width=200)
                prompt = gr.Textbox(label="Prompt (opcional)", value=config.DEFAULT_PROMPT)
                run = gr.Button("Generar swap", variant="primary")
            with gr.Column():
                out_video = gr.Video(label="Resultado", height=RESULT_H, elem_id="out_video")
                btn_fs_out = gr.Button("Resultado a pantalla completa", size="sm")
                log = gr.Textbox(label="Registro", lines=14, max_lines=14, autoscroll=True)

        with gr.Accordion("Ajustes", open=False):
            backend = gr.Dropdown(_backend_choices(), value="wan-animate", label="Motor")
            with gr.Row():
                preset = gr.Dropdown(list(config.RESOLUTION_PRESETS), value=DEF_PRESET,
                                     label="Resolucion")
                seconds = gr.Slider(1, 10, value=4, step=0.5, label="Segundos a procesar")
                fps = gr.Slider(8, 30, value=16, step=1, label="FPS")
            with gr.Row():
                steps = gr.Slider(4, 30, value=20, step=1, label="Pasos (menos = borrador)")
                seed = gr.Number(value=-1, precision=0, label="Semilla (-1 = aleatoria)")
            with gr.Row():
                weights = gr.Dropdown(config.WEIGHT_CHOICES, value=DEF_WEIGHTS, label="Pesos")
                offload = gr.Dropdown(["model", "sequential"], value="model",
                                      label="Offload a CPU")
            with gr.Row():
                mask_iter = gr.Slider(1, 8, value=3, step=1, label="Mascara: dilatacion (iteraciones)")
                mask_k = gr.Slider(3, 15, value=7, step=2, label="Mascara: tamano de kernel")
            with gr.Row():
                seg_choice = gr.Dropdown(["auto", "29", "37", "45", "53", "61", "69", "77"], value="auto",
                                         label="Frames por bloque (auto = el maximo que entra en tu VRAM)")
                prev_choice = gr.Dropdown(["1", "5"], value="1", label="Frames de continuidad entre bloques")
            keep_audio = gr.Checkbox(value=True, label="Conservar el audio original")

        btn_fs_in.click(fn=None, js=_fullscreen_js("in_video"))
        btn_fs_out.click(fn=None, js=_fullscreen_js("out_video"))

        run.click(
            run_swap,
            [video, image, backend, prompt, preset, seconds, fps, steps, seed,
             weights, offload, mask_iter, mask_k, seg_choice, prev_choice, keep_audio],
            [log, out_video],
        )
    return demo


if __name__ == "__main__":
    build().queue().launch(inbrowser=True)
