# SwapStudio

**Character swap en video, 100 % local y sin ComfyUI.** Subes un video y una imagen de un
personaje, y obtienes el mismo video con ese personaje en lugar de la persona original.
Interfaz web con Gradio, pensada para PCs modestas (probado en una RTX 5060 de 16 GB de VRAM
con 32 GB de RAM).

*English: local, standalone (no ComfyUI) video character replacement with Gradio, built around
Wan 2.2 Animate GGUF quantizations to run on 16 GB GPUs.*

> **Motor: Wan 2.2 Animate (modo replacement)**, no MiniMax H3. El swap con H3 necesita el video
> como referencia y hoy eso no es viable en una PC de 32 GB de RAM (ver [Por que no H3](#por-que-no-h3)).

## Estado

| | |
|---|---|
| Windows 11 + RTX 5060 16 GB + Python 3.13 | **Probado de punta a punta** |
| Linux | No probado todavia (el codigo es multiplataforma por diseno; reportes bienvenidos) |
| GPU con menos de 16 GB | No probado; ver las cuantizaciones mas chicas |

Medicion real (4 s a 16 fps, 640x352, `Q3_K_M`, 20 pasos): ~30 s por paso, ~10 min de generacion,
~15.3 GB de VRAM en uso. La primera corrida suma ~3 min de carga de modelos y ~1 min de preprocesado.

## Requisitos

- GPU NVIDIA con 16 GB de VRAM (RTX serie 50 incluida: se instala PyTorch con CUDA 12.8).
- Python 3.10 a 3.13 y `git` en el PATH.
- ~45 GB libres en disco (modelos + entorno).

## Instalacion

```bash
git clone https://github.com/Ferreyrajp/SwapStudio.git
cd SwapStudio
python install.py            # Windows: install.bat | Linux: ./install.sh
```

Crea `.venv`, instala PyTorch + dependencias + SAM2 y clona el repo oficial de Wan2.2 (solo se
usa su script de preprocesado). Se puede volver a ejecutar si se corta: salta lo ya instalado.

Detalles que el instalador resuelve por ti:
- SAM2 se instala desde `third_party/sam2` **sin compilar su extension CUDA** (falla si el CUDA
  Toolkit del sistema, p. ej. 13.x, no coincide con el de PyTorch). Solo se pierde el relleno de
  huecos pequenos en la mascara.
- Se declaran los `.yaml` de SAM2, que su `setup.py` no empaqueta.
- Si `decord` no tiene wheel para tu sistema, se usa un reemplazo con OpenCV (`shims/decord`).

## Descargar modelos (una vez)

```bash
.venv\Scripts\python -m swapstudio.models --quant Q3_K_M     # Windows
.venv/bin/python -m swapstudio.models --quant Q3_K_M         # Linux
```

Baja el transformer GGUF, los componentes base (texto, VAE, CLIP) y los modelos de preprocesado
(deteccion, pose, SAM2): unos 25-35 GB. Si algo falta, la app lo descarga al primer uso.
Opcional: define `HF_TOKEN` para descargas mas rapidas.

| Cuantizacion | Tamano | Nota |
|---|---|---|
| `Q2_K` | 6.5 GB | Calidad baja; para ~8 GB de VRAM |
| **`Q3_K_M`** | **8.6 GB** | **Recomendada para 16 GB: es la que se probo** |
| `Q4_K_M` | 11.5 GB | En 16 GB quedo al limite (ver [Solucion de problemas](#solucion-de-problemas)) |
| `Q5_K_M` / `Q6_K` | 13 / 14.6 GB | Para 20-24 GB |
| `Q8_0` | 18.7 GB | Para 24 GB o mas |

## Uso

```bash
run.bat        # Windows
./run.sh       # Linux
```

Sube un video con **una sola persona** y una imagen del personaje (cuerpo visible, frontal si se
puede), y pulsa *Generar swap*. El resultado queda en `outputs/`; los intermedios (pose, mascara,
fondo) en `work/<fecha>/pre/`, utiles para diagnosticar resultados raros. Las vistas previas
tienen boton de pantalla completa.

### Como elegir los ajustes en 16 GB

La memoria y el tiempo dependen de los **tokens por bloque** (frames latentes x pixeles/256).
Con `Q3_K_M`, ~15 mil tokens entran en 16 GB; ~30 mil, no. Por eso, con *Frames por bloque = auto*,
la app parte el video en bloques que caben y te muestra el plan y una estimacion antes de empezar.

- Prueba primero con **4 s**, resolucion *Borrador (640x360)* y **8-12 pasos**.
- Un clip de 10 s vertical en *Borrador* son ~3 bloques: orden de ~20 min con 12 pasos.
- Entre bloques puede notarse un salto de aspecto; prueba *5 frames de continuidad*.
- `Offload: sequential` ahorra mas VRAM pero es mucho mas lento y puede no funcionar con GGUF.

## Solucion de problemas

**La generacion se queda en `0/20` (la VRAM marca 15.9/15.9 GB).** La configuracion no cabe.
Baja resolucion, segundos o cuantizacion. Ademas, en Windows conviene desactivar el desborde a
memoria compartida: Panel de control de NVIDIA > Administrar configuracion 3D >
*CUDA - Politica de la memoria del sistema de reserva* > *No prefiero la memoria del sistema de
reserva*, aplicado al `python.exe` del `.venv` (y a la instalacion real de Python, porque el del
`.venv` puede ser solo un lanzador). Asi, si no cabe, falla con un error de memoria en vez de
arrastrarse. En el registro, las lineas `[monitor]` muestran la VRAM y cada paso muestra su tiempo.

**`Cannot find primary config 'sam2_hiera_l.yaml'`.** Vuelve a ejecutar `install.py`
(reinstala SAM2 con sus configuraciones).

**Error al cargar el GGUF.** La app sugiere la opcion `nf4` (descarga el transformer completo y
lo cuantiza con bitsandbytes; es mucho mas pesada). No esta probada.

**`Sin memoria de GPU`.** Baja resolucion/segundos, o usa `Q2_K`.

El aviso `Some weights ... motion_encoder ... kernel were not used` aparece al cargar el GGUF y no
impidio obtener resultados; si ves movimientos anomalos, abre un issue.

## Estructura

```
app.py                      interfaz Gradio
install.py / .bat / .sh     instalador multiplataforma
run.bat / run.sh            lanzadores
swapstudio/
  backends/base.py          interfaz comun de motores (SwapBackend, SwapParams)
  backends/wan_animate.py   preprocesado oficial + WanAnimatePipeline (diffusers) + planificador de bloques
  backends/minimax_h3.py    hueco reservado para H3
  models.py                 descarga de modelos desde Hugging Face
  video_io.py               ffmpeg (via imageio-ffmpeg; no requiere instalarlo)
  hw.py, config.py          deteccion de hardware y constantes
shims/decord/               reemplazo de decord con OpenCV
```

## Por que no H3

El swap con MiniMax H3 necesita el video como referencia. Segun la ficha de Hyperflow H3 Flash
(la version ligera de 14B), los runtimes GGUF solo aceptan imagenes de referencia; el video
requiere el paquete Python INT8, que pide >= 69 GiB de RAM con offload. Ademas, las LoRAs de
character swap estan hechas para el H3 completo. `backends/minimax_h3.py` queda como hueco para
implementarlo cuando haya hardware o modelos compatibles. H3 no esta licenciado en EE. UU., UE,
Reino Unido ni Corea del Sur sin autorizacion de MiniMax.

## Limitaciones conocidas

- El modo replacement funciona bien con **una sola persona** en el video.
- No incluye la LoRA de relighting del flujo oficial de Wan.
- El audio original se reutiliza tal cual; no se genera audio.

## Creditos y licencias

Este repositorio (codigo propio) se publica bajo licencia MIT. Los modelos y proyectos que usa
tienen sus propias licencias; revisalas antes de un uso comercial:

- [Wan 2.2 / Wan2.2-Animate](https://github.com/Wan-Video/Wan2.2) (Alibaba Wan-AI): modelo y script de preprocesado.
- [QuantStack/Wan2.2-Animate-14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-Animate-14B-GGUF): cuantizaciones GGUF.
- [SAM 2](https://github.com/facebookresearch/sam2) (Meta), [diffusers](https://github.com/huggingface/diffusers), [Gradio](https://github.com/gradio-app/gradio).

Este proyecto no incluye ni redistribuye pesos de modelos: se descargan desde sus repositorios
originales. Usa la herramienta con consentimiento de las personas involucradas y respetando las
leyes sobre imagen y suplantacion de identidad.
