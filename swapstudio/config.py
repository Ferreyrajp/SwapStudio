from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
WORK_DIR = ROOT / "work"
THIRD_PARTY = ROOT / "third_party"
SHIMS_DIR = ROOT / "shims"

# Repo oficial de Wan2.2: de ahi sale el script de preprocesado (pose, cara, mascara).
WAN_REPO_DIR = THIRD_PARTY / "Wan2.2"
WAN_REPO_URL = "https://github.com/Wan-Video/Wan2.2.git"
WAN_REPO_COMMIT = "1ea34ff48f87168174e12956e200b1d908b1c5ff"  # lo fija install.py
PREPROCESS_SCRIPT = WAN_REPO_DIR / "wan" / "modules" / "animate" / "preprocess" / "preprocess_data.py"

# Repos de Hugging Face
DIFFUSERS_REPO = "Wan-AI/Wan2.2-Animate-14B-Diffusers"   # componentes (texto, VAE, CLIP, config)
ORIGINAL_REPO = "Wan-AI/Wan2.2-Animate-14B"              # modelos de preprocesado (pose, SAM2)
GGUF_REPO = "QuantStack/Wan2.2-Animate-14B-GGUF"         # transformer cuantizado

# Tamano de cada GGUF en GB (segun el listado del repo)
GGUF_SIZES_GB = {
    "Q2_K": 6.46,
    "Q3_K_M": 8.63,
    "Q4_K_M": 11.5,
    "Q5_K_M": 13.0,
    "Q6_K": 14.6,
    "Q8_0": 18.7,
}
WEIGHT_CHOICES = [f"gguf:{q}" for q in GGUF_SIZES_GB] + ["nf4"]

# resolution_area = (ancho, alto): el preprocesado conserva la proporcion del video
# y ajusta el AREA total a ancho*alto, con lados multiplo de 16.
RESOLUTION_PRESETS = {
    "Minimo (480x272)": (480, 272),
    "Borrador (640x360)": (640, 360),
    "Estandar (832x480)": (832, 480),
    "Alto (1024x576)": (1024, 576),
}

DEFAULT_PROMPT = "A person moving naturally, high quality, realistic."

# Archivos de preprocesado que esperamos dentro del repo original
PROCESS_FILES = {
    "det": "yolov10m.onnx",
    "pose2d": "vitpose_h_wholebody.onnx",
    "sam2": "sam2_hiera_large.pt",
}
