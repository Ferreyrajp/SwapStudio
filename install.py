"""Instalador multiplataforma (Windows / Linux).  Uso:  python install.py [--cpu] [--no-bnb]

Crea .venv, instala PyTorch (CUDA 12.8 por defecto, necesario para RTX serie 50),
las dependencias, SAM2, y clona el repo oficial de Wan2.2 para el preprocesado.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
SAM2_URL = "https://github.com/facebookresearch/sam2.git"
SAM2_COMMIT = "0e78a118995e66bb27d78518c4bd9a3e95b4e266"


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd: list[str], check: bool = True) -> int:
    print("\n>", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd])
    if check and r.returncode != 0:
        sys.exit(f"Fallo: {' '.join(str(c) for c in cmd)}")
    return r.returncode


def patch_sam2(src: Path) -> None:
    """Prepara SAM2 para instalarlo SIN compilar su extension CUDA.

    Ese commit de SAM2 siempre intenta compilar `sam2._C` y falla si el CUDA Toolkit instalado
    en el sistema (p. ej. 13.x) no coincide con el de PyTorch (12.8). La extension solo sirve
    para rellenar huecos pequenos en las mascaras; sin ella se omite ese paso (efecto menor,
    y es lo que hace SAM2 en versiones nuevas cuando la extension no esta).
    Ademas declara los .yaml de configuracion como datos del paquete.
    """
    setup = src / "setup.py"
    text = setup.read_text(encoding="utf-8")
    text = text.replace("ext_modules=get_extensions(),", "ext_modules=[],")
    text = text.replace('cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},', "")
    # El commit de SAM2 no empaqueta sus .yaml (sam2_configs/*.yaml): sin ellos falla con
    # "Cannot find primary config 'sam2_hiera_l.yaml'".
    pkg = 'packages=find_packages(exclude="notebooks"),'
    if "package_data" not in text:
        if pkg not in text:
            sys.exit("No pude parchear setup.py de SAM2 (no coincide con el commit esperado).")
        text = text.replace(pkg, pkg + '\n    package_data={"sam2_configs": ["*.yaml"]},')
    setup.write_text(text, encoding="utf-8")

    misc = src / "sam2" / "utils" / "misc.py"
    text = misc.read_text(encoding="utf-8")
    marker = "except ImportError:  # extension sam2._C"
    old = "    labels, areas = get_connected_components(mask <= 0)\n"
    new = (
        "    try:\n"
        "        labels, areas = get_connected_components(mask <= 0)\n"
        "    " + marker + " no compilada: se omite el relleno de huecos\n"
        "        return mask\n"
    )
    if marker not in text:
        if old not in text:
            sys.exit("No pude parchear SAM2 (el codigo no coincide con el commit esperado).")
        text = text.replace(old, new)
    misc.write_text(text, encoding="utf-8")


def install_sam2(py: Path, third_party: Path) -> None:
    src = third_party / "sam2"
    if not src.exists():
        third_party.mkdir(exist_ok=True)
        run(["git", "clone", SAM2_URL, src])
    run(["git", "-C", src, "checkout", "-f", SAM2_COMMIT])
    patch_sam2(src)
    run([py, "-m", "pip", "install", "--no-build-isolation", src])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch-index", default=os.environ.get("TORCH_INDEX_URL",
                    "https://download.pytorch.org/whl/cu128"),
                    help="Indice de wheels de PyTorch (cu128 soporta Blackwell / RTX 50).")
    ap.add_argument("--cpu", action="store_true", help="Instalar PyTorch sin CUDA (solo pruebas).")
    ap.add_argument("--no-bnb", action="store_true", help="No instalar bitsandbytes (modo NF4).")
    args = ap.parse_args()

    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        print(f"AVISO: Python {sys.version_info.major}.{sys.version_info.minor}. "
              "Se recomienda 3.10-3.13.")

    for tool in ("git",):
        if run([tool, "--version"], check=False) != 0:
            sys.exit(f"Necesitas instalar `{tool}` y tenerlo en el PATH.")

    if not venv_python().exists():
        print("Creando entorno virtual en .venv ...")
        venv.EnvBuilder(with_pip=True).create(VENV)
    py = venv_python()
    run([py, "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"])

    torch_cmd = [py, "-m", "pip", "install", "torch", "torchvision"]
    if not args.cpu:
        torch_cmd += ["--index-url", args.torch_index]
    run(torch_cmd)

    run([py, "-m", "pip", "install", "-r", ROOT / "requirements.txt"])

    # onnxruntime: version GPU si se puede, si no CPU (el preprocesado funciona igual, mas lento)
    if run([py, "-m", "pip", "install", "onnxruntime-gpu"], check=False) != 0:
        run([py, "-m", "pip", "install", "onnxruntime"])

    # decord es opcional: si no hay wheel, la app usa un reemplazo con OpenCV (shims/decord)
    run([py, "-m", "pip", "install", "decord"], check=False)

    if not args.no_bnb:
        run([py, "-m", "pip", "install", "bitsandbytes"], check=False)

    sys.path.insert(0, str(ROOT))
    from swapstudio import config

    # SAM2 local y parcheado (sin extension CUDA)
    install_sam2(py, config.THIRD_PARTY)

    # Repo oficial de Wan2.2 (solo se usa el script de preprocesado)

    if not config.WAN_REPO_DIR.exists():
        config.THIRD_PARTY.mkdir(exist_ok=True)
        run(["git", "clone", config.WAN_REPO_URL, config.WAN_REPO_DIR])
    run(["git", "-C", config.WAN_REPO_DIR, "checkout", config.WAN_REPO_COMMIT])

    for d in (config.MODELS_DIR, config.OUTPUTS_DIR, config.WORK_DIR):
        d.mkdir(exist_ok=True)

    print("\nInstalacion lista.")
    print("Siguiente paso (descarga ~25-35 GB, una sola vez):")
    print("   Windows:  .venv\\Scripts\\python -m swapstudio.models --quant Q4_K_M")
    print("   Linux:    .venv/bin/python -m swapstudio.models --quant Q4_K_M")
    print("Luego ejecuta run.bat (Windows) o ./run.sh (Linux).")


if __name__ == "__main__":
    main()
