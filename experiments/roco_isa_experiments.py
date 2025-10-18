# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📑 Experimentos ISA en ROCO (WhyXrayCLIP & PubMedCLIP)
#
# Este cuaderno unificado agrupa los flujos de trabajo necesarios para
# reproducir los experimentos de *Image-Sentence Alignment (ISA)* sobre el
# dataset **ROCO**, utilizando los modelos **WhyXrayCLIP** y **PubMedCLIP**.
#
# ➕ Ventajas de esta versión consolidada:
#
# - Todas las utilidades de ambos experimentos residen en un único archivo.
# - Es sencillo conmutar entre modelos sin duplicar código.
# - Las celdas están listas para exportarse a Google Colab mediante Jupytext.
#
# Ejecuta las celdas en orden y selecciona el modelo deseado en la sección de
# configuración.

# %% [markdown]
# ## 1. Clonar/actualizar repositorio
#
# La siguiente celda clona el repositorio cuando se ejecuta en Google Colab. En
# entornos locales ya dentro del repositorio, simplemente informa la ruta
# detectada.

# %%
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Alberto-97sc/mmshap_medclip.git"
DEFAULT_BRANCH = "main"
IS_COLAB = "google.colab" in sys.modules
LOCAL_DIR = Path("/content/mmshap_medclip") if IS_COLAB else Path.cwd()

if IS_COLAB:
    os.chdir("/content")
    if not (LOCAL_DIR / ".git").is_dir():
        print("Clonando repositorio...")
        subprocess.run(["git", "clone", REPO_URL, str(LOCAL_DIR)], check=True)
    else:
        print("Repositorio encontrado; actualizando rama principal...")
        subprocess.run(["git", "-C", str(LOCAL_DIR), "fetch", "origin"], check=True)
        subprocess.run(["git", "-C", str(LOCAL_DIR), "checkout", DEFAULT_BRANCH], check=True)
        subprocess.run(
            ["git", "-C", str(LOCAL_DIR), "reset", "--hard", f"origin/{DEFAULT_BRANCH}"],
            check=True,
        )
    os.chdir(str(LOCAL_DIR))
else:
    print(f"Repositorio local detectado en {LOCAL_DIR}")

# %% [markdown]
# ## 2. Montar Google Drive (solo Colab)

# %%
try:  # noqa: SIM105 (módulo opcional)
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
except ModuleNotFoundError:
    print("Google Drive no está disponible fuera de Google Colab.")

# %% [markdown]
# ## 3. Asegurar instalación editable de `mmshap_medclip`
#
# Esta celda comprueba si el paquete está importable; en caso contrario lo
# instala en modo editable usando `pip install -e`.

# %%
import importlib

repo_root = LOCAL_DIR.resolve()
try:
    importlib.import_module("mmshap_medclip")
    print("mmshap_medclip ya estaba disponible en la sesión.")
except ModuleNotFoundError:
    print("Instalando mmshap_medclip en modo editable...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(repo_root)], check=True)

# %% [markdown]
# ## 4. Configuración de experimentos y utilidades comunes

# %%
from functools import lru_cache
from typing import Any, Dict, Iterable, Sequence

import torch
import torch.nn.functional as F
from IPython.display import display

from mmshap_medclip.devices import get_device
from mmshap_medclip.io_utils import load_config
from mmshap_medclip.registry import build_dataset, build_model
from mmshap_medclip.tasks.isa import run_isa_one
from mmshap_medclip.tasks.whyxrayclip import (
    filter_roco_by_keywords,
    pick_chestxray_sample,
    sample_negative_captions,
    score_alignment as score_alignment_openclip,
)

DEFAULT_KEYWORDS = ("chest x-ray", "lung")
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "pubmedclip": {
        "cfg_path": str(repo_root / "configs/roco_isa_pubmedclip.yaml"),
        "description": "Evaluación ISA con PubMedCLIP",
        "filter_before_use": False,
        "keywords": DEFAULT_KEYWORDS,
    },
    "whyxrayclip": {
        "cfg_path": str(repo_root / "configs/roco_isa_whyxrayclip.yaml"),
        "description": "Evaluación ISA con WhyXrayCLIP",
        "filter_before_use": True,
        "keywords": DEFAULT_KEYWORDS,
    },
}


@lru_cache(maxsize=len(MODEL_CONFIGS))
def prepare_session(model_key: str) -> Dict[str, Any]:
    """Carga configuración, dataset y modelo para el experimento indicado."""

    if model_key not in MODEL_CONFIGS:
        raise KeyError(f"Modelo desconocido: {model_key!r}.")

    model_cfg = MODEL_CONFIGS[model_key]
    cfg = load_config(model_cfg["cfg_path"])
    device = get_device(cfg.get("device", "auto"))
    dataset = build_dataset(cfg["dataset"])
    model = build_model(cfg["model"], device=device)

    keywords = tuple(model_cfg.get("keywords", ()))
    if model_cfg.get("filter_before_use") and keywords:
        dataset = filter_roco_by_keywords(dataset, keywords=keywords)

    session = {
        "cfg": cfg,
        "device": device,
        "dataset": dataset,
        "model": model,
        "keywords": keywords,
        "description": model_cfg.get("description", model_key),
    }
    return session


def score_alignment_generic(
    model_wrapper,
    image,
    caption: str,
    device: torch.device,
    negatives: Iterable[str] | None = None,
    temperature: float = 100.0,
    amp_if_cuda: bool = True,
) -> Dict[str, Any]:
    """Calcula alineación imagen-texto con soporte para modelos HF y OpenCLIP."""

    negatives = list(negatives or [])
    processor = getattr(model_wrapper, "processor", None)
    if processor is None:
        raise AttributeError("El modelo no expone un processor compatible.")

    # Modelos basados en open_clip (WhyXrayCLIP) exponen `process_images`.
    if hasattr(processor, "process_images"):
        return score_alignment_openclip(
            model_wrapper,
            image,
            caption,
            device=device,
            negatives=negatives,
            temperature=temperature,
            amp_if_cuda=amp_if_cuda,
        )

    # Ramas CLIP de HuggingFace (PubMedCLIP, CLIP base, etc.).
    texts = [caption] + negatives
    batch = processor(text=texts, images=[image], return_tensors="pt", padding=True)
    pixel_values = batch["pixel_values"].to(device)
    text_inputs = {k: v.to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask"}}

    with torch.inference_mode():
        image_features = model_wrapper.model.get_image_features(pixel_values=pixel_values)
        text_features = model_wrapper.model.get_text_features(**text_inputs)

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    sims = (image_features @ text_features.T)[0]
    sim_pos = float(sims[0].item())
    score_01 = (sim_pos + 1.0) / 2.0

    if not negatives:
        return {"similarity": sim_pos, "score_01": score_01, "negatives": []}

    logits = temperature * sims.unsqueeze(0)
    probs = torch.softmax(logits, dim=-1)[0]

    candidates = texts
    ranking = sorted(zip(candidates, probs.tolist()), key=lambda x: x[1], reverse=True)
    rank_true = 1 + next(i for i, (txt, _) in enumerate(ranking) if txt == caption)

    return {
        "similarity": sim_pos,
        "score_01": score_01,
        "candidates": candidates,
        "logits": logits.detach().cpu(),
        "probs": probs.detach().cpu(),
        "ranking": ranking,
        "rank_true": rank_true,
        "temperature": temperature,
    }


# %% [markdown]
# ## 5. Seleccionar modelo y preparar sesión
#
# Ajusta `MODEL_KEY` para cambiar entre WhyXrayCLIP y PubMedCLIP. Los recursos
# (dataset, modelo y dispositivo) se almacenan en caché para evitar recargas.

# %%
MODEL_KEY = "whyxrayclip"  # @param ["whyxrayclip", "pubmedclip"]
session = prepare_session(MODEL_KEY)
dataset = session["dataset"]
model = session["model"]
device = session["device"]

print(
    f"Modelo seleccionado: {MODEL_KEY}\n"
    f"Descripción: {session['description']}\n"
    f"Dataset: {len(dataset)} muestras | Device: {device}"
)

# %% [markdown]
# ## 6. Ejecutar SHAP en una muestra del dataset
#
# Configura `SAMPLE_INDEX` para elegir la muestra sobre la que se calcularán las
# explicaciones ISA.

# %%
SAMPLE_INDEX = 154  # @param {type:"integer"}
EXPLAIN = True  # @param {type:"boolean"}
PLOT = True  # @param {type:"boolean"}

sample = dataset[SAMPLE_INDEX]
image, caption = sample["image"], sample["text"]
result = run_isa_one(model, image, caption, device, explain=EXPLAIN, plot=PLOT)

print(
    f"logit={result['logit']:.4f}  "
    f"TScore={result['tscore']:.2%}  "
    f"IScore={result['iscore']:.2%}"
)

# %% [markdown]
# ## 7. Muestreo dirigido y evaluación de alineación
#
# Esta sección replica las utilidades de ambos experimentos originales:
#
# - Selecciona captions relacionadas con radiografías de tórax mediante
#   `pick_chestxray_sample`.
# - Permite muestrear negativos opcionales y calcular la alineación
#   imagen–texto con el modelo activo (WhyXrayCLIP o PubMedCLIP).

# %%
USE_NEGATIVES = False  # @param {type:"boolean"}
NUM_NEGATIVES = 15  # @param {type:"integer"}
REPRODUCIBLE = False  # @param {type:"boolean"}
SEED = 42  # @param {type:"integer"}
TEMPERATURE = 100.0  # @param {type:"number"}
KEYWORDS: Sequence[str] = session.get("keywords") or DEFAULT_KEYWORDS

sample_info = pick_chestxray_sample(
    dataset,
    keywords=KEYWORDS,
    reproducible=REPRODUCIBLE,
    seed=SEED,
)
image = sample_info["image"]
caption_clean = sample_info["caption_clean"]

print(f"Índice filtrado: {sample_info['index']} | Índice original: {sample_info['source_index']}")
print("Caption limpio:", caption_clean)
display(image)

negatives = []
if USE_NEGATIVES:
    negatives = sample_negative_captions(
        dataset,
        positive_caption=caption_clean,
        k=NUM_NEGATIVES,
        reproducible=REPRODUCIBLE,
        seed=SEED,
    )
    print(f"Negativos muestreados: {len(negatives)}")

scores = score_alignment_generic(
    model,
    image,
    caption_clean,
    device=device,
    negatives=negatives,
    temperature=TEMPERATURE,
)

print("\n=== Alineación imagen–caption ===")
print(f"Similitud coseno: {scores['similarity']:.4f}   |   Score[0,1]: {scores['score_01']:.4f}")

if negatives:
    print("\n=== Zero-shot elección (con negativos) ===")
    print(f"Total candidatos: {len(negatives) + 1}")
    print(f"Rank del caption verdadero: {scores['rank_true']}")
    print("\nTop-5 candidatos por probabilidad:")
    for lbl, prob in scores["ranking"][:5]:
        short = (lbl[:120] + "…") if len(lbl) > 120 else lbl
        print(f"{prob:7.4f} | {short}")
else:
    print("(Sin negativos adicionales en esta corrida.)")
