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

# %% [markdown] id="9b467b94"
# # 📑 Evaluación de WhyXrayCLIP en ROCO
#
# Este notebook forma parte del proyecto de tesis sobre **medición del balance multimodal en modelos CLIP aplicados a dominios médicos**.
#
# Modelo actual: **WhyXrayCLIP**
# Dataset: **ROCO (Radiology Objects in COntext)**
# Tarea: **ISA (Image-Sentence Alignment)**
# ---
#
#

# %% [markdown] id="10c7622f"
# ## Clonar repositorio

# %% id="87c53e96"
# 📌 Código
REPO_URL  = "https://github.com/Alberto-97sc/mmshap_medclip.git"
LOCAL_DIR = "/content/mmshap_medclip"
BRANCH    = "codex/adaptar-modelo-whyxrayclip-al-repositorio"

# %cd /content
import os, shutil, subprocess, sys

if not os.path.isdir(f"{LOCAL_DIR}/.git"):
    # No está clonado aún
    # !git clone $REPO_URL $LOCAL_DIR
else:
    # Ya existe: actualiza a la última versión del remoto
    # %cd $LOCAL_DIR
    # !git fetch origin
    # !git checkout $BRANCH
    # !git reset --hard origin/$BRANCH
# %cd $LOCAL_DIR
# !git rev-parse --short HEAD


# %% [markdown] id="0efe36cb"
# ## Instalar dependencias y montar google drive

# %% id="35d8329f"
from google.colab import drive; drive.mount('/content/drive')

# === Instalar en modo editable (pyproject.toml) ===
# %pip install -e /content/mmshap_medclip

# %% id="adb0cf00"
# 📌 Código
from google.colab import drive; drive.mount('/content/drive')

# %% [markdown] id="6f4ab762"
# ## Cargar modelos y datos

# %% id="b6485339"
# 📌 Código
CFG_PATH="/content/mmshap_medclip/configs/roco_isa_whyxrayclip.yaml"

from mmshap_medclip.tasks.whyxrayclip import filter_roco_by_keywords

CHESTXRAY_KEYWORDS = ("chest x-ray", "lung")

# Asegura que cfg, device, dataset y model estén listos en esta sesión
if not all(k in globals() for k in ("cfg", "device", "dataset", "model")):
    from mmshap_medclip.io_utils import load_config
    from mmshap_medclip.registry import build_dataset, build_model
    from mmshap_medclip.devices import get_device

    cfg = load_config(CFG_PATH)
    device  = get_device(cfg.get("device", "auto"))
    dataset = build_dataset(cfg["dataset"])
    model   = build_model(cfg["model"], device=device)

# Filtra a radiografías de tórax/pulmón
dataset = filter_roco_by_keywords(dataset, keywords=CHESTXRAY_KEYWORDS)

print("OK → len(dataset) =", len(dataset), "(subset radiografías)", "| device =", device)


# %% [markdown] id="55231ba0"
# ## Ejecutar SHAP en una muestra

# %% id="0d06718a"
# 📌 Código
from mmshap_medclip.tasks.isa import run_isa_one

muestra = 154
sample  = dataset[muestra]
image, caption = sample['image'], sample['text']

res = run_isa_one(model, image, caption, device, explain=True, plot=True)
print(f"logit={res['logit']:.4f}  TScore={res['tscore']:.2%}  IScore={res['iscore']:.2%}")


# %% id="tzOwIfL0D5-5"
# ==== Config ====
use_negatives = False      # True para zero-shot con negativos
num_negatives = 15
reproducible = False       # Si True, usará la semilla para muestreo determinista
seed = 42                  # Se usa solo cuando reproducible=True
keywords = CHESTXRAY_KEYWORDS
temperature = 100.0

# ==== Funciones auxiliares específicas de WhyXrayCLIP ====
from mmshap_medclip.tasks.whyxrayclip import (
    pick_chestxray_sample,
    sample_negative_captions,
    score_alignment,
)
from IPython.display import display

sample = pick_chestxray_sample(
    dataset,
    keywords=keywords,
    reproducible=reproducible,
    seed=seed,
)
image = sample['image']
caption_clean = sample['caption_clean']

print('Índice en dataset:', sample['index'])
print('Caption limpio:', caption_clean)
display(image)

negatives = []
if use_negatives:
    negatives = sample_negative_captions(
        dataset,
        positive_caption=caption_clean,
        k=num_negatives,
        reproducible=reproducible,
        seed=seed,
    )

scores = score_alignment(
    model,
    image,
    caption_clean,
    device=device,
    negatives=negatives,
    temperature=temperature,
    amp_if_cuda=True,
)

print('
=== Alineación imagen–caption ===')
print(f"Similitud coseno: {scores['similarity']:.4f}   |   Score[0,1]: {scores['score_01']:.4f}")

if negatives:
    print('
=== Zero-shot elección (con negativos) ===')
    print(f'Total candidatos: {len(negatives) + 1}')
    print(f"Rank del caption verdadero: {scores['rank_true']}")
    print('
Top-5 candidatos por probabilidad:')
    for lbl, p in scores['ranking'][:5]:
        short = (lbl[:120] + '…') if len(lbl) > 120 else lbl
        print(f"{p:7.4f} | {short}")
else:
    print('(Sin negativos adicionales en esta corrida.)')

