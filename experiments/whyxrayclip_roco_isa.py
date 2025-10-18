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


