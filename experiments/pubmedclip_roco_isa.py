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
# # 📑 Evaluación de PubMedCLIP en ROCO
#
# Este notebook forma parte del proyecto de tesis sobre **medición del balance multimodal en modelos CLIP aplicados a dominios médicos**.
#
# **Modelo**: PubMedCLIP
# **Dataset**: ROCO (Radiology Objects in COntext)
# **Tarea**: ISA (Image-Sentence Alignment)
#
# ---

# %% [markdown]
# ## Cargar modelos y datos

# %%
import os
from pathlib import Path

# 📌 Configuración - Asegurar que estamos en el directorio correcto
# Funciona tanto en scripts como en notebooks
try:
    # En scripts Python
    PROJECT_ROOT = Path(__file__).parent.parent
except NameError:
    # En notebooks de Jupyter
    PROJECT_ROOT = Path.cwd()
    # Si estamos en experiments/, subir un nivel
    if PROJECT_ROOT.name == "experiments":
        PROJECT_ROOT = PROJECT_ROOT.parent

os.chdir(PROJECT_ROOT)
CFG_PATH = "configs/roco_isa_pubmedclip.yaml"

# Asegura que cfg, device, dataset y model estén listos en esta sesión
if not all(k in globals() for k in ("cfg", "device", "dataset", "model")):
    from mmshap_medclip.io_utils import load_config
    from mmshap_medclip.devices import get_device
    from mmshap_medclip.registry import build_dataset, build_model

    cfg = load_config(CFG_PATH)
    device = get_device()
    dataset = build_dataset(cfg["dataset"])
    model = build_model(cfg["model"], device=device)

print("OK → len(dataset) =", len(dataset), "| device =", device)

# %% [markdown]
# ## Ejecutar SHAP en una muestra

# %%
# 📌 Código
from mmshap_medclip.tasks.isa import run_isa_one

muestra = 154
sample = dataset[muestra]
image, caption = sample['image'], sample['text']

res = run_isa_one(model, image, caption, device, explain=True, plot=True)
print(f"logit={res['logit']:.4f}  TScore={res['tscore']:.2%}  IScore={res['iscore']:.2%}")
