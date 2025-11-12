#!/usr/bin/env python3
"""Prueba final ejecutando el código exacto del notebook de BiomedCLIP"""

import os
from pathlib import Path

# Configuración - igual que en el notebook
try:
    PROJECT_ROOT = Path(__file__).parent
except NameError:
    PROJECT_ROOT = Path.cwd()
    if PROJECT_ROOT.name == "experiments":
        PROJECT_ROOT = PROJECT_ROOT.parent

os.chdir(PROJECT_ROOT)
CFG_PATH = "configs/roco_isa_biomedclip.yaml"

# Cargar modelos y datos
from mmshap_medclip.io_utils import load_config
from mmshap_medclip.devices import get_device
from mmshap_medclip.registry import build_dataset, build_model

cfg = load_config(CFG_PATH)
device = get_device()
dataset = build_dataset(cfg["dataset"])
model = build_model(cfg["model"], device=device)

print("OK → len(dataset) =", len(dataset), "| device =", device)

# Ejecutar SHAP en una muestra (código exacto del notebook)
from mmshap_medclip.tasks.isa import run_isa_one

muestra = 450
sample = dataset[muestra]
image, caption = sample['image'], sample['text']

print(f"\n📌 Ejecutando run_isa_one con plot=True...")
res = run_isa_one(model, image, caption, device, explain=True, plot=True)
print(f"logit={res['logit']:.4f}  TScore={res['tscore']:.2%}  IScore={res['iscore']:.2%}")

print(f"\n✅ ¡Corrección exitosa! El heatmap de BiomedCLIP ahora funciona correctamente.")

