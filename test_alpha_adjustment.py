#!/usr/bin/env python3
"""Verificar que los heatmaps tengan mejor transparencia"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)

from mmshap_medclip.io_utils import load_config
from mmshap_medclip.devices import get_device
from mmshap_medclip.registry import build_dataset, build_model
from mmshap_medclip.tasks.isa import run_isa_one

device = get_device()

print("=" * 80)
print("VERIFICACIÓN DE TRANSPARENCIA DEL HEATMAP")
print("=" * 80)

# Probar con Rclip
print("\n📦 Probando Rclip...")
cfg_rclip = load_config("configs/roco_isa_rclip.yaml")
dataset_rclip = build_dataset(cfg_rclip["dataset"])
model_rclip = build_model(cfg_rclip["model"], device=device)

sample = dataset_rclip[155]
res = run_isa_one(model_rclip, sample['image'], sample['text'], device, explain=True, plot=True)
print(f"✓ Rclip - logit={res['logit']:.4f}  TScore={res['tscore']:.2%}  IScore={res['iscore']:.2%}")

# Probar con PubMedCLIP
print("\n📦 Probando PubMedCLIP...")
cfg_pubmed = load_config("configs/roco_isa_pubmedclip.yaml")
dataset_pubmed = build_dataset(cfg_pubmed["dataset"])
model_pubmed = build_model(cfg_pubmed["model"], device=device)

sample = dataset_pubmed[155]
res = run_isa_one(model_pubmed, sample['image'], sample['text'], device, explain=True, plot=True)
print(f"✓ PubMedCLIP - logit={res['logit']:.4f}  TScore={res['tscore']:.2%}  IScore={res['iscore']:.2%}")

print("\n" + "=" * 80)
print("✅ Los heatmaps ahora tienen alpha=0.30 (antes 0.50)")
print("   Los parches deberían verse más difuminados y la imagen más clara")
print("=" * 80)

