#!/usr/bin/env python3
"""Script de depuración para diagnosticar problema con heatmap de BiomedCLIP"""

import os
from pathlib import Path
import numpy as np

# Configuración
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
CFG_PATH = "configs/roco_isa_biomedclip.yaml"

from mmshap_medclip.io_utils import load_config
from mmshap_medclip.devices import get_device
from mmshap_medclip.registry import build_dataset, build_model
from mmshap_medclip.tasks.isa import run_isa_one

# Cargar configuración
cfg = load_config(CFG_PATH)
device = get_device()
dataset = build_dataset(cfg["dataset"])
model = build_model(cfg["model"], device=device)

print("=" * 80)
print("DIAGNÓSTICO DE HEATMAP - BiomedCLIP")
print("=" * 80)
print(f"\n✓ Dataset cargado: {len(dataset)} muestras")
print(f"✓ Modelo: {model.__class__.__name__}")
print(f"✓ Device: {device}")

# Inspeccionar propiedades del modelo relacionadas con parches
print(f"\n📊 Propiedades del modelo relacionadas con parches:")
print(f"  - patch_size: {getattr(model, 'patch_size', 'NO ENCONTRADO')}")
print(f"  - vision_patch_size: {getattr(model, 'vision_patch_size', 'NO ENCONTRADO')}")
print(f"  - vision_input_size: {getattr(model, 'vision_input_size', 'NO ENCONTRADO')}")

# Probar en una muestra
muestra = 450
sample = dataset[muestra]
image, caption = sample['image'], sample['text']

print(f"\n🔍 Muestra {muestra}:")
print(f"  Caption: {caption[:80]}...")

# Ejecutar sin plot para inspeccionar valores
res = run_isa_one(model, image, caption, device, explain=True, plot=False)

print(f"\n📈 Resultados:")
print(f"  Logit: {res['logit']:.4f}")
print(f"  TScore: {res['tscore']:.2%}")
print(f"  IScore: {res['iscore']:.2%}")

# Inspeccionar valores SHAP
if 'shap_values' in res:
    sv = res['shap_values']
    vals = sv.values if hasattr(sv, 'values') else sv
    print(f"\n🔬 Valores SHAP:")
    print(f"  Forma: {vals.shape}")
    
    # Determinar longitud de texto
    inputs = res['inputs']
    seq_len = int(inputs["attention_mask"][0].sum().item()) if "attention_mask" in inputs else int(inputs["input_ids"][0].shape[0])
    
    text_vals = vals[0, :seq_len] if vals.ndim > 1 else vals[:seq_len]
    img_vals = vals[0, seq_len:] if vals.ndim > 1 else vals[seq_len:]
    
    print(f"\n  📝 Texto ({len(text_vals)} tokens):")
    print(f"    - Min: {text_vals.min():.6f}")
    print(f"    - Max: {text_vals.max():.6f}")
    print(f"    - Mean: {text_vals.mean():.6f}")
    print(f"    - Sum(abs): {np.abs(text_vals).sum():.6f}")
    
    print(f"\n  🖼️  Imagen ({len(img_vals)} parches):")
    print(f"    - Min: {img_vals.min():.6f}")
    print(f"    - Max: {img_vals.max():.6f}")
    print(f"    - Mean: {img_vals.mean():.6f}")
    print(f"    - Sum(abs): {np.abs(img_vals).sum():.6f}")
    print(f"    - ¿Todos ceros?: {np.all(img_vals == 0)}")
    print(f"    - ¿Todos muy pequeños (<1e-6)?: {np.all(np.abs(img_vals) < 1e-6)}")
    
    # Verificar si los valores de imagen están presentes pero muy pequeños
    if not np.all(img_vals == 0):
        print(f"\n  ✓ Los valores SHAP de imagen NO son todos cero")
        print(f"  Primeros 10 valores: {img_vals[:10]}")
    else:
        print(f"\n  ❌ PROBLEMA: Los valores SHAP de imagen son todos CERO")

# Ahora intentar generar el plot
print("\n" + "=" * 80)
print("Intentando generar plot...")
print("=" * 80)

res_with_plot = run_isa_one(model, image, caption, device, explain=True, plot=True)
print(f"\n✓ Plot generado (si no hay errores arriba)")
print(f"  Logit: {res_with_plot['logit']:.4f}")
print(f"  TScore: {res_with_plot['tscore']:.2%}")
print(f"  IScore: {res_with_plot['iscore']:.2%}")

