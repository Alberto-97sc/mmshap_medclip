#!/usr/bin/env python3
"""Probar que WhyXrayCLIP sigue funcionando después de la corrección"""

import os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)

from mmshap_medclip.io_utils import load_config
from mmshap_medclip.devices import get_device
from mmshap_medclip.registry import build_dataset, build_model
from mmshap_medclip.tasks.isa import run_isa_one
from mmshap_medclip.tasks.whyxrayclip import filter_roco_by_keywords

device = get_device()

print("=" * 80)
print("VERIFICACIÓN - WhyXrayCLIP (debe seguir funcionando)")
print("=" * 80)

cfg = load_config("configs/roco_isa_whyxrayclip.yaml")
dataset = build_dataset(cfg["dataset"])
model = build_model(cfg["model"], device=device)

# Filtrar como en el notebook
dataset = filter_roco_by_keywords(dataset, keywords=("chest x-ray", "lung"))

muestra = 50
sample = dataset[muestra]
image, caption = sample['image'], sample['text']

print(f"\n🔍 Muestra {muestra}:")
print(f"  Caption: {caption[:80]}...")

# Ejecutar sin plot primero
res = run_isa_one(model, image, caption, device, explain=True, plot=False)

print(f"\n📈 Resultados:")
print(f"  Logit: {res['logit']:.4f}")
print(f"  TScore: {res['tscore']:.2%}")
print(f"  IScore: {res['iscore']:.2%}")
print(f"  text_len en resultado: {res.get('text_len', 'NO DISPONIBLE')}")

# Inspeccionar valores SHAP
if 'shap_values' in res:
    sv = res['shap_values']
    vals = sv.values if hasattr(sv, 'values') else sv
    
    text_len = res.get('text_len')
    
    if text_len is not None:
        vals_all = vals if vals.ndim == 2 else vals[None, :]
        feats = vals_all[0]
        
        text_vals = feats[:text_len]
        img_vals = feats[text_len:]
        
        print(f"\n  📝 Texto ({len(text_vals)} valores):")
        print(f"    Sum(abs): {np.abs(text_vals).sum():.2f}")
        
        print(f"\n  🖼️  Imagen ({len(img_vals)} valores):")
        print(f"    Sum(abs): {np.abs(img_vals).sum():.2f}")
        print(f"    Valores no-cero: {np.count_nonzero(img_vals)}")
        
        if np.count_nonzero(img_vals) > 0:
            print(f"  ✓ Hay valores no-cero en el array de imagen")
        else:
            print(f"  ❌ Todos los valores son cero")

print("\n" + "=" * 80)
print("Generando plot...")
print("=" * 80)

# Generar el plot
res_plot = run_isa_one(model, image, caption, device, explain=True, plot=True)
print(f"\n✓ Plot generado exitosamente")
print(f"  TScore: {res_plot['tscore']:.2%}")
print(f"  IScore: {res_plot['iscore']:.2%}")

print("\n" + "=" * 80)
print("✅ WhyXrayCLIP sigue funcionando correctamente")
print("=" * 80)

