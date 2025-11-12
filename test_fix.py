#!/usr/bin/env python3
"""Probar la corrección del heatmap de BiomedCLIP"""

import os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)

from mmshap_medclip.io_utils import load_config
from mmshap_medclip.devices import get_device
from mmshap_medclip.registry import build_dataset, build_model
from mmshap_medclip.tasks.isa import run_isa_one

device = get_device()

print("=" * 80)
print("PRUEBA DE CORRECCIÓN - BiomedCLIP")
print("=" * 80)

cfg = load_config("configs/roco_isa_biomedclip.yaml")
dataset = build_dataset(cfg["dataset"])
model = build_model(cfg["model"], device=device)

muestra = 450
sample = dataset[muestra]
image, caption = sample['image'], sample['text']

print(f"\n🔍 Muestra {muestra}:")
print(f"  Caption: {caption[:80]}...")

# Ejecutar sin plot primero para inspeccionar
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
    
    inputs = res['inputs']
    text_len = res.get('text_len')
    
    if text_len is not None:
        print(f"\n✓ text_len disponible: {text_len}")
        
        vals_all = vals if vals.ndim == 2 else vals[None, :]
        feats = vals_all[0]
        
        text_vals = feats[:text_len]
        img_vals = feats[text_len:]
        
        print(f"\n  📝 Texto ({len(text_vals)} valores):")
        print(f"    Sum(abs): {np.abs(text_vals).sum():.2f}")
        
        print(f"\n  🖼️  Imagen ({len(img_vals)} valores):")
        print(f"    Sum(abs): {np.abs(img_vals).sum():.2f}")
        print(f"    Valores no-cero: {np.count_nonzero(img_vals)}")
        print(f"    Primeros 10: {img_vals[:10]}")
        print(f"    Últimos 10: {img_vals[-10:]}")
        
        # Verificar que ahora los valores de imagen son correctos
        if len(img_vals) == 196:
            print(f"\n  ✓ ¡CORRECTO! Ahora hay exactamente 196 valores de imagen")
            if np.count_nonzero(img_vals) > 0:
                print(f"  ✓ ¡Y hay valores no-cero en el array de imagen!")
            else:
                print(f"  ❌ Pero todos los valores son cero...")
        else:
            print(f"\n  ❌ ERROR: Se esperaban 196 valores de imagen, pero hay {len(img_vals)}")
    else:
        print(f"\n❌ text_len no está disponible en el resultado")

print("\n" + "=" * 80)
print("Generando plot...")
print("=" * 80)

# Ahora generar el plot
res_plot = run_isa_one(model, image, caption, device, explain=True, plot=True)
print(f"\n✓ Plot generado exitosamente")
print(f"  TScore: {res_plot['tscore']:.2%}")
print(f"  IScore: {res_plot['iscore']:.2%}")

