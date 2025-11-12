#!/usr/bin/env python3
"""Comparar BiomedCLIP y WhyXrayCLIP para encontrar la diferencia"""

import os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)

from mmshap_medclip.io_utils import load_config
from mmshap_medclip.devices import get_device
from mmshap_medclip.registry import build_dataset, build_model
from mmshap_medclip.tasks.isa import run_isa_one
from mmshap_medclip.tasks.utils import make_image_token_ids

device = get_device()

print("=" * 80)
print("COMPARACIÓN BiomedCLIP vs WhyXrayCLIP")
print("=" * 80)

# === BiomedCLIP ===
print("\n📦 BiomedCLIP:")
cfg_bio = load_config("configs/roco_isa_biomedclip.yaml")
dataset_bio = build_dataset(cfg_bio["dataset"])
model_bio = build_model(cfg_bio["model"], device=device)

sample = dataset_bio[450]
res_bio = run_isa_one(model_bio, sample['image'], sample['text'], device, explain=True, plot=False)

inputs_bio = res_bio['inputs']
image_token_ids_bio, imginfo_bio = make_image_token_ids(inputs_bio, model_bio, strict=False)

print(f"  Dimensiones imagen: {inputs_bio['pixel_values'].shape}")
print(f"  patch_size: {imginfo_bio.get('patch_size')}")
print(f"  grid: {imginfo_bio.get('grid_h')}x{imginfo_bio.get('grid_w')}")
print(f"  num_patches esperado: {imginfo_bio.get('num_patches')}")

# Contar tokens de imagen en SHAP
sv_bio = res_bio['shap_values']
vals_bio = sv_bio.values if hasattr(sv_bio, 'values') else sv_bio
seq_len_bio = int(inputs_bio["attention_mask"][0].sum().item())
img_vals_bio = vals_bio[0, seq_len_bio:] if vals_bio.ndim > 1 else vals_bio[seq_len_bio:]
print(f"  num_patches SHAP real: {len(img_vals_bio)}")
print(f"  ¿Es 21x21?: {len(img_vals_bio) == 21*21} ({21*21})")
print(f"  ¿Es 14x14?: {len(img_vals_bio) == 14*14} ({14*14})")

# === WhyXrayCLIP ===
print("\n📦 WhyXrayCLIP:")
cfg_why = load_config("configs/roco_isa_whyxrayclip.yaml")
dataset_why = build_dataset(cfg_why["dataset"])
model_why = build_model(cfg_why["model"], device=device)

# Filtrar dataset como en el notebook
from mmshap_medclip.tasks.whyxrayclip import filter_roco_by_keywords
dataset_why = filter_roco_by_keywords(dataset_why, keywords=("chest x-ray", "lung"))

sample_why = dataset_why[50]  # Una muestra cualquiera
res_why = run_isa_one(model_why, sample_why['image'], sample_why['text'], device, explain=True, plot=False)

inputs_why = res_why['inputs']
image_token_ids_why, imginfo_why = make_image_token_ids(inputs_why, model_why, strict=False)

print(f"  Dimensiones imagen: {inputs_why['pixel_values'].shape}")
print(f"  patch_size: {imginfo_why.get('patch_size')}")
print(f"  grid: {imginfo_why.get('grid_h')}x{imginfo_why.get('grid_w')}")
print(f"  num_patches esperado: {imginfo_why.get('num_patches')}")

# Contar tokens de imagen en SHAP
sv_why = res_why['shap_values']
vals_why = sv_why.values if hasattr(sv_why, 'values') else sv_why
seq_len_why = int(inputs_why["attention_mask"][0].sum().item())
img_vals_why = vals_why[0, seq_len_why:] if vals_why.ndim > 1 else vals_why[seq_len_why:]
print(f"  num_patches SHAP real: {len(img_vals_why)}")

print("\n" + "=" * 80)
print("ANÁLISIS:")
print("=" * 80)

if len(img_vals_bio) != imginfo_bio.get('num_patches'):
    print(f"\n❌ BiomedCLIP tiene DESAJUSTE:")
    print(f"   Esperado: {imginfo_bio.get('num_patches')} (grid {imginfo_bio.get('grid_h')}x{imginfo_bio.get('grid_w')})")
    print(f"   Real: {len(img_vals_bio)}")
    print(f"   Diferencia: {len(img_vals_bio) - imginfo_bio.get('num_patches')}")
    
    # Calcular qué grid daría 441
    import math
    side = int(math.sqrt(len(img_vals_bio)))
    if side * side == len(img_vals_bio):
        print(f"   Los {len(img_vals_bio)} valores corresponden a un grid {side}x{side}")
        print(f"   Esto sugiere patch_size de {inputs_bio['pixel_values'].shape[-1] // side}x{inputs_bio['pixel_values'].shape[-1] // side}")
else:
    print(f"\n✓ BiomedCLIP está OK")

if len(img_vals_why) != imginfo_why.get('num_patches'):
    print(f"\n❌ WhyXrayCLIP tiene DESAJUSTE:")
    print(f"   Esperado: {imginfo_why.get('num_patches')}")
    print(f"   Real: {len(img_vals_why)}")
else:
    print(f"\n✓ WhyXrayCLIP está OK")

print("\n" + "=" * 80)

