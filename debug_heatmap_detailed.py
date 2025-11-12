#!/usr/bin/env python3
"""Script de depuración detallado para el heatmap de BiomedCLIP"""

import os
from pathlib import Path
import numpy as np
import torch

# Configuración
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
CFG_PATH = "configs/roco_isa_biomedclip.yaml"

from mmshap_medclip.io_utils import load_config
from mmshap_medclip.devices import get_device
from mmshap_medclip.registry import build_dataset, build_model
from mmshap_medclip.tasks.isa import run_isa_one
from mmshap_medclip.tasks.utils import make_image_token_ids

# Cargar configuración
cfg = load_config(CFG_PATH)
device = get_device()
dataset = build_dataset(cfg["dataset"])
model = build_model(cfg["model"], device=device)

muestra = 450
sample = dataset[muestra]
image, caption = sample['image'], sample['text']

# Ejecutar sin plot
res = run_isa_one(model, image, caption, device, explain=True, plot=False)

print("=" * 80)
print("DEPURACIÓN DETALLADA DEL HEATMAP")
print("=" * 80)

# Extraer información
inputs = res['inputs']
sv = res['shap_values']
vals = sv.values if hasattr(sv, 'values') else sv

# Info de imagen
B, C, H, W = inputs["pixel_values"].shape
print(f"\n📐 Dimensiones de imagen:")
print(f"  Batch: {B}, Canales: {C}, Alto: {H}, Ancho: {W}")

# Info de parches usando make_image_token_ids
image_token_ids, imginfo = make_image_token_ids(inputs, model, strict=False)
print(f"\n🔍 Info de parches (make_image_token_ids):")
print(f"  patch_size: {imginfo.get('patch_size')}")
print(f"  grid_h: {imginfo.get('grid_h')}")
print(f"  grid_w: {imginfo.get('grid_w')}")
print(f"  num_patches: {imginfo.get('num_patches')}")

# Extraer valores SHAP
seq_len = int(inputs["attention_mask"][0].sum().item()) if "attention_mask" in inputs else int(inputs["input_ids"][0].shape[0])
print(f"\n📝 Longitud de secuencia de texto: {seq_len}")

vals_all = vals if vals.ndim == 2 else vals[None, :]
feats = vals_all[0]
img_slice = feats[seq_len:]

print(f"\n🖼️  Valores SHAP de imagen:")
print(f"  Cantidad de valores: {len(img_slice)}")
print(f"  Cantidad esperada: {imginfo.get('num_patches')}")
print(f"  ¿Coinciden?: {'SÍ' if len(img_slice) == imginfo.get('num_patches') else 'NO'}")

# Inspeccionar valores por sección
print(f"\n📊 Distribución de valores SHAP de imagen:")
n_patches = imginfo.get('num_patches', len(img_slice))
full_vec = np.asarray(img_slice).reshape(-1)

print(f"  Primeros 20: {full_vec[:20]}")
print(f"  Últimos 20: {full_vec[-20:]}")
print(f"  Centro (210-230): {full_vec[210:230]}")

# Verificar si hay valores no-cero
non_zero = np.count_nonzero(full_vec)
print(f"\n  Valores no-cero: {non_zero} de {len(full_vec)} ({non_zero/len(full_vec)*100:.1f}%)")

if non_zero > 0:
    non_zero_vals = full_vec[full_vec != 0]
    print(f"  Rango de valores no-cero:")
    print(f"    Min: {non_zero_vals.min():.2f}")
    print(f"    Max: {non_zero_vals.max():.2f}")
    print(f"    Media: {non_zero_vals.mean():.2f}")

# Simular el procesamiento que hace plot_text_image_heatmaps
grid_h = imginfo.get('grid_h')
grid_w = imginfo.get('grid_w')
n_expected = grid_h * grid_w

print(f"\n🔧 Simulando procesamiento de plot_text_image_heatmaps:")
print(f"  n_expected (grid_h * grid_w): {n_expected}")
print(f"  full_vec.size: {full_vec.size}")
print(f"  Relación: {full_vec.size / n_expected if n_expected > 0 else 'N/A'}")

# Simular la limpieza de parches (líneas 422-444 de heatmaps.py)
m = int(full_vec.size)
patch_vals = full_vec[:n_expected] if n_expected > 0 else np.zeros((0,), dtype=full_vec.dtype)

if m == n_expected:
    pv_clean = patch_vals
    print(f"  ✓ Caso 1: m == n_expected, usando patch_vals directamente")
elif n_expected > 0 and m % n_expected == 0:
    pv_clean = full_vec.reshape(-1, n_expected).mean(axis=0)
    print(f"  ✓ Caso 2: m múltiplo de n_expected, promediando {m // n_expected} repeticiones")
elif m > n_expected:
    pv_clean = patch_vals
    print(f"  ✓ Caso 3: m > n_expected, truncando a n_expected")
else:  # m < n_expected
    pad = n_expected - m
    if m == 0:
        pv_clean = np.zeros((n_expected,), dtype=full_vec.dtype)
        print(f"  ⚠️  Caso 4a: m == 0, creando array de ceros")
    else:
        pv_clean = np.pad(full_vec, (0, pad), mode="edge")
        print(f"  ⚠️  Caso 4b: m < n_expected, padding con 'edge'")

print(f"\n  pv_clean después de limpieza:")
print(f"    Tamaño: {pv_clean.size}")
print(f"    Min: {pv_clean.min():.2f}")
print(f"    Max: {pv_clean.max():.2f}")
print(f"    Media: {pv_clean.mean():.2f}")
print(f"    Valores no-cero: {np.count_nonzero(pv_clean)}")
print(f"    Primeros 20: {pv_clean[:20]}")

# Reshape a grid
if pv_clean.size == n_expected:
    patch_grid = np.reshape(pv_clean, (grid_h, grid_w), order="C")
    print(f"\n  ✓ patch_grid creado: {patch_grid.shape}")
    print(f"    Min: {patch_grid.min():.2f}")
    print(f"    Max: {patch_grid.max():.2f}")
    print(f"    Valores no-cero: {np.count_nonzero(patch_grid)}")
    
    # Mostrar algunas filas del grid
    print(f"\n  Primera fila del grid:")
    print(f"    {patch_grid[0, :10]}")
    print(f"\n  Fila central del grid:")
    center_row = grid_h // 2
    print(f"    {patch_grid[center_row, :10]}")
    print(f"\n  Última fila del grid:")
    print(f"    {patch_grid[-1, :10]}")
else:
    print(f"\n  ❌ ERROR: pv_clean.size ({pv_clean.size}) != n_expected ({n_expected})")

print("\n" + "=" * 80)

