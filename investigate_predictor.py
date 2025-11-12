#!/usr/bin/env python3
"""Investigar cómo el Predictor está calculando los parches"""

import os
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)

from mmshap_medclip.io_utils import load_config
from mmshap_medclip.devices import get_device
from mmshap_medclip.registry import build_dataset, build_model
from mmshap_medclip.tasks.utils import prepare_batch, make_image_token_ids, compute_text_token_lengths, concat_text_image_tokens
from mmshap_medclip.shap_tools.predictor import Predictor

device = get_device()

print("=" * 80)
print("INVESTIGAR PREDICTOR - BiomedCLIP")
print("=" * 80)

cfg = load_config("configs/roco_isa_biomedclip.yaml")
dataset = build_dataset(cfg["dataset"])
model = build_model(cfg["model"], device=device)

sample = dataset[450]
image, caption = sample['image'], sample['text']

# Preparar inputs como hace run_isa_one
inputs, logits = prepare_batch(model, [caption], [image], device=device, debug_tokens=False, amp_if_cuda=True)

print(f"\n📐 Inputs:")
print(f"  input_ids shape: {inputs['input_ids'].shape}")
print(f"  pixel_values shape: {inputs['pixel_values'].shape}")
print(f"  attention_mask shape: {inputs['attention_mask'].shape if 'attention_mask' in inputs else 'N/A'}")

# Calcular info como hace _compute_isa_shap
nb_text_tokens_tensor, _ = compute_text_token_lengths(inputs, model.tokenizer)
print(f"\n📝 Tokens de texto:")
print(f"  nb_text_tokens_tensor: {nb_text_tokens_tensor}")

# make_image_token_ids
image_token_ids_expanded, imginfo = make_image_token_ids(inputs, model)
print(f"\n🖼️  make_image_token_ids:")
print(f"  patch_size: {imginfo['patch_size']}")
print(f"  grid: {imginfo['grid_h']}x{imginfo['grid_w']}")
print(f"  num_patches: {imginfo['num_patches']}")

# concat_text_image_tokens
X_clean, text_len = concat_text_image_tokens(
    inputs, image_token_ids_expanded, device=device, 
    nb_text_tokens=nb_text_tokens_tensor, tokenizer=model.tokenizer
)
print(f"\n🔗 concat_text_image_tokens:")
print(f"  X_clean shape: {X_clean.shape}")
print(f"  text_len: {text_len}")
print(f"  Tokens imagen esperados: {X_clean.shape[1] - text_len}")

# Crear Predictor
predictor = Predictor(
    model,
    inputs,
    patch_size=imginfo["patch_size"],
    device=device,
    use_amp=True,
    text_len=text_len,
)

print(f"\n🔮 Predictor:")
print(f"  patch_size: {predictor.patch_size}")
print(f"  patch_h: {predictor.patch_h}, patch_w: {predictor.patch_w}")
print(f"  grid_h: {predictor.grid_h}, grid_w: {predictor.grid_w}")
print(f"  num_patches: {predictor.num_patches}")
print(f"  text_len: {predictor.text_len}")

# Ver dimensiones de pixel_values en predictor
pv_shape = predictor.base_inputs["pixel_values"].shape
print(f"  pixel_values shape: {pv_shape}")

# Verificar modelo
print(f"\n🔍 Inspección del modelo:")
print(f"  Tipo de wrapper: {type(model).__name__}")
print(f"  patch_size del wrapper: {model.patch_size}")
print(f"  vision_patch_size del wrapper: {model.vision_patch_size}")

# Inspeccionar visual model
visual = getattr(model.model, "visual", None)
if visual is not None:
    print(f"\n  Visual model encontrado:")
    print(f"    type: {type(visual).__name__}")
    print(f"    patch_size attr: {getattr(visual, 'patch_size', 'N/A')}")
    
    conv1 = getattr(visual, "conv1", None)
    if conv1 is not None:
        print(f"    conv1.kernel_size: {getattr(conv1, 'kernel_size', 'N/A')}")
        print(f"    conv1.stride: {getattr(conv1, 'stride', 'N/A')}")
    
    # Ver config
    config = getattr(visual, "config", None)
    if config is not None:
        print(f"    config.patch_size: {getattr(config, 'patch_size', 'N/A')}")
    
    # Intentar encontrar patch_embed
    patch_embed = getattr(visual, "patch_embed", None)
    if patch_embed is not None:
        print(f"    patch_embed type: {type(patch_embed).__name__}")
        print(f"    patch_embed.patch_size: {getattr(patch_embed, 'patch_size', 'N/A')}")

print("\n" + "=" * 80)

