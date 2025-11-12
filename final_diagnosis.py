#!/usr/bin/env python3
"""Diagnóstico final del problema"""

import os
from pathlib import Path
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)

from mmshap_medclip.io_utils import load_config
from mmshap_medclip.devices import get_device
from mmshap_medclip.registry import build_dataset, build_model
from mmshap_medclip.tasks.utils import prepare_batch, make_image_token_ids, compute_text_token_lengths, concat_text_image_tokens

device = get_device()

print("=" * 80)
print("DIAGNÓSTICO FINAL")
print("=" * 80)

cfg = load_config("configs/roco_isa_biomedclip.yaml")
dataset = build_dataset(cfg["dataset"])
model = build_model(cfg["model"], device=device)

sample = dataset[450]
image, caption = sample['image'], sample['text']

inputs, logits = prepare_batch(model, [caption], [image], device=device)

nb_text_tokens_tensor, _ = compute_text_token_lengths(inputs, model.tokenizer)
print(f"\n📝 nb_text_tokens_tensor: {nb_text_tokens_tensor.item()}")

image_token_ids_expanded, imginfo = make_image_token_ids(inputs, model)
print(f"\n🖼️  imginfo:")
print(f"  num_patches: {imginfo['num_patches']}")

X_clean, text_len = concat_text_image_tokens(
    inputs, image_token_ids_expanded, device=device, 
    nb_text_tokens=nb_text_tokens_tensor, tokenizer=model.tokenizer
)

print(f"\n🔗 concat_text_image_tokens:")
print(f"  X_clean.shape: {X_clean.shape}")
print(f"  text_len (devuelto): {text_len}")
print(f"  Tokens imagen en X_clean: {X_clean.shape[1] - text_len}")

# Verificar si el tokenizer tiene context_length
inner_tok = getattr(model.tokenizer, "_tokenizer", model.tokenizer)
context_length = getattr(inner_tok, "context_length", None)
print(f"\n🔍 Tokenizer:")
print(f"  context_length: {context_length}")
print(f"  needs_fixed_length: {context_length is not None and context_length > nb_text_tokens_tensor.item() * 1.5}")

# Verificar inputs["input_ids"]
print(f"\n📌 inputs['input_ids']:")
print(f"  shape: {inputs['input_ids'].shape}")
print(f"  Primeros 20 IDs: {inputs['input_ids'][0, :20].tolist()}")
print(f"  IDs desde 11 hasta 20: {inputs['input_ids'][0, 11:20].tolist()}")

# Verificar X_clean
print(f"\n📌 X_clean:")
print(f"  shape: {X_clean.shape}")
print(f"  Primeros 20 valores: {X_clean[0, :20].tolist()}")
print(f"  Valores desde text_len-5 hasta text_len+5: {X_clean[0, text_len-5:text_len+5].tolist()}")

print("\n" + "=" * 80)
print("CONCLUSIÓN:")
print("=" * 80)

if text_len == 256:
    print("\n❌ PROBLEMA IDENTIFICADO:")
    print("  text_len=256 (longitud fija con padding)")
    print(f"  Pero attention_mask indica solo {nb_text_tokens_tensor.item()} tokens reales")
    print("  Esto causa que SHAP divida incorrectamente:")
    print(f"    - compute_mm_score usa {nb_text_tokens_tensor.item()} tokens de texto")
    print(f"    - Deja {X_clean.shape[1] - nb_text_tokens_tensor.item()} 'tokens de imagen'")
    print(f"    - Pero deberían ser solo {imginfo['num_patches']} tokens de imagen")
else:
    print(f"\n✓ text_len={text_len} (solo tokens reales)")
    print(f"  Tokens imagen: {X_clean.shape[1] - text_len}")
    print(f"  Esperado: {imginfo['num_patches']}")
    if X_clean.shape[1] - text_len == imginfo['num_patches']:
        print("  ✓ ¡Coincide!")
    else:
        print(f"  ❌ NO coincide (diferencia: {(X_clean.shape[1] - text_len) - imginfo['num_patches']})")

print("\n" + "=" * 80)

