# src/mmshap_medclip/tasks/utils.py
from typing import List, Tuple, Optional, Dict
import torch
from mmshap_medclip.devices import move_to_device


def prepare_batch(
    model_wrapper,                 # p.ej., CLIPWrapper (expone .processor y .tokenizer)
    texts: List[str],
    images,                        # List[PIL.Image.Image] o PIL.Image
    device: torch.device = None,
    padding: bool = True,
    to_rgb: bool = True,
    debug_tokens: bool = False,
    amp_if_cuda: bool = True,
) -> Tuple[dict, torch.Tensor]:
    """
    Prepara el batch (texto+imagen), lo mueve al device y hace un forward sin gradientes.
    Devuelve: (inputs_dict, logits_per_image) donde logits_per_image es [B, 1].
    """
    # normaliza inputs a listas
    if not isinstance(texts, (list, tuple)):
        texts = [texts]
    if not isinstance(images, (list, tuple)):
        images = [images]

    # asegúrate de RGB si se solicita
    if to_rgb:
        images = [im.convert("RGB") for im in images]

    # tokenización con el processor del wrapper
    processor = model_wrapper.processor
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=padding)

    # mover al device del wrapper si no se pasó explícito
    if device is None:
        device = next(model_wrapper.parameters()).device
    inputs = move_to_device(inputs, device)

    # (opcional) inspección de tokens
    if debug_tokens:
        tok = model_wrapper.tokenizer
        print("\n🔍 Tokens que entran al modelo:")
        for i, ids in enumerate(inputs["input_ids"]):
            print(f"Texto {i+1}: {tok.convert_ids_to_tokens(ids.tolist())}")

    # forward sin gradientes (con AMP si hay CUDA)
    with torch.inference_mode():
        amp_ctx = torch.amp.autocast("cuda") if (amp_if_cuda and device.type == "cuda") else nullcontext()
        with amp_ctx:
            outputs = model_wrapper(**inputs)

    logits = outputs.logits_per_image  # [B, 1]
    return inputs, logits



def compute_text_token_lengths(
    inputs: dict,
    tokenizer=None,                 # opcional: para fallback sin attention_mask
) -> Tuple[torch.Tensor, List[int]]:
    """
    Devuelve (#tokens reales por texto) usando attention_mask si existe.
    """
    if "attention_mask" in inputs:
        nb_text_tokens_tensor = inputs["attention_mask"].sum(dim=1)           # (B,)
    else:
        if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
            nb_text_tokens_tensor = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
        else:
            # si no hay pad ni attention_mask, asume que todos los tokens son válidos
            B, T = inputs["input_ids"].shape
            nb_text_tokens_tensor = torch.full((B,), T, dtype=torch.long, device=inputs["input_ids"].device)
    return nb_text_tokens_tensor, nb_text_tokens_tensor.tolist()


def make_image_token_ids(
    inputs: dict,
    model_wrapper,
    patch_size: Optional[int] = None,  # permítele override manual
    strict: bool = True,
    debug: bool = False,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Genera IDs de parches 1..N por batch a partir de pixel_values y patch_size del modelo.
    Retorna: (image_token_ids_expanded [B, N], info={patch_size, grid_h, grid_w, num_patches})
    """
    device = inputs["input_ids"].device
    B, C, H, W = inputs["pixel_values"].shape

    # 1) resolver patch_size
    ps = patch_size
    if ps is None:
        m = getattr(model_wrapper, "model", model_wrapper)
        # intentos comunes en CLIP
        ps = getattr(getattr(getattr(m, "config", None), "vision_config", None), "patch_size", None)
        if ps is None:
            ps = getattr(getattr(getattr(m, "vision_model", None), "config", None), "patch_size", None)

    if ps is None:
        if strict:
            raise ValueError("No se pudo inferir patch_size desde el modelo. Pásalo por argumento.")
        ps = 32  # fallback razonable para ViT-B/32
        if debug:
            print("[make_image_token_ids] Usando patch_size=32 (fallback).")

    # 2) validaciones y grid
    if strict:
        assert H % ps == 0 and W % ps == 0, f"Imagen {H}×{W} no divisible por patch_size={ps}"
    grid_h, grid_w = H // ps, W // ps
    num_patches = grid_h * grid_w
    if debug:
        print(f"[make_image_token_ids] patch_size={ps} | grid={grid_h}×{grid_w} → N={num_patches}")

    # 3) IDs 1..N y expandir a batch
    ids = torch.arange(1, num_patches + 1, device=device, dtype=torch.long).unsqueeze(0)  # [1, N]
    image_token_ids_expanded = ids.expand(B, -1)                                           # [B, N]

    info = dict(patch_size=ps, grid_h=grid_h, grid_w=grid_w, num_patches=num_patches)
    return image_token_ids_expanded, info



def concat_text_image_tokens(
    inputs: dict,
    image_token_ids_expanded: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.long,
) -> Tuple[torch.Tensor, int]:
    """
    Concatena [texto | parches] para SHAP.
    - inputs["input_ids"]: [B, T_text]
    - image_token_ids_expanded: [B, N_patches]  (salida de make_image_token_ids)
    Retorna:
      X_clean: [B, T_text + N_patches] (dtype long, en 'device')
      text_seq_len: T_text
    """
    input_ids = inputs["input_ids"]
    if image_token_ids_expanded.dim() == 1:
        image_token_ids_expanded = image_token_ids_expanded.unsqueeze(0)

    B_txt = input_ids.shape[0]
    B_img = image_token_ids_expanded.shape[0]
    assert B_txt == B_img, f"Batch mismatch: texto={B_txt}, imagen={B_img}"

    text_seq_len = input_ids.shape[1]

    if device is None:
        device = input_ids.device

    X_clean = torch.cat((input_ids, image_token_ids_expanded), dim=1)
    X_clean = X_clean.to(dtype=dtype).to(device)
    return X_clean, text_seq_len

