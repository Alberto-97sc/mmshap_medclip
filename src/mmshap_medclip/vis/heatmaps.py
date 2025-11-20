# src/mmshap_medclip/vis/heatmaps.py
import math
import textwrap
from typing import List, Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import Normalize, TwoSlopeNorm
from PIL import Image

from mmshap_medclip.tasks.utils import make_image_token_ids

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32)

PLOT_ISA_IMG_PERCENTILE = 90   # escala robusta al percentil 90
PLOT_ISA_ALPHA_IMG = 0.30      # opacidad del overlay (reducida para mejor visibilidad)
PLOT_ISA_COARSEN_G = 2        # tamaño de super-parches (3x3)


def wrap_text(text: str, max_width: int = 80, max_lines: Optional[int] = None, 
              prefer_long_lines: bool = False) -> str:
    """
    Envuelve un texto largo en múltiples líneas de manera simétrica y equilibrada.
    NO trunca el texto con puntos suspensivos; en su lugar, ajusta el ancho o número de líneas.
    Intenta crear líneas de longitud similar para mejor legibilidad.
    
    Args:
        text: Texto a envolver
        max_width: Ancho máximo de caracteres por línea (por defecto 80)
        max_lines: Número máximo de líneas deseado (None = sin límite, se ajusta automáticamente)
        prefer_long_lines: Si True, prefiere líneas más largas en lugar de más líneas
    
    Returns:
        Texto envuelto en múltiples líneas simétricas, SIN truncamiento
    """
    if not text:
        return text
    
    # Dividir en palabras para mejor control
    words = text.split()
    if not words:
        return text
    
    # Si max_lines está especificado, intentar crear líneas simétricas
    if max_lines is not None and max_lines > 0:
        # Calcular longitud total (caracteres + espacios entre palabras)
        text_length = sum(len(word) for word in words) + len(words) - 1  # palabras + espacios
        avg_chars_per_line = text_length / max_lines
        
        # Objetivo: crear líneas de longitud similar
        target_line_length = max(int(avg_chars_per_line * 1.05), max_width // 2)
        
        # Si el texto es muy largo, aumentar el ancho objetivo
        if text_length > max_width * max_lines:
            # Ajustar target_line_length pero mantener límites razonables
            target_line_length = min(max(target_line_length, max_width), 
                                   100 if prefer_long_lines else 90)
        
        wrapped_lines = []
        current_line = []
        current_length = 0
        
        for idx, word in enumerate(words):
            word_len = len(word)
            space_needed = 1 if current_line else 0
            total_length_if_added = current_length + space_needed + word_len
            
            # Verificar si ya estamos en la última línea permitida
            is_last_allowed_line = len(wrapped_lines) == max_lines - 1
            
            # Si ya estamos en la última línea permitida, agregar todas las palabras restantes
            if is_last_allowed_line and current_line:
                # Agregar la palabra actual y todas las restantes
                current_line.append(word)
                if idx < len(words) - 1:
                    remaining_words = words[idx + 1:]
                    current_line.extend(remaining_words)
                current_length = len(" ".join(current_line))
                break
            
            # Si agregar esta palabra excedería el objetivo Y ya tenemos contenido
            # Y aún no hemos alcanzado el máximo de líneas
            if (current_line and total_length_if_added > target_line_length):
                # Guardar la línea actual y comenzar una nueva
                wrapped_lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_len
            else:
                # Agregar palabra a la línea actual
                if current_line:
                    current_length += 1  # espacio
                current_line.append(word)
                current_length += word_len
        
        # Agregar la última línea si quedó pendiente
        if current_line:
            wrapped_lines.append(" ".join(current_line))
        
        # Si después de este proceso aún tenemos más líneas de las deseadas,
        # re-envolver con un ancho mayor (pero sin truncar)
        if len(wrapped_lines) > max_lines:
            # Calcular nuevo ancho para que quepa en max_lines
            new_width = min(int(text_length / max_lines * 1.2), 100)
            wrapped_lines = textwrap.wrap(text, width=new_width, 
                                         break_long_words=True, 
                                         break_on_hyphens=False)
    else:
        # Sin límite de líneas, usar textwrap estándar con el ancho especificado
        wrapped_lines = textwrap.wrap(text, width=max_width, 
                                     break_long_words=True, 
                                     break_on_hyphens=False)
    
    # NUNCA truncar - siempre mostrar el texto completo
    return "\n".join(wrapped_lines)

def _infer_patch_size(model_wrapper, inputs, shap_values):
    ps = None
    m = model_wrapper
    base = getattr(model_wrapper, "model", None)

    candidates = [
        getattr(m, "vision_patch_size", None),
        getattr(base, "vision_patch_size", None) if base is not None else None,
        getattr(getattr(m, "vision_model", None), "patch_size", None),
        getattr(getattr(base, "vision_model", None), "patch_size", None) if base is not None else None,
        getattr(getattr(getattr(m, "vision_model", None), "config", None), "patch_size", None),
        getattr(getattr(getattr(base, "vision_model", None), "config", None), "patch_size", None) if base is not None else None,
        getattr(getattr(m, "visual", None), "patch_size", None),
        getattr(getattr(base, "visual", None), "patch_size", None) if base is not None else None,
        getattr(getattr(getattr(m, "vision_model", None), "patch_embed", None), "patch_size", None),
        getattr(getattr(getattr(base, "vision_model", None), "patch_embed", None), "patch_size", None) if base is not None else None,
    ]

    for cand in candidates:
        if cand is not None:
            ps = cand
            break

    if isinstance(ps, (tuple, list)):
        if len(ps) == 0:
            ps = None
        else:
            ps = int(ps[0])

    if ps is None:
        n_tokens = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
        sv = shap_values.values if hasattr(shap_values, "values") else shap_values
        fdim = sv.shape[-1]
        n_patches = max(fdim - n_tokens, 1)

        if "pixel_values" in inputs:
            H = int(inputs["pixel_values"].shape[-2])
        else:
            H = getattr(m, "vision_input_size", None)
            if H is None and base is not None:
                H = getattr(base, "vision_input_size", None)
            if H is None:
                H = getattr(base, "image_size", None) if base is not None else None
            if isinstance(H, (list, tuple)):
                H = int(H[0])
            if H is None:
                H = 224

        side = int(round(math.sqrt(max(n_patches, 1))))
        if side > 0:
            ps = max(1, int(round(H / side)))

    return ps


def _get_special_ids(tokenizer) -> set:
    special = set()
    if tokenizer is None:
        return special

    for name in ["all_special_ids", "special_ids", "special_token_ids"]:
        if hasattr(tokenizer, name):
            try:
                special.update(getattr(tokenizer, name))
            except Exception:
                pass

    for name in [
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "cls_token_id",
        "sep_token_id",
        "mask_token_id",
    ]:
        val = getattr(tokenizer, name, None)
        if val is not None:
            special.add(int(val))

    # IDs comunes de tokens especiales en diferentes tokenizadores
    # OpenCLIP: 49406 (SOT), 49407 (EOT), 0 (PAD)
    # BERT/PubMedBERT: 101 (CLS), 102 (SEP), 0 (PAD)
    special.update({49406, 49407, 0, 101, 102})
    return special


def _clean_token_string(raw_tok: str) -> str:
    tok_clean = raw_tok if raw_tok is not None else ""
    tok_clean = tok_clean.replace("Ċ", " ")
    for prefix in ("Ġ", "▁"):
        if tok_clean.startswith(prefix):
            tok_clean = tok_clean[len(prefix):]
    tok_clean = tok_clean.replace("</w>", "")
    if tok_clean.startswith("##"):
        tok_clean = tok_clean[2:]
    tok_clean = tok_clean.strip()
    return tok_clean


def _decode_single_token(tokenizer, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)

    decoded = ""
    if hasattr(tokenizer, "decode"):
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=True)
        except TypeError:
            decoded = tokenizer.decode([token_id])
        except Exception:
            decoded = ""
    if decoded:
        cleaned = _clean_token_string(decoded)
        if cleaned == str(token_id) or not cleaned:
            cleaned = ""
        if cleaned:
            return cleaned

    if hasattr(tokenizer, "convert_ids_to_tokens"):
        try:
            raw = tokenizer.convert_ids_to_tokens([token_id])
            if raw:
                cleaned = _clean_token_string(raw[0])
                if cleaned == str(token_id) or not cleaned:
                    cleaned = ""
                if cleaned:
                    return cleaned
        except Exception:
            pass

    if hasattr(tokenizer, "id_to_token"):
        try:
            raw = tokenizer.id_to_token(token_id)
            cleaned = _clean_token_string(raw)
            if cleaned:
                return cleaned
        except Exception:
            pass

    return str(token_id)


def _decode_tokens_for_plot(tokenizer, input_ids):
    """Return display tokens, decoded text and kept indices for SHAP visualization."""
    if hasattr(input_ids, "detach"):
        ids = input_ids.detach().cpu().tolist()
    else:
        ids = list(input_ids)
    ids = [int(x) for x in ids]

    if tokenizer is None:
        tokens = [str(x) for x in ids]
        text_clean = " ".join(tokens)
        keep = list(range(len(ids)))
        return tokens, text_clean, keep

    special_ids = _get_special_ids(tokenizer)
    keep_idx = [i for i, tid in enumerate(ids) if tid not in special_ids]
    kept_ids = [ids[i] for i in keep_idx]

    if hasattr(tokenizer, "decode"):
        try:
            text_clean = tokenizer.decode(kept_ids, skip_special_tokens=True)
        except TypeError:
            text_clean = tokenizer.decode(kept_ids)
    else:
        text_clean = ""

    # Decodificar tokens
    tokens_vis = []
    for tid in kept_ids:
        decoded_tok = _decode_single_token(tokenizer, tid)
        tokens_vis.append(decoded_tok if decoded_tok else str(tid))

    # Filtrar tokens especiales que puedan haberse colado por nombre
    # (algunos tokenizadores devuelven "[CLS]", "[SEP]", etc como strings)
    special_token_strings = {"[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]", 
                             "<s>", "</s>", "<pad>", "<unk>", "<mask>",
                             "<|startoftext|>", "<|endoftext|>"}
    
    # Filtrar tokens y sus índices
    filtered_tokens = []
    filtered_keep_idx = []
    for tok, idx in zip(tokens_vis, keep_idx):
        if tok.strip() not in special_token_strings:
            filtered_tokens.append(tok)
            filtered_keep_idx.append(idx)
    
    # Si todos fueron filtrados, usar el texto limpio
    if not filtered_tokens and text_clean:
        filtered_tokens = text_clean.split()
        # En este caso, keep_idx ya no es preciso, pero mantener consistencia
        filtered_keep_idx = list(range(len(filtered_tokens)))

    if not filtered_tokens and kept_ids:
        filtered_tokens = [str(t) for t in kept_ids]
        filtered_keep_idx = keep_idx

    return filtered_tokens, text_clean, filtered_keep_idx


def plot_text_image_heatmaps(
    shap_values,  # Union[np.ndarray, shap._explanation.Explanation]
    inputs: dict,
    tokenizer,
    images: Union[Image.Image, List[Image.Image]],
    texts: List[str],
    mm_scores: List[Tuple[float, dict]],     # [(tscore, OrderedDict(word->score)), ...]
    model_wrapper,
    cmap_name: str = "coolwarm",
    alpha_img: float = None,
    return_fig: bool = False,
    text_len: Optional[int] = None,
):
    """
    Dibuja, por muestra del batch, el heatmap de parches de imagen usando valores SHAP
    y un renglón de palabras coloreado por la contribución SHAP (con signo) a nivel palabra.
    - Cada palabra recibe un parche con su valor SHAP agregado (suma de subtokens).
    - Soporta shap_values con formas (B, L), (B,1,L) o (L,).
    - Usa text_len si se proporciona, o attention_mask por muestra para el # de tokens de texto.

    Params:
      images: PIL o lista de PILs (si pasas una sola, se replica para todo el batch)
      texts:  lista de strings (títulos)
      mm_scores: lista [(tscore, word_shap_dict_con_signo), ...] por muestra donde
                 word_shap_dict es OrderedDict[str, float] con palabra→valor SHAP
      text_len: longitud de la secuencia de texto (si None, usa attention_mask)
    """
    # --- normalizar inputs ---
    B = inputs["input_ids"].shape[0]
    if not isinstance(images, (list, tuple)):
        images_for_batch = [images] * B
    else:
        images_for_batch = list(images)
        if len(images_for_batch) == 1 and B > 1:
            images_for_batch = images_for_batch * B
        assert len(images_for_batch) == B, f"len(images)={len(images_for_batch)} != batch={B}"
    assert len(texts) == B, f"len(texts)={len(texts)} != batch={B}"
    assert len(mm_scores) == B, f"len(mm_scores)={len(mm_scores)} != batch={B}"

    # --- resolver patch/grid ---
    image_token_ids, imginfo = make_image_token_ids(
        inputs,
        model_wrapper,
        strict=False,
    )
    # liberar inmediatamente la matriz de IDs; solo necesitamos la metadata
    del image_token_ids

    grid_h = int(imginfo.get("grid_h", 0) or 0)
    grid_w = int(imginfo.get("grid_w", 0) or 0)

    patch_h = patch_w = None
    patch_size_info = imginfo.get("patch_size")
    if isinstance(patch_size_info, (list, tuple)):
        if len(patch_size_info) >= 2:
            patch_h, patch_w = int(patch_size_info[0]), int(patch_size_info[1])
        elif len(patch_size_info) == 1:
            patch_h = patch_w = int(patch_size_info[0])
    elif patch_size_info is not None:
        patch_h = patch_w = int(patch_size_info)

    _, _, H, W = inputs["pixel_values"].shape

    if patch_h is None and grid_h > 0:
        patch_h = max(1, int(round(H / max(grid_h, 1))))
    if patch_w is None and grid_w > 0:
        patch_w = max(1, int(round(W / max(grid_w, 1))))

    if patch_h is None or patch_w is None or patch_h <= 0 or patch_w <= 0 or grid_h <= 0 or grid_w <= 0:
        ps = _infer_patch_size(model_wrapper, inputs, shap_values)
        if ps is None:
            raise ValueError("No pude inferir patch_size; pásalo vía model_wrapper o inputs.")
        if isinstance(ps, (list, tuple)):
            if len(ps) == 0:
                raise ValueError(f"patch_size iterable inesperado: {ps}")
            if len(ps) == 1:
                patch_h = patch_w = int(ps[0])
            else:
                patch_h, patch_w = int(ps[0]), int(ps[1])
        else:
            patch_h = patch_w = int(ps)
        grid_h = max(1, int(round(H / max(patch_h, 1))))
        grid_w = max(1, int(round(W / max(patch_w, 1))))

    patch_h = max(1, int(patch_h))
    patch_w = max(1, int(patch_w))
    side_h_base = grid_h if grid_h > 0 else None
    side_w_base = grid_w if grid_w > 0 else None

    # --- normalizar SHAP a matriz (B, L) ---
    vals = shap_values.values if hasattr(shap_values, "values") else shap_values
    if vals.ndim == 1:
        vals_all = vals[None, :]                   # (1, L)
    elif vals.ndim == 2:
        vals_all = vals                             # (B, L)
    elif vals.ndim == 3:
        vals_all = vals[:, 0, :]                    # (B, L)
    else:
        raise ValueError(f"Forma inesperada de shap_values: {vals.shape}")

    # --- longitudes de texto por muestra ---
    # Si text_len está disponible (de _compute_isa_shap), usarlo para todas las muestras
    # Esto es crítico para modelos con tokenizadores de longitud fija (OpenCLIP con BERT)
    if text_len is not None:
        seq_lens = [text_len] * B
    else:
        # Fallback: usar attention_mask por muestra
        seq_lens = [int(inputs["attention_mask"][i].sum().item()) if "attention_mask" in inputs
                    else int(inputs["input_ids"][i].shape[0]) for i in range(B)]

    # Extraer valores de texto e imagen para normalización
    all_text_values = []
    all_image_values = []
    for i in range(B):
        seq_len = seq_lens[i]
        feats = vals_all[i]
        
        # Para normalización de colores, usar los valores de palabras de mm_scores
        _, word_shap_dict = mm_scores[i]
        if word_shap_dict:
            word_vals = np.array([word_shap_dict[w] for w in word_shap_dict.keys()])
            all_text_values.append(word_vals)
        else:
            all_text_values.append(np.zeros((0,), dtype=feats.dtype))

        image_vals = feats[seq_len:]
        all_image_values.append(image_vals)

    def _concat_or_zero(arrays):
        valid = [a for a in arrays if a.size > 0]
        if not valid:
            return np.zeros((1,), dtype=np.float32)
        return np.concatenate(valid)

    text_concat = _concat_or_zero(all_text_values)
    img_concat = _concat_or_zero(all_image_values)

    if np.any(text_concat < 0):
        absmax = float(np.percentile(np.abs(text_concat), 95))
        if absmax <= 0:
            absmax = float(np.max(np.abs(text_concat))) if text_concat.size else 0.0
        absmax = max(absmax, 1e-6)
        norm_text = TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
        cmap_text = plt.get_cmap("coolwarm")
    else:
        vmax_text = float(np.percentile(text_concat, 95)) if text_concat.size else 0.0
        vmax_text = max(vmax_text, 1e-6)
        norm_text = Normalize(vmin=0.0, vmax=vmax_text)
        cmap_text = plt.get_cmap("Reds")

    alpha_overlay = (
        float(alpha_img)
        if alpha_img is not None
        else float(PLOT_ISA_ALPHA_IMG)
    )
    alpha_overlay = min(max(alpha_overlay, 0.0), 1.0)
    coarsen_factor = int(PLOT_ISA_COARSEN_G) if PLOT_ISA_COARSEN_G else 0
    percentile_img = float(PLOT_ISA_IMG_PERCENTILE)
    image_overlay_entries = []
    coarsened_abs_values = []

    # --- figura ---
    # Aumentar altura de la figura y dar más espacio a la sección de texto
    fig = plt.figure(figsize=(5 * B, 7.5), layout="constrained")
    gs  = fig.add_gridspec(2, B, height_ratios=[3.5, 1.5], hspace=-0.55, wspace=0.03)

    # for measuring token widths to center text row
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _format_tokens(tokens: List[str]) -> List[str]:
        formatted = []
        for tok in tokens:
            tok_clean = tok if tok is not None else ""
            tok_clean = tok_clean.strip()
            if not tok_clean:
                tok_clean = "∅"
            if len(tok_clean) > 19:
                tok_clean = tok_clean[:18] + "…"
            formatted.append(tok_clean)
        return formatted

    for i, (tscore, _) in enumerate(mm_scores):
        feats = vals_all[i]
        tlen = seq_lens[i]
        text_slice = feats[:tlen]
        img_slice = feats[tlen:]
        ta = np.abs(text_slice)
        ia = np.abs(img_slice)
        tot = ta.sum() + ia.sum()
        iscore  = float(ia.sum() / tot) if tot > 0 else 0.0

        # --- imagen ---
        px = inputs["pixel_values"][i].detach().cpu()
        H = int(px.shape[-2])
        W = int(px.shape[-1])

        patch_h_eff = max(1, int(round(patch_h)))
        patch_w_eff = max(1, int(round(patch_w)))
        side_h = side_h_base if side_h_base else max(1, int(round(H / float(patch_h_eff))))
        side_w = side_w_base if side_w_base else max(1, int(round(W / float(patch_w_eff))))
        n_expected = max(1, side_h * side_w)

        full_vec = np.asarray(img_slice).reshape(-1)

        if full_vec.size == 0:
            pv_clean = np.zeros((n_expected,), dtype=feats.dtype)
        else:
            pv = full_vec.reshape(-1)
            m = int(pv.size)
            patch_vals = pv[:n_expected] if n_expected > 0 else np.zeros((0,), dtype=pv.dtype)

            if m == n_expected:
                pv_clean = patch_vals
            elif n_expected > 0 and m % n_expected == 0:
                pv_clean = pv.reshape(-1, n_expected).mean(axis=0)
            elif m > n_expected:
                pv_clean = patch_vals
            else:  # m < n_expected
                pad = n_expected - m
                if m == 0:
                    pv_clean = np.zeros((n_expected,), dtype=feats.dtype)
                else:
                    pv_clean = np.pad(pv, (0, pad), mode="edge")

        if pv_clean.size != n_expected:
            if pv_clean.size > n_expected:
                pv_clean = pv_clean[:n_expected]
            else:
                pv_clean = np.pad(pv_clean, (0, n_expected - pv_clean.size), mode="constant")

        if pv_clean.dtype != feats.dtype:
            pv_clean = pv_clean.astype(feats.dtype, copy=False)

        assert pv_clean.size == n_expected, (
            f"Tras limpieza, m={pv_clean.size} != side_h*side_w={n_expected}"
        )

        patch_grid = np.reshape(pv_clean, (side_h, side_w), order="C")

        grid_vis = patch_grid
        if coarsen_factor and coarsen_factor > 1:
            sh, sw = grid_vis.shape
            sh2 = (sh // coarsen_factor) * coarsen_factor
            sw2 = (sw // coarsen_factor) * coarsen_factor
            if sh2 >= coarsen_factor and sw2 >= coarsen_factor and sh2 > 0 and sw2 > 0:
                grid_vis = grid_vis[:sh2, :sw2].reshape(
                    sh2 // coarsen_factor,
                    coarsen_factor,
                    sw2 // coarsen_factor,
                    coarsen_factor,
                ).mean(axis=(1, 3))

        grid_abs = np.abs(grid_vis).reshape(-1)
        if grid_abs.size == 0:
            grid_abs = np.zeros((1,), dtype=np.float32)
        else:
            grid_abs = grid_abs.astype(np.float32, copy=False)
        coarsened_abs_values.append(grid_abs)

        mean = _CLIP_MEAN.to(dtype=px.dtype, device=px.device).view(3, 1, 1)
        std = _CLIP_STD.to(dtype=px.dtype, device=px.device).view(3, 1, 1)
        img_vis = torch.clamp(px * std + mean, 0, 1).permute(1, 2, 0).numpy()

        heat_tensor = torch.as_tensor(grid_vis, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        heat_up = F.interpolate(heat_tensor, size=(H, W), mode="nearest").squeeze().numpy()

        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(img_vis, origin="upper", interpolation="nearest", zorder=0)
        
        # Mostrar solo TScore e IScore juntos en la parte superior
        # Usar text() con posición absoluta para subir el título y evitar que choque con la imagen
        title_text = f"TScore: {tscore:.2%}  |  IScore: {iscore:.2%}"
        ax_img.text(0.5, 1.08, title_text, fontsize=12, weight='bold',
                   ha='center', va='bottom', transform=ax_img.transAxes,
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="white", 
                            alpha=0.9, edgecolor="gray", linewidth=1.2))
        image_overlay_entries.append({
            "ax": ax_img,
            "heat": heat_up,
            "H": H,
            "W": W,
        })

        # --- texto ---
        # Usar las palabras y sus valores SHAP directamente de mm_scores
        # para que haya un parche por palabra, no por subtoken
        _, word_shap_dict = mm_scores[i]
        words = list(word_shap_dict.keys())
        word_vals = np.array([word_shap_dict[w] for w in words])

        ax_txt = fig.add_subplot(gs[1, i])
        ax_txt.axis("off")
        ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)

        if len(words) == 0 or len(word_vals) == 0:
            # Fallback: decodificar tokens para mostrar el texto limpio
            token_ids = inputs["input_ids"][i][:seq_lens[i]]
            _, text_clean, _ = _decode_tokens_for_plot(tokenizer, token_ids)
            # Envolver el texto limpio largo en múltiples líneas SIN truncar
            text_to_wrap = text_clean if text_clean else "(sin tokens)"
            text_len = len(text_to_wrap)
            if text_len > 150:
                wrapped_text = wrap_text(text_to_wrap, max_width=75, max_lines=5, prefer_long_lines=False)
            elif text_len > 100:
                wrapped_text = wrap_text(text_to_wrap, max_width=70, max_lines=4, prefer_long_lines=False)
            else:
                wrapped_text = wrap_text(text_to_wrap, max_width=65, max_lines=3, prefer_long_lines=False)
            # Ya no mostrar TScore aquí, está arriba junto con IScore
            ax_txt.text(0.5, 0.5, wrapped_text,
                        ha="center", va="center", transform=ax_txt.transAxes, fontsize=12)
            continue

        # Formatear palabras para mostrar
        words_display = _format_tokens(words)

        # Medir ancho de palabras para centrar, incluyendo espacios
        widths = []
        bbox_kw = dict(facecolor="white", pad=0.2, alpha=0)
        for word in words_display:
            t = ax_txt.text(0, 0, word, ha="left", va="center", fontsize=14, bbox=bbox_kw)
            bb = t.get_window_extent(renderer=renderer)
            bb_data = ax_txt.transAxes.inverted().transform([[bb.x1, bb.y1], [bb.x0, bb.y0]])
            widths.append(bb_data[0,0] - bb_data[1,0])
            t.remove()

        # Agregar espacio entre palabras - usar espaciado normal del caption original
        gap = 0.015  # Espacio entre palabras (reducido para espaciado normal)
        
        # Usar ancho real de las palabras para dividir en líneas simétricas
        # Calcular ancho total de todas las palabras
        total_width = sum(widths) + gap * max(0, len(words_display) - 1)
        
        # Determinar número de líneas según la longitud del texto (reducido para más palabras por línea)
        text_plain = " ".join(words_display)
        text_length = len(text_plain)
        
        if text_length > 200:
            target_num_lines = 4
        elif text_length > 120:
            target_num_lines = 3
        else:
            target_num_lines = 2
        
        # Calcular ancho objetivo por línea (usar 95% del ancho disponible para más palabras por línea)
        max_width_per_line = min(0.95, total_width / max(target_num_lines, 1) * 1.2)
        
        # Dividir palabras en líneas usando el ancho real
        lines = []
        current_line_words = []
        current_line_vals = []
        current_line_widths = []
        current_line_width = 0
        
        for word, val, w in zip(words_display, word_vals, widths):
            word_width_with_gap = w + (gap if current_line_words else 0)
            
            # Si agregar esta palabra excedería el ancho máximo Y ya tenemos contenido
            # Ser más permisivo: solo crear nueva línea si excede significativamente (1.05x) o si ya tenemos muchas líneas
            if current_line_words and (current_line_width + word_width_with_gap) > max_width_per_line * 1.05:
                # Si aún no hemos alcanzado el número objetivo de líneas, crear nueva línea
                if len(lines) < target_num_lines - 1:
                    lines.append((current_line_words, current_line_vals, current_line_widths))
                    current_line_words = [word]
                    current_line_vals = [val]
                    current_line_widths = [w]
                    current_line_width = w
                else:
                    # Si ya estamos en la última línea permitida, permitir que sea más larga
                    # Solo crear nueva línea si excede mucho (1.15x)
                    if current_line_width > max_width_per_line * 1.15:
                        lines.append((current_line_words, current_line_vals, current_line_widths))
                        current_line_words = [word]
                        current_line_vals = [val]
                        current_line_widths = [w]
                        current_line_width = w
                    else:
                        # Agregar a la línea actual (permitir líneas más largas)
                        current_line_words.append(word)
                        current_line_vals.append(val)
                        current_line_widths.append(w)
                        current_line_width += word_width_with_gap
            else:
                # Agregar palabra a la línea actual
                current_line_words.append(word)
                current_line_vals.append(val)
                current_line_widths.append(w)
                current_line_width += word_width_with_gap
        
        # Agregar la última línea
        if current_line_words:
            lines.append((current_line_words, current_line_vals, current_line_widths))
        
        # No dividir líneas largas - permitir que las líneas sean más largas con más palabras
        
        # Calcular el espacio vertical necesario y ajustar posición del TScore
        # Aumentar significativamente el espaciado entre líneas para evitar superposición
        # Usar más espacio cuando hay más líneas para mejor legibilidad
        if len(lines) > 5:
            line_height = 0.12  # Espaciado muy grande para muchas líneas
        elif len(lines) > 3:
            line_height = 0.11  # Espaciado grande para varias líneas
        elif len(lines) > 1:
            line_height = 0.10  # Espaciado medio para múltiples líneas
        else:
            line_height = 0.09  # Espaciado normal para una línea
        
        total_height = len(lines) * line_height
        # Calcular posición inicial de las palabras cerca de la parte superior
        # para reducir el espacio entre la imagen y el caption
        start_y = 0.95 + total_height / 2 - line_height / 2
        
        # Dibujar cada línea de palabras con mejor espaciado
        for line_idx, (line_words, line_vals, line_widths) in enumerate(lines):
            line_total_w = sum(line_widths) + gap * max(0, len(line_words)-1)
            start_x = 0.5 - line_total_w / 2
            x = start_x
            # Calcular posición Y con espaciado uniforme entre líneas
            y = start_y - line_idx * line_height
            
            for word, val, w in zip(line_words, line_vals, line_widths):
                color = cmap_text(norm_text(val))
                ax_txt.text(
                    x, y, word,
                    ha="left", va="center", fontsize=13, color="black",
                    transform=ax_txt.transAxes,
                    # Padding reducido para espaciado normal entre palabras
                    bbox=dict(facecolor=color, alpha=0.85, edgecolor="white", 
                             linewidth=0.5, boxstyle="square,pad=0.15")  # Padding reducido para espaciado normal
                )
                x += w + gap

    if coarsened_abs_values:
        combined_abs = np.concatenate(coarsened_abs_values)
    else:
        combined_abs = np.zeros((1,), dtype=np.float32)

    vmax_img = float(np.percentile(combined_abs, percentile_img)) if combined_abs.size else 0.0
    if not np.isfinite(vmax_img) or vmax_img <= 0:
        vmax_img = float(np.max(combined_abs)) if combined_abs.size else 0.0
    
    # Si los valores son extremadamente pequeños, usar el máximo absoluto
    # en lugar del percentil para hacer la visualización más visible
    if vmax_img < 1e-3:
        vmax_img = float(np.max(combined_abs)) if combined_abs.size else 0.0
    
    # Asegurar un mínimo razonable para la normalización
    vmax_img = max(vmax_img, 1e-8)
    
    # Calcular los valores originales (con signo) para la normalización
    if all_image_values:
        img_vals_concat = np.concatenate([v for v in all_image_values if v.size > 0])
        if img_vals_concat.size > 0 and np.any(img_vals_concat != 0):
            # Usar el rango real de los valores
            vmax_img = max(abs(float(np.percentile(img_vals_concat, 95))), 
                          abs(float(np.percentile(img_vals_concat, 5))))
            vmax_img = max(vmax_img, 1e-8)

    norm_img = TwoSlopeNorm(vmin=-vmax_img, vcenter=0.0, vmax=vmax_img)
    cmap_img = plt.get_cmap(cmap_name or "coolwarm")

    for idx, entry in enumerate(image_overlay_entries):
        ax = entry["ax"]
        heat_up = entry["heat"]
        H = entry["H"]
        W = entry["W"]
        
        # Aumentar alpha ligeramente si los valores son muy pequeños para mejorar visibilidad
        alpha_to_use = min(alpha_overlay * 1.3, 0.6) if vmax_img < 1.0 else alpha_overlay
        
        ax.imshow(
            heat_up,
            cmap=cmap_img,
            norm=norm_img,
            alpha=alpha_to_use,
            origin="upper",
            interpolation="nearest",
            zorder=1,
        )
        
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.margins(0)
        ax.axis("off")

    # colorbars
    # Encontrar los ejes de imagen (fila 0) para alineación correcta del colorbar
    # y evitar que se cruce con el caption
    img_axes = []
    for ax in fig.axes:
        if hasattr(ax, 'get_subplotspec') and ax.get_subplotspec() is not None:
            subplot_spec = ax.get_subplotspec()
            if hasattr(subplot_spec, 'rowspan') and subplot_spec.rowspan.start == 0:  # Fila superior (imagen)
                img_axes.append(ax)
    
    if img_axes:
        # Usar la posición real del primer eje de imagen para alinear el colorbar
        # Esto asegura que el colorbar esté a la misma altura que la imagen médica
        first_img_pos = img_axes[0].get_position()
        # Usar y0 y height del eje de imagen directamente para alineación perfecta
        cax_i = fig.add_axes([first_img_pos.x1 + 0.03, first_img_pos.y0, 0.015, first_img_pos.height])
    elif image_overlay_entries:
        # Fallback: usar image_overlay_entries con la posición real del eje
        first_img_pos = image_overlay_entries[0]["ax"].get_position()
        cax_i = fig.add_axes([first_img_pos.x1 + 0.03, first_img_pos.y0, 0.015, first_img_pos.height])
    else:
        # Fallback si no encontramos los ejes de imagen
        first_pos = fig.axes[0].get_position()
        cax_i = fig.add_axes([first_pos.x1 + 0.03, first_pos.y0, 0.015, first_pos.height])
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_img, norm=norm_img), cax=cax_i, label="Valor SHAP por parche")

    # Colorbar del texto cerca del caption
    # Obtener las posiciones de los subplots de texto para calcular el ancho total
    text_axes = []
    for ax in fig.axes:
        if hasattr(ax, 'get_subplotspec') and ax.get_subplotspec() is not None:
            subplot_spec = ax.get_subplotspec()
            if hasattr(subplot_spec, 'rowspan') and subplot_spec.rowspan.start == 1:  # Fila inferior (texto)
                text_axes.append(ax)
    
    if text_axes:
        # Usar el primer y último subplot para calcular el ancho total
        first_text_pos = text_axes[0].get_position()
        last_text_pos = text_axes[-1].get_position()
        # Ancho total = posición final del último - posición inicial del primero
        text_width_total = last_text_pos.x1 - first_text_pos.x0
        # Colocar el colorbar muy cerca del caption (reducir espacio entre caption y colorbar)
        cax_t = fig.add_axes([first_text_pos.x0, first_text_pos.y0 - 0.005, text_width_total, 0.015])
    else:
        # Fallback si no encontramos los subplots de texto
        cax_t = fig.add_axes([0.05, 0.01, 0.9, 0.015])
    
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_text, norm=norm_text),
                 cax=cax_t, orientation="horizontal", label="Valor SHAP por palabra")

    if return_fig:
        return fig
    plt.show()
