# src/mmshap_medclip/vis/heatmaps.py
import math
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import Normalize, TwoSlopeNorm
from PIL import Image

from mmshap_medclip.tasks.utils import make_image_token_ids

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32)

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

    special.update({49406, 49407, 0})
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

    tokens_vis = []
    for tid in kept_ids:
        decoded_tok = _decode_single_token(tokenizer, tid)
        tokens_vis.append(decoded_tok if decoded_tok else str(tid))

    if not any(tok.strip() for tok in tokens_vis) and text_clean:
        tokens_vis = text_clean.split()

    if not tokens_vis and kept_ids:
        tokens_vis = [str(t) for t in kept_ids]

    return tokens_vis, text_clean, keep_idx


def plot_text_image_heatmaps(
    shap_values: Union[np.ndarray, "shap._explanation.Explanation"],
    inputs: dict,
    tokenizer,
    images: Union[Image.Image, List[Image.Image]],
    texts: List[str],
    mm_scores: List[Tuple[float, dict]],     # [(tscore, OrderedDict(word->score)), ...]
    model_wrapper,
    cmap_name: str = "viridis",
    alpha_img: float = 0.60,
    return_fig: bool = False,
):
    """
    Dibuja, por muestra del batch, el heatmap de parches de imagen usando valores SHAP
    y un renglón de tokens coloreado por la contribución SHAP (con signo) a nivel token.
    - Soporta shap_values con formas (B, L), (B,1,L) o (L,).
    - Usa attention_mask por muestra para el # de tokens de texto.

    Params:
      images: PIL o lista de PILs (si pasas una sola, se replica para todo el batch)
      texts:  lista de strings (títulos)
      mm_scores: lista [(tscore, word_shap_dict_con_signo), ...] por muestra
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
    seq_lens = [int(inputs["attention_mask"][i].sum().item()) if "attention_mask" in inputs
                else int(inputs["input_ids"][i].shape[0]) for i in range(B)]

    decoded_info = []
    all_text_values = []
    all_image_values = []
    for i in range(B):
        seq_len = seq_lens[i]
        token_ids = inputs["input_ids"][i][:seq_len]
        tokens_vis, text_clean, keep_idx = _decode_tokens_for_plot(tokenizer, token_ids)
        decoded_info.append((tokens_vis, text_clean, keep_idx))

        feats = vals_all[i]
        text_raw = feats[:seq_len]
        valid_idx = [idx for idx in keep_idx if idx < len(text_raw)]
        if valid_idx:
            text_vals = text_raw[valid_idx]
        else:
            text_vals = np.zeros((0,), dtype=feats.dtype)
        all_text_values.append(text_vals)

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

    if np.any(img_concat < 0):
        absmax_img = float(np.percentile(np.abs(img_concat), 95))
        if absmax_img <= 0:
            absmax_img = float(np.max(np.abs(img_concat))) if img_concat.size else 0.0
        absmax_img = max(absmax_img, 1e-6)
        norm_img = TwoSlopeNorm(vmin=-absmax_img, vcenter=0.0, vmax=absmax_img)
        cmap_img = plt.get_cmap("coolwarm")
    else:
        vmax_img = float(np.percentile(img_concat, 95)) if img_concat.size else 0.0
        vmax_img = max(vmax_img, 1e-6)
        norm_img = Normalize(vmin=0.0, vmax=vmax_img)
        cmap_img = plt.get_cmap(cmap_name)

    # --- figura ---
    fig = plt.figure(figsize=(5 * B, 6))
    gs  = fig.add_gridspec(2, B, height_ratios=[4, 1], hspace=0.05, wspace=0.03)

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
        px_tensor = inputs["pixel_values"][i].detach().cpu()
        H = int(px_tensor.shape[-2])
        W = int(px_tensor.shape[-1])
        side_h = side_h_base if side_h_base else max(1, int(round(H / float(patch_h))))
        side_w = side_w_base if side_w_base else max(1, int(round(W / float(patch_w))))

        patch_vals = np.asarray(img_slice)
        expected = max(1, side_h * side_w)
        actual = patch_vals.size
        if actual == 0:
            patch_vals = np.zeros((expected,), dtype=feats.dtype)
            actual = expected
        if actual != expected:
            side_guess = int(round(math.sqrt(actual))) if actual > 0 else 1
            if side_guess > 0 and side_guess * side_guess == actual:
                side_h = side_w = side_guess
                expected = actual
            else:
                side_h = max(1, int(round(H / float(patch_h))))
                side_w = max(1, int(round(W / float(patch_w))))
                expected = max(1, side_h * side_w)
        if actual < expected:
            patch_vals = np.pad(patch_vals, (0, expected - actual), mode="edge")
        elif actual > expected:
            patch_vals = patch_vals[:expected]

        patch_grid = patch_vals.reshape(side_h, side_w)

        mean = _CLIP_MEAN.to(dtype=px_tensor.dtype, device=px_tensor.device).view(3, 1, 1)
        std = _CLIP_STD.to(dtype=px_tensor.dtype, device=px_tensor.device).view(3, 1, 1)
        img_vis = torch.clamp(px_tensor * std + mean, 0, 1).permute(1, 2, 0).numpy()

        heat_tensor = torch.as_tensor(patch_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        heat_up = F.interpolate(heat_tensor, size=(H, W), mode="nearest").squeeze().numpy()

        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(img_vis, origin="upper", interpolation="nearest", zorder=0)
        ax_img.imshow(
            heat_up,
            cmap=cmap_img,
            norm=norm_img,
            alpha=alpha_img,
            origin="upper",
            interpolation="nearest",
            zorder=1,
        )
        ax_img.set_aspect("equal")
        ax_img.set_xlim(0, W)
        ax_img.set_ylim(H, 0)
        ax_img.margins(0)
        ax_img.set_title(f"{texts[i]}\nIScore {iscore:.2%}", fontsize=14, pad=8)
        ax_img.axis("off")

        # --- texto ---
        toks_vis, text_clean, keep_idx = decoded_info[i]
        text_raw = feats[:tlen]
        valid_idx = [idx for idx in keep_idx if idx < len(text_raw)]
        text_vals = text_raw[valid_idx] if valid_idx else np.zeros((0,), dtype=text_raw.dtype)
        if len(toks_vis) > len(text_vals):
            toks_vis = toks_vis[:len(text_vals)]
        elif len(text_vals) > len(toks_vis):
            text_vals = text_vals[:len(toks_vis)]

        toks_display = _format_tokens(toks_vis)

        ax_txt = fig.add_subplot(gs[1, i])
        ax_txt.axis("off")
        ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)

        ax_txt.text(0.5, 0.85, f"TScore {tscore:.2%}",
                    ha="center", va="center", transform=ax_txt.transAxes, fontsize=13)

        if len(toks_display) == 0 or len(text_vals) == 0:
            ax_txt.text(0.5, 0.35, text_clean if text_clean else "(sin tokens)",
                        ha="center", va="center", transform=ax_txt.transAxes, fontsize=12)
            continue

        # medir ancho de tokens para centrar
        widths = []
        bbox_kw = dict(facecolor="white", pad=0.2, alpha=0)
        for tok in toks_display:
            t = ax_txt.text(0, 0, tok, ha="left", va="center", fontsize=14, bbox=bbox_kw)
            bb = t.get_window_extent(renderer=renderer)
            bb_data = ax_txt.transAxes.inverted().transform([[bb.x1, bb.y1], [bb.x0, bb.y0]])
            widths.append(bb_data[0,0] - bb_data[1,0])
            t.remove()

        gap     = 0.01
        total_w = sum(widths) + gap * max(0, len(toks_display)-1)
        start_x = 0.5 - total_w/2
        x = start_x

        for tok, val, w in zip(toks_display, text_vals, widths):
            color = cmap_text(norm_text(val))
            ax_txt.text(
                x, 0.5, tok,
                ha="left", va="center", fontsize=14, color="black",
                transform=ax_txt.transAxes,
                bbox=dict(facecolor=color, alpha=0.8, edgecolor="white", linewidth=0.5, boxstyle="square,pad=0.2")
            )
            x += w + gap

    # colorbars
    first_pos = fig.axes[0].get_position()
    cax_i = fig.add_axes([first_pos.x1 + 0.01, first_pos.y0, 0.015, first_pos.height])
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_img, norm=norm_img), cax=cax_i, label="Valor SHAP por parche")

    cax_t = fig.add_axes([0.05, 0.03, 0.9, 0.02])
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_text, norm=norm_text),
                 cax=cax_t, orientation="horizontal", label="Valor SHAP por token")

    if return_fig:
        return fig
    plt.show()
