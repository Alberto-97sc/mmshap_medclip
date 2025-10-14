# src/mmshap_medclip/vis/heatmaps.py
from typing import List, Tuple, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image

def plot_text_image_heatmaps(
    shap_values: Union[np.ndarray, "shap._explanation.Explanation"],
    inputs: dict,
    tokenizer,
    images: Union[Image.Image, List[Image.Image]],
    texts: List[str],
    mm_scores: List[Tuple[float, dict]],     # [(tscore, OrderedDict(word->score)), ...]
    model_wrapper,
    cmap_name: str = "RdYlBu_r",
    alpha_img: float = 0.60,
    return_fig: bool = False,
):
    """
    Dibuja, por muestra del batch, el heatmap de parches de imagen (fracción visual)
    y un renglón de tokens coloreado por |SHAP| agregado a nivel palabra.
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
    m = getattr(model_wrapper, "model", model_wrapper)
    ps = getattr(getattr(getattr(m, "config", None), "vision_config", None), "patch_size", None)
    if ps is None:
        ps = getattr(getattr(getattr(m, "vision_model", None), "config", None), "patch_size", None)
    if ps is None:
        raise ValueError("No pude inferir patch_size; pásalo vía model_wrapper.")

    if isinstance(ps, (list, tuple)):
        if len(ps) != 2:
            raise ValueError(f"patch_size iterable inesperado: {ps}")
        patch_h, patch_w = int(ps[0]), int(ps[1])
    else:
        patch_h = patch_w = int(ps)

    _, _, H, W = inputs["pixel_values"].shape
    assert H % patch_h == 0 and W % patch_w == 0, f"Imagen {H}x{W} no divisible por patch_size={(patch_h, patch_w)}"
    grid_h, grid_w = H // patch_h, W // patch_w
    num_patches = grid_h * grid_w

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

    # --- normalizaciones globales ---
    # a) para imagen: fracción visual por parche (0..1)
    all_fracs = []
    for i in range(B):
        feats = vals_all[i]
        tlen  = seq_lens[i]
        ta = np.abs(feats[:tlen])
        ia = np.abs(feats[tlen:tlen + num_patches])
        s  = ta.sum() + ia.sum()
        all_fracs.append(ia / s if s > 0 else np.zeros_like(ia))
    global_vmax_img = float(np.max(np.concatenate(all_fracs))) if B > 0 else 1.0

    # b) para texto: magnitud por palabra (usar mm_scores)
    all_word_mags = []
    for _, word_shap in mm_scores:
        if word_shap:
            all_word_mags.extend([abs(v) for v in word_shap.values()])
    global_vmax_txt = float(max(all_word_mags)) if all_word_mags else 1.0

    cmap      = plt.get_cmap(cmap_name)
    norm_img  = Normalize(vmin=0, vmax=global_vmax_img)
    norm_text = Normalize(vmin=0, vmax=global_vmax_txt)

    # --- figura ---
    fig = plt.figure(figsize=(5*B, 6))
    gs  = fig.add_gridspec(2, B, height_ratios=[4, 1], hspace=0.05, wspace=0.03)

    # for measuring token widths to center text row
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def reconstruct_heat(rel_img: np.ndarray) -> np.ndarray:
        heat = np.zeros((H, W), dtype=np.float32)
        for k, v in enumerate(rel_img[:num_patches]):
            r, c = k // grid_w, k % grid_w
            r0, r1 = r * patch_h, (r + 1) * patch_h
            c0, c1 = c * patch_w, (c + 1) * patch_w
            heat[r0:r1, c0:c1] = v
        return heat

    for i, (tscore, word_shap) in enumerate(mm_scores):
        feats  = vals_all[i]
        tlen   = seq_lens[i]
        ta, ia = np.abs(feats[:tlen]), np.abs(feats[tlen:tlen + num_patches])
        tot    = ta.sum() + ia.sum()
        rel_img = ia / tot if tot > 0 else np.zeros_like(ia)
        iscore  = float(ia.sum() / tot) if tot > 0 else 0.0

        # --- imagen ---
        heat = reconstruct_heat(rel_img)
        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(images_for_batch[i].resize((W, H)).convert("RGB"), alpha=1.0)
        ax_img.imshow(heat, cmap=cmap, norm=norm_img, alpha=alpha_img)
        ax_img.set_title(f"{texts[i]}\nIScore {iscore:.2%}", fontsize=14, pad=8)
        ax_img.axis("off")

        # --- texto ---
        toks     = list(word_shap.keys())
        vals_tok = [abs(v) for v in word_shap.values()]

        ax_txt = fig.add_subplot(gs[1, i])
        ax_txt.axis("off")
        ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)

        ax_txt.text(0.5, 0.85, f"TScore {tscore:.2%}",
                    ha="center", va="center", transform=ax_txt.transAxes, fontsize=13)

        # medir ancho de tokens para centrar
        widths = []
        bbox_kw = dict(facecolor="white", pad=0.2, alpha=0)
        for tok in toks:
            t = ax_txt.text(0, 0, tok, ha="left", va="center", fontsize=14, bbox=bbox_kw)
            bb = t.get_window_extent(renderer=renderer)
            bb_data = ax_txt.transAxes.inverted().transform([[bb.x1, bb.y1], [bb.x0, bb.y0]])
            widths.append(bb_data[0,0] - bb_data[1,0])
            t.remove()

        gap     = 0.01
        total_w = sum(widths) + gap * max(0, len(toks)-1)
        start_x = 0.5 - total_w/2
        x = start_x

        for tok, val, w in zip(toks, vals_tok, widths):
            color = cmap(norm_text(val))
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
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm_img), cax=cax_i, label="Fracción visual por parche")

    cax_t = fig.add_axes([0.05, 0.03, 0.9, 0.02])
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm_text),
                 cax=cax_t, orientation="horizontal", label="Importancia SHAP por palabra")

    if return_fig:
        return fig
    plt.show()
