# src/mmshap_medclip/metrics.py
from typing import Tuple, Optional, Union, Dict
from collections import OrderedDict
import numpy as np

def compute_mm_score(
    shap_values: Union[np.ndarray, "shap._explanation.Explanation"],
    tokenizer,
    inputs: Dict[str, "np.ndarray"],
    i: int,
    text_length: Optional[int] = None,
) -> Tuple[float, "OrderedDict[str, float]"]:
    """
    Multimodal Score (TScore) para el ejemplo i.
    Devuelve (tscore, dict_palabra→score_con_signo).

    - tscore = sum(|SHAP_text|) / (sum(|SHAP_text|) + sum(|SHAP_img|))
    - dict_palabra usa el signo original de SHAP para cada palabra agregada.
    """
    vals = shap_values.values if hasattr(shap_values, "values") else shap_values
    # Aplanar a vector de features del ejemplo i
    if vals.ndim == 3:      # (B, O, L)
        v = vals[i, 0, :]
    elif vals.ndim == 2:    # (B, L)
        v = vals[i, :]
    elif vals.ndim == 1:    # (L,)
        v = vals
    else:
        raise ValueError(f"Forma inesperada de shap_values: {vals.shape}")

    # Longitud de texto (tokens válidos)
    if text_length is None:
        if "attention_mask" in inputs:
            text_length = int(inputs["attention_mask"][i].sum().item())
        else:
            # fallback: usar toda la secuencia de texto
            text_length = int(inputs["input_ids"][i].shape[0])
    text_length = max(1, min(text_length, v.shape[-1]))

    # Particiones texto / imagen
    txt_vals = np.abs(v[:text_length])
    img_vals = np.abs(v[text_length:])
    text_contrib  = float(txt_vals.sum())
    image_contrib = float(img_vals.sum())
    denom = (text_contrib + image_contrib) if (text_contrib + image_contrib) > 0 else 1.0
    tscore = text_contrib / denom

    # Diccionario palabra→score (con signo)
    token_ids = inputs["input_ids"][i][:text_length].detach().cpu().tolist()
    subtokens = tokenizer.convert_ids_to_tokens(token_ids)
    raw_shap  = v[:text_length]

    # Ignorar especiales (si el tokenizer no los tiene, usa set vacío)
    ignore = set(getattr(tokenizer, "all_special_tokens", []))
    word_shap = OrderedDict()
    cur_word, cur_score = "", 0.0

    def flush():
        nonlocal cur_word, cur_score
        if cur_word:
            word_shap[cur_word] = float(cur_score)
            cur_word, cur_score = "", 0.0

    for tok, score in zip(subtokens, raw_shap):
        if tok in ignore:
            continue

        # heurísticas de segmentación
        start_of_word = tok.startswith("Ġ") or tok.startswith("▁")
        end_of_word   = tok.endswith("</w>")

        piece = tok
        piece = piece.lstrip("Ġ").lstrip("▁")
        piece = piece.replace("</w>", "")
        piece = piece.lstrip("#")  # WordPiece

        if start_of_word:
            flush()
        cur_word += piece
        cur_score += float(score)
        if end_of_word:
            flush()
    flush()

    return tscore, word_shap


def compute_iscore(
    shap_values: Union[np.ndarray, "shap._explanation.Explanation"],
    inputs: Dict[str, "np.ndarray"],
    i: int,
    text_length: Optional[int] = None,
) -> float:
    """
    IScore (fracción visual) = sum(|SHAP_img|) / (sum(|SHAP_text|) + sum(|SHAP_img|)).
    Útil para reportar el complemento del TScore.
    """
    vals = shap_values.values if hasattr(shap_values, "values") else shap_values
    if vals.ndim == 3:
        v = vals[i, 0, :]
    elif vals.ndim == 2:
        v = vals[i, :]
    elif vals.ndim == 1:
        v = vals
    else:
        raise ValueError(f"Forma inesperada de shap_values: {vals.shape}")

    if text_length is None:
        if "attention_mask" in inputs:
            text_length = int(inputs["attention_mask"][i].sum().item())
        else:
            text_length = int(inputs["input_ids"][i].shape[0])
    text_length = max(1, min(text_length, v.shape[-1]))

    txt_vals = np.abs(v[:text_length])
    img_vals = np.abs(v[text_length:])
    text_contrib  = float(txt_vals.sum())
    image_contrib = float(img_vals.sum())
    denom = (text_contrib + image_contrib) if (text_contrib + image_contrib) > 0 else 1.0
    return image_contrib / denom
