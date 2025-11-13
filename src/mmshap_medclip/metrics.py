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
    token_ids = inputs["input_ids"][i][:text_length]
    if hasattr(token_ids, "detach"):
        token_ids = token_ids.detach().cpu()
    token_ids_list = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    
    raw_shap = v[:text_length]
    
    # Intentar decodificar los tokens
    # Primero intentar obtener el texto completo
    text_decoded = None
    if hasattr(tokenizer, "decode"):
        try:
            text_decoded = tokenizer.decode(token_ids_list, skip_special_tokens=True)
        except:
            try:
                text_decoded = tokenizer.decode(token_ids_list)
            except:
                pass
    
    # Si pudimos decodificar el texto completo, dividir en palabras
    word_shap = OrderedDict()
    
    if text_decoded and isinstance(text_decoded, str) and text_decoded.strip():
        # Tenemos el texto decodificado, dividir en palabras
        words = text_decoded.strip().split()
        
        # Intentar obtener subtokens para mapear scores
        subtokens = []
        try:
            if hasattr(tokenizer, "convert_ids_to_tokens"):
                subtokens = tokenizer.convert_ids_to_tokens(token_ids_list)
            else:
                subtokens = [str(tid) for tid in token_ids_list]
        except:
            subtokens = [str(tid) for tid in token_ids_list]
        
        # Filtrar tokens especiales
        special_ids = set(getattr(tokenizer, "all_special_ids", []))
        special_tokens = set(getattr(tokenizer, "all_special_tokens", []))
        special_token_strings = {"[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]", 
                                "<s>", "</s>", "<pad>", "<unk>", "<mask>",
                                "<|startoftext|>", "<|endoftext|>", "<|endoftext|>"}
        
        # Filtrar subtokens y sus scores
        filtered_subtokens = []
        filtered_scores = []
        for tid, tok, score in zip(token_ids_list, subtokens, raw_shap):
            if (tid not in special_ids and 
                tok not in special_tokens and 
                tok not in special_token_strings and
                str(tok).strip() not in special_token_strings):
                filtered_subtokens.append(tok)
                filtered_scores.append(score)
        
        # Si tenemos el mismo número de palabras que subtokens filtrados, asignar directamente
        if len(words) == len(filtered_scores):
            for word, score in zip(words, filtered_scores):
                if word and word.strip():
                    word_shap[word] = float(score)
        else:
            # Intentar agrupar subtokens en palabras
            cur_word = ""
            cur_score = 0.0
            subtoken_idx = 0
            
            for word in words:
                # Acumular subtokens hasta reconstruir la palabra
                word_clean = word.lower().strip()
                temp_word = ""
                temp_score = 0.0
                
                while subtoken_idx < len(filtered_subtokens) and len(temp_word) < len(word_clean) + 5:
                    tok = str(filtered_subtokens[subtoken_idx])
                    # Limpiar token
                    tok_clean = tok.lstrip("Ġ").lstrip("▁").replace("</w>", "").lstrip("#").lower()
                    temp_word += tok_clean
                    temp_score += float(filtered_scores[subtoken_idx])
                    subtoken_idx += 1
                    
                    if word_clean in temp_word:
                        break
                
                if temp_word:
                    word_shap[word] = temp_score
                elif subtoken_idx < len(filtered_scores):
                    # Fallback: asignar score del siguiente token
                    word_shap[word] = float(filtered_scores[subtoken_idx])
                    subtoken_idx += 1
    else:
        # Fallback: usar subtokens directamente
        subtokens = []
        try:
            if hasattr(tokenizer, "convert_ids_to_tokens"):
                subtokens = tokenizer.convert_ids_to_tokens(token_ids_list)
            else:
                subtokens = [str(tid) for tid in token_ids_list]
        except:
            subtokens = [str(tid) for tid in token_ids_list]
        
        ignore = set(getattr(tokenizer, "all_special_tokens", []))
        cur_word, cur_score = "", 0.0
        
        def flush():
            nonlocal cur_word, cur_score
            if cur_word and cur_word.strip():
                word_shap[cur_word] = float(cur_score)
                cur_word, cur_score = "", 0.0
        
        for tok, score in zip(subtokens, raw_shap):
            if tok in ignore:
                continue
            
            # heurísticas de segmentación
            start_of_word = tok.startswith("Ġ") or tok.startswith("▁")
            end_of_word = tok.endswith("</w>")
            
            piece = str(tok)
            piece = piece.lstrip("Ġ").lstrip("▁")
            piece = piece.replace("</w>", "")
            piece = piece.lstrip("#")  # WordPiece
            
            if start_of_word and cur_word:
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
