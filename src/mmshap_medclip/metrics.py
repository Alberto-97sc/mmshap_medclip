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
    original_text: Optional[str] = None,
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

    # Limpiar texto decodificado de tokens especiales residuales
    if text_decoded:
        # Eliminar tokens especiales comunes que a veces quedan
        for special in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<|startoftext|>", "<|endoftext|>", "<start_of_text>", "<end_of_text>"]:
            text_decoded = text_decoded.replace(special, "")
        text_decoded = text_decoded.strip()

    # Si pudimos decodificar el texto completo, dividir en palabras
    word_shap = OrderedDict()

    # Si tenemos el texto original, usarlo directamente (más confiable)
    if original_text and isinstance(original_text, str) and original_text.strip():
        words = original_text.strip().split()
        
        # Verificar que no haya palabras vacías
        words = [w for w in words if w.strip()]
        
        # IMPORTANTE: Inicializar word_shap con todas las palabras del texto original
        # Esto garantiza que todas las palabras estén presentes, incluso si no se mapean tokens
        for word in words:
            if word.strip() and word not in word_shap:
                word_shap[word] = 0.0  # Valor por defecto, se actualizará si se mapean tokens
        
        # Obtener subtokens y sus scores
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
        special_token_strings = {
            "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]",
            "<s>", "</s>", "<pad>", "<unk>", "<mask>",
            "<|startoftext|>", "<|endoftext|>",
            "<start_of_text>", "<end_of_text>",
            "cls", "sep", "pad", "mask", "unk",
            "", " ", "  "
        }
        special_token_ids = {0, 49406, 49407, 101, 102}

        filtered_subtokens = []
        filtered_scores = []
        for tid, tok, score in zip(token_ids_list, subtokens, raw_shap):
            tok_str = str(tok).strip().lower()
            tok_original = str(tok).strip()
            is_special = (
                tid in special_ids or
                tid in special_token_ids or
                tok in special_tokens or
                tok_str in special_token_strings or
                tok_original in special_token_strings or
                tok_str.startswith("[") and tok_str.endswith("]") or
                tok_str.startswith("<") and tok_str.endswith(">") or
                not tok_str
            )
            if not is_special:
                filtered_subtokens.append(tok)
                filtered_scores.append(score)

        # Mapear tokens a palabras del texto original
        # Estrategia mejorada: usar el texto decodificado como referencia para mapear tokens a palabras
        # Primero intentar decodificar el texto completo para tener una referencia
        text_decoded_ref = None
        try:
            if hasattr(tokenizer, "decode"):
                text_decoded_ref = tokenizer.decode(token_ids_list, skip_special_tokens=True)
        except:
            pass
        
        # Si tenemos texto decodificado, usarlo para mapear mejor
        if text_decoded_ref and text_decoded_ref.strip():
            # Limpiar el texto decodificado
            text_decoded_ref = text_decoded_ref.strip()
            for special in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<|startoftext|>", "<|endoftext|>", "<start_of_text>", "<end_of_text>"]:
                text_decoded_ref = text_decoded_ref.replace(special, "")
            text_decoded_ref = text_decoded_ref.strip()
            
            # Dividir el texto decodificado en palabras para referencia
            words_decoded = text_decoded_ref.split()
            
            # Si el número de palabras coincide aproximadamente, usar mapeo directo
            if abs(len(words) - len(words_decoded)) <= 2 and len(words) > 0:
                # Mapeo más directo: asignar tokens proporcionalmente
                tokens_per_word = len(filtered_subtokens) / len(words)
                
                for word_idx, word in enumerate(words):
                    # La palabra ya está en word_shap (inicializada con 0.0)
                    # Solo actualizar si se mapean tokens
                    word_clean = word.lower().strip().rstrip('.,!?;:')
                    temp_score = 0.0
                    tokens_assigned = 0
                    
                    # Calcular cuántos tokens deberían corresponder a esta palabra
                    start_idx = int(word_idx * tokens_per_word)
                    end_idx = int((word_idx + 1) * tokens_per_word) if word_idx < len(words) - 1 else len(filtered_subtokens)
                    
                    # Asegurar que no excedamos el límite
                    end_idx = min(end_idx, len(filtered_scores))
                    
                    # Acumular scores de los tokens asignados a esta palabra
                    for idx in range(start_idx, end_idx):
                        if idx < len(filtered_scores):
                            temp_score += float(filtered_scores[idx])
                            tokens_assigned += 1
                    
                    if word and word.strip() and tokens_assigned > 0:
                        word_shap[word] = temp_score
                    # Si no se asignaron tokens, la palabra ya tiene 0.0 (inicializada arriba)
            else:
                # Si no coincide, usar estrategia de acumulación mejorada
                subtoken_idx = 0
                for word_idx, word in enumerate(words):
                    # La palabra ya está en word_shap (inicializada con 0.0)
                    # Solo actualizar si se mapean tokens
                    word_clean = word.lower().strip().rstrip('.,!?;:')
                    temp_word = ""
                    temp_score = 0.0
                    tokens_in_word = 0

                    # Acumular tokens hasta formar la palabra completa
                    # Limitar el número de tokens a procesar para evitar procesar tokens de más
                    max_tokens_to_check = min(len(filtered_subtokens) - subtoken_idx, len(word_clean) * 3 + 5)
                    
                    for _ in range(max_tokens_to_check):
                        if subtoken_idx >= len(filtered_subtokens):
                            break
                            
                        tok = str(filtered_subtokens[subtoken_idx])
                        tok_clean = tok.lstrip("Ġ").lstrip("▁").replace("</w>", "").lstrip("#").lower().strip()
                        
                        if not tok_clean:
                            subtoken_idx += 1
                            continue
                        
                        temp_word += tok_clean
                        temp_score += float(filtered_scores[subtoken_idx])
                        tokens_in_word += 1
                        subtoken_idx += 1

                        # Verificar si hemos formado la palabra completa
                        temp_word_clean = temp_word.lower().strip().rstrip('.,!?;:')
                        # Comparación más estricta: la palabra debe estar contenida o ser igual
                        if word_clean == temp_word_clean or (word_clean in temp_word_clean and len(temp_word_clean) <= len(word_clean) * 1.2):
                            if word and word.strip() and tokens_in_word > 0:
                                word_shap[word] = temp_score
                            break
                        elif len(temp_word) > len(word_clean) * 1.3:
                            # Si acumulamos demasiado, asignar y continuar
                            if word and word.strip() and tokens_in_word > 0:
                                word_shap[word] = temp_score
                            break

                    # Si no se asignó score pero se procesaron tokens, actualizar
                    if word and word.strip() and tokens_in_word > 0 and word_shap.get(word, 0.0) == 0.0:
                        word_shap[word] = temp_score
        else:
            # Fallback: usar estrategia de acumulación simple
            subtoken_idx = 0
            for word_idx, word in enumerate(words):
                # La palabra ya está en word_shap (inicializada con 0.0)
                # Solo actualizar si se mapean tokens
                word_clean = word.lower().strip().rstrip('.,!?;:')
                temp_word = ""
                temp_score = 0.0
                tokens_in_word = 0

                # Acumular tokens hasta formar la palabra completa
                # Limitar el número de tokens a procesar para evitar procesar tokens de más
                max_tokens_to_check = min(len(filtered_subtokens) - subtoken_idx, len(word_clean) * 3 + 5)
                
                for _ in range(max_tokens_to_check):
                    if subtoken_idx >= len(filtered_subtokens):
                        break
                        
                    tok = str(filtered_subtokens[subtoken_idx])
                    tok_clean = tok.lstrip("Ġ").lstrip("▁").replace("</w>", "").lstrip("#").lower().strip()
                    
                    if not tok_clean:
                        subtoken_idx += 1
                        continue
                    
                    temp_word += tok_clean
                    temp_score += float(filtered_scores[subtoken_idx])
                    tokens_in_word += 1
                    subtoken_idx += 1

                    # Verificar si hemos formado la palabra completa (comparación más estricta)
                    temp_word_clean = temp_word.lower().strip().rstrip('.,!?;:')
                    if word_clean == temp_word_clean or (word_clean in temp_word_clean and len(temp_word_clean) <= len(word_clean) * 1.2):
                        if word and word.strip() and tokens_in_word > 0:
                            word_shap[word] = temp_score
                        break
                    elif len(temp_word) > len(word_clean) * 1.3:
                        if word and word.strip() and tokens_in_word > 0:
                            word_shap[word] = temp_score
                        break

                # Si no se asignó score pero se procesaron tokens, actualizar
                if word and word.strip() and tokens_in_word > 0 and word_shap.get(word, 0.0) == 0.0:
                    word_shap[word] = temp_score

    elif text_decoded and isinstance(text_decoded, str) and text_decoded.strip():
        # Tenemos el texto decodificado, dividir en palabras
        words = text_decoded.strip().split()

        # Verificar si el texto decodificado es válido
        # IMPORTANTE: BioMedCLIP y algunos tokenizadores BERT no insertan espacios correctamente

        # 1. Verificar número mínimo de palabras
        min_expected_words = max(5, len(token_ids_list) // 2)

        # 2. Detectar palabras extremadamente largas (tokens pegados sin espacios)
        has_very_long_words = any(len(w) > 20 for w in words)

        # 3. Verificar si hay muy pocas palabras (menos de 1/3 de los tokens)
        has_too_few_words = len(words) < (len(token_ids_list) // 3)

        # 4. Verificar si la primera palabra es sospechosamente larga
        first_word_too_long = len(words) > 0 and len(words[0]) > 15

        # 5. NUEVO: Contar espacios en el texto original
        # Si el texto tiene muy pocos espacios, probablemente no se decodificó bien
        space_count = text_decoded.count(' ')
        min_expected_spaces = max(3, len(token_ids_list) // 4)
        has_too_few_spaces = space_count < min_expected_spaces

        # 6. NUEVO: Ratio caracteres/palabras (detecta palabras pegadas)
        # Si el promedio de caracteres por palabra es muy alto, hay problema
        avg_word_length = len(text_decoded.replace(' ', '')) / max(len(words), 1)
        avg_word_too_long = avg_word_length > 12

        # Si alguna validación falla, usar método de subtokens
        decode_is_valid = (
            len(words) >= min_expected_words and
            not has_very_long_words and
            not has_too_few_words and
            not first_word_too_long and
            not has_too_few_spaces and
            not avg_word_too_long
        )

        if (decode_is_valid or len(token_ids_list) <= 5):
            # El texto parece válido, procesar normalmente
            # Intentar obtener subtokens para mapear scores
            subtokens = []
            try:
                if hasattr(tokenizer, "convert_ids_to_tokens"):
                    subtokens = tokenizer.convert_ids_to_tokens(token_ids_list)
                else:
                    subtokens = [str(tid) for tid in token_ids_list]
            except:
                subtokens = [str(tid) for tid in token_ids_list]

            # Filtrar tokens especiales - lista más completa
            special_ids = set(getattr(tokenizer, "all_special_ids", []))
            special_tokens = set(getattr(tokenizer, "all_special_tokens", []))

            # Tokens especiales comunes en diferentes tokenizadores
            special_token_strings = {
                "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]",
                "<s>", "</s>", "<pad>", "<unk>", "<mask>",
                "<|startoftext|>", "<|endoftext|>",
                "<start_of_text>", "<end_of_text>",
                "cls", "sep", "pad", "mask", "unk",
                "", " ", "  "  # tokens vacíos
            }

            # IDs especiales comunes
            special_token_ids = {0, 49406, 49407, 101, 102}  # PAD, SOT, EOT, CLS, SEP

            # Filtrar subtokens y sus scores
            filtered_subtokens = []
            filtered_scores = []
            for tid, tok, score in zip(token_ids_list, subtokens, raw_shap):
                tok_str = str(tok).strip().lower()
                tok_original = str(tok).strip()

                # Verificar si es token especial
                is_special = (
                    tid in special_ids or
                    tid in special_token_ids or
                    tok in special_tokens or
                    tok_str in special_token_strings or
                    tok_original in special_token_strings or
                    tok_str.startswith("[") and tok_str.endswith("]") or
                    tok_str.startswith("<") and tok_str.endswith(">") or
                    not tok_str  # token vacío
                )

                if not is_special:
                    filtered_subtokens.append(tok)
                    filtered_scores.append(score)

            # Si tenemos el mismo número de palabras que subtokens filtrados, asignar directamente
            if len(words) == len(filtered_scores):
                for word, score in zip(words, filtered_scores):
                    if word and word.strip():
                        word_shap[word] = float(score)
            else:
                # Intentar agrupar subtokens en palabras
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
            # El decode falló (muy pocas palabras), usar método de subtokens
            text_decoded = None

    if not text_decoded or not word_shap:
        # Fallback: usar subtokens directamente
        subtokens = []
        try:
            if hasattr(tokenizer, "convert_ids_to_tokens"):
                subtokens = tokenizer.convert_ids_to_tokens(token_ids_list)
            else:
                subtokens = [str(tid) for tid in token_ids_list]
        except:
            subtokens = [str(tid) for tid in token_ids_list]

        # Filtrado robusto de tokens especiales
        ignore = set(getattr(tokenizer, "all_special_tokens", []))
        special_ids = set(getattr(tokenizer, "all_special_ids", []))
        special_token_ids = {0, 49406, 49407, 101, 102}
        special_token_strings = {
            "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]",
            "<s>", "</s>", "<pad>", "<unk>", "<mask>",
            "<|startoftext|>", "<|endoftext|>",
            "<start_of_text>", "<end_of_text>",
            "cls", "sep", "pad", "mask", "unk"
        }

        cur_word, cur_score = "", 0.0

        def flush():
            nonlocal cur_word, cur_score
            if cur_word and cur_word.strip():
                word_shap[cur_word] = float(cur_score)
                cur_word, cur_score = "", 0.0

        for tid, tok, score in zip(token_ids_list, subtokens, raw_shap):
            tok_str = str(tok).strip().lower()
            tok_original = str(tok).strip()

            # Verificar si es token especial
            is_special = (
                tid in special_ids or
                tid in special_token_ids or
                tok in ignore or
                tok_str in special_token_strings or
                tok_original in special_token_strings or
                tok_str.startswith("[") and tok_str.endswith("]") or
                tok_str.startswith("<") and tok_str.endswith(">") or
                not tok_str
            )

            if is_special:
                continue

            # Heurísticas de segmentación para diferentes tipos de tokenizadores
            tok_str_original = str(tok)

            # WordPiece (BERT, PubMedBERT): tokens que continúan empiezan con ##
            is_continuation = tok_str_original.startswith("##")

            # SentencePiece: tokens que empiezan palabra tienen Ġ o ▁
            start_of_word = tok_str_original.startswith("Ġ") or tok_str_original.startswith("▁")

            # BPE: tokens que terminan palabra tienen </w>
            end_of_word = tok_str_original.endswith("</w>")

            # Limpiar el token
            piece = tok_str_original
            piece = piece.lstrip("Ġ").lstrip("▁")
            piece = piece.replace("</w>", "")
            piece = piece.lstrip("#")  # Remover ## de WordPiece

            # Lógica de agrupación:
            # - Si NO es continuación y NO empieza con Ġ/▁ → nueva palabra (WordPiece)
            # - Si empieza con Ġ/▁ → nueva palabra (SentencePiece)
            # - Si es continuación (##) → agregar a palabra actual

            if start_of_word and cur_word:
                # SentencePiece: Ġ o ▁ indica nueva palabra
                flush()
            elif not is_continuation and cur_word and not start_of_word:
                # WordPiece: si NO tiene ## y ya hay palabra, es nueva palabra
                flush()

            cur_word += piece
            cur_score += float(score)

            if end_of_word:
                flush()

        # Flush final para la última palabra
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
