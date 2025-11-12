from typing import Callable, Optional, Union
import numpy as np
import torch

def build_masker(
    nb_text_tokens: Union[torch.Tensor, np.ndarray, int, list],
    tokenizer=None,
    bos_id: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> Callable:
    """
    Crea un masker para SHAP que:
      - Recibe (mask, x) y devuelve x enmascarado (ids → 0 donde mask=False).
      - Asegura preservar BOS/EOS en la parte de texto de cada fila del batch.
    Supone que x = [input_ids_text | image_patch_ids].

    Args:
        nb_text_tokens: nº de tokens reales por texto (B,) o escalar (broadcast).
                        Idealmente de attention_mask.sum(1).
        tokenizer:     opcional, para obtener bos/eos id si existen.
        bos_id/eos_id: opcionales; si no vienen y el tokenizer no tiene, usa 49406/49407.

    Returns:
        custom_masker(mask, x) -> torch.LongTensor (misma forma que x)
    """
    # --- normalizar nb_text_tokens a tensor 1D ---
    if isinstance(nb_text_tokens, torch.Tensor):
        nt = nb_text_tokens.clone()
    else:
        nt = torch.as_tensor(nb_text_tokens)
    if nt.dim() == 0:
        nt = nt.unsqueeze(0)

    # --- resolver BOS/EOS ---
    if bos_id is None or eos_id is None:
        bos_from_tok = getattr(tokenizer, "bos_token_id", None) if tokenizer is not None else None
        eos_from_tok = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
        bos_id = bos_id if bos_id is not None else (bos_from_tok if bos_from_tok is not None else 49406)
        eos_id = eos_id if eos_id is not None else (eos_from_tok if eos_from_tok is not None else 49407)

    def custom_masker(mask, x):
        """
        Mascarilla para SHAP (robusta a 1D/2D y CPU/GPU).
        - x: ids [texto | parches], puede venir como [L] o [B, L]
        - mask: bool, puede venir como [L] o [B, L]
        """
        # --- normalizar x a tensor long ---
        if isinstance(x, np.ndarray):
            masked_X = torch.from_numpy(x).to(dtype=torch.long)
        else:
            masked_X = x.to(dtype=torch.long)

        # asegurar 2D
        if masked_X.dim() == 1:
            masked_X = masked_X.unsqueeze(0)

        # --- normalizar mask ---
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=masked_X.device)
        if mask_t.dim() == 1:
            mask_t = mask_t.unsqueeze(0)
        if mask_t.shape != masked_X.shape:
            mask_t = mask_t.expand_as(masked_X)  # broadcast si es [1, L]

        # --- aplicar máscara (ids → 0) ---
        masked_X = masked_X.clone()
        masked_X.masked_fill_(~mask_t, 0)

        # --- preservar BOS/EOS en cada fila (parte de texto) ---
        B, L = masked_X.shape
        nt_dev = nt.to(masked_X.device)

        # ajustar longitud de nt a B (broadcast o recorte)
        if nt_dev.shape[0] != B:
            if nt_dev.numel() == 1:
                nt_dev = nt_dev.expand(B)
            else:
                nt_dev = nt_dev[:B]

        # Obtener vocab_size si está disponible del tokenizer
        vocab_size = None
        if tokenizer is not None:
            vocab_size = getattr(tokenizer, "vocab_size", None)
            if vocab_size is None:
                # Intentar desde config del modelo si el tokenizer lo tiene
                if hasattr(tokenizer, "model_max_length"):
                    # Algunos tokenizers exponen esto, pero si no hay vocab_size usar valor seguro
                    vocab_size = None

        # Validar y clamp BOS/EOS IDs al rango válido (usar variables locales)
        valid_bos_id = bos_id
        valid_eos_id = eos_id
        if vocab_size is not None:
            valid_bos_id = max(0, min(bos_id, vocab_size - 1))
            valid_eos_id = max(0, min(eos_id, vocab_size - 1))

        # Asegurar que los valores estén en rango válido antes de asignar
        masked_X[:, 0] = valid_bos_id
        eos_idx = (nt_dev - 1).clamp(min=0, max=L - 1)
        rows = torch.arange(B, device=masked_X.device)
        masked_X[rows, eos_idx] = valid_eos_id

        # Validación final: clamp todos los valores al rango válido si tenemos vocab_size
        if vocab_size is not None:
            masked_X = masked_X.clamp(min=0, max=vocab_size - 1)

        return masked_X

    return custom_masker
