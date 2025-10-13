# src/mmshap_medclip/tasks/isa.py
from typing import List, Dict, Any, Optional
import numpy as np
import shap

from mmshap_medclip.tasks.utils import (
    prepare_batch, compute_text_token_lengths, make_image_token_ids, concat_text_image_tokens
)
from mmshap_medclip.shap_tools.masker import build_masker
from mmshap_medclip.shap_tools.predictor import Predictor
from mmshap_medclip.metrics import compute_mm_score, compute_iscore

def run_isa_one(
    model,
    image,
    caption: str,
    device,
    explain: bool = True,
    plot: bool = False,
    amp_if_cuda: bool = True,
) -> Dict[str, Any]:
    """Pipeline mínimo de ISA para 1 (imagen, texto)."""
    # 1) forward
    inputs, logits = prepare_batch(model, [caption], [image], device=device, debug_tokens=False, amp_if_cuda=amp_if_cuda)
    out: Dict[str, Any] = {
        "inputs": inputs,
        "logit": float(logits[0, 0]),
        "image": image,
        "text": caption,
    }
    if not explain:
        return out

    # 2) tokens y X_clean
    nb_text_tokens_tensor, _ = compute_text_token_lengths(inputs, model.tokenizer)
    image_token_ids_expanded, imginfo = make_image_token_ids(inputs, model)
    X_clean, _ = concat_text_image_tokens(inputs, image_token_ids_expanded, device=device)

    # 3) SHAP
    masker = build_masker(nb_text_tokens_tensor, tokenizer=model.tokenizer)
    predict_fn = Predictor(model, inputs, patch_size=imginfo["patch_size"], device=device, use_amp=amp_if_cuda)

    # SHAP (Permutation) requiere al menos 2 * num_features + 1 evaluaciones para una
    # explicación válida. Ajustamos dinámicamente ``max_evals`` para evitar errores cuando
    # el número de tokens (features) supera el default (500).
    num_features = int(X_clean.shape[1])
    min_evals = 2 * num_features + 1
    max_evals = max(min_evals, 512)

    explainer = shap.Explainer(predict_fn, masker, silent=True, max_evals=max_evals)
    shap_values = explainer(X_clean.cpu())

    # 4) métricas
    tscore, word_shap = compute_mm_score(shap_values, model.tokenizer, inputs, i=0)
    iscore = compute_iscore(shap_values, inputs, i=0)
    out.update({
        "shap_values": shap_values,
        "mm_scores": [(tscore, word_shap)],
        "tscore": float(tscore),
        "iscore": float(iscore),
    })

    # 5) figura opcional
    if plot:
        from mmshap_medclip.vis.heatmaps import plot_text_image_heatmaps
        fig = plot_text_image_heatmaps(
            shap_values=shap_values,
            inputs=inputs,
            tokenizer=model.tokenizer,
            images=image,
            texts=[caption],
            mm_scores=[(tscore, word_shap)],
            model_wrapper=model,
            return_fig=True,
        )
        out["fig"] = fig

    return out


def run_isa_batch(
    model,
    images,
    captions: List[str],
    device,
    explain: bool = True,
    amp_if_cuda: bool = True,
) -> Dict[str, Any]:
    """Pipeline de ISA para batch (lista de PILs y textos)."""
    inputs, logits = prepare_batch(model, captions, images, device=device, debug_tokens=False, amp_if_cuda=amp_if_cuda)
    result: Dict[str, Any] = {
        "inputs": inputs,
        "logits": logits.detach().cpu().numpy(),
    }
    if not explain:
        return result

    nb_text_tokens_tensor, _ = compute_text_token_lengths(inputs, model.tokenizer)
    image_token_ids_expanded, imginfo = make_image_token_ids(inputs, model)
    X_clean, _ = concat_text_image_tokens(inputs, image_token_ids_expanded, device=device)

    masker = build_masker(nb_text_tokens_tensor, tokenizer=model.tokenizer)
    predict_fn = Predictor(model, inputs, patch_size=imginfo["patch_size"], device=device, use_amp=amp_if_cuda)

    num_features = int(X_clean.shape[1])
    min_evals = 2 * num_features + 1
    max_evals = max(min_evals, 512)

    explainer = shap.Explainer(predict_fn, masker, silent=True, max_evals=max_evals)
    shap_values = explainer(X_clean.cpu())

    mm_scores = [compute_mm_score(shap_values, model.tokenizer, inputs, i=i) for i in range(len(captions))]
    iscores   = [compute_iscore(shap_values, inputs, i=i) for i in range(len(captions))]

    result.update({
        "shap_values": shap_values,
        "mm_scores": mm_scores,
        "iscores": iscores,
    })
    return result
