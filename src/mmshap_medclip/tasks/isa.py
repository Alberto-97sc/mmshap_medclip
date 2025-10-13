# src/mmshap_medclip/tasks/isa.py
import re
from typing import List, Dict, Any, Optional, Tuple
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
    explain: Any = True,
    plot: bool = False,
    amp_if_cuda: bool = True,
    max_evals: Optional[int] = None,
) -> Dict[str, Any]:
    """Pipeline mínimo de ISA para 1 (imagen, texto).

    Args:
        explain: Puede ser un booleano o un diccionario con opciones. Si es un
            diccionario, se reconocen las llaves ``enabled`` (bool), ``plot`` (bool)
            y ``max_evals`` (int). Cualquier valor de ``max_evals`` tendrá prioridad
            sobre el argumento homónimo de la función.
        max_evals: Presupuesto opcional de evaluaciones para el explicador de SHAP.
            Si no se especifica, se calculará automáticamente para satisfacer el
            requisito ``2 * num_features + 1``.
    """
    explain_enabled, plot = _resolve_explain_options(explain, plot)
    max_evals = _resolve_max_evals(explain, max_evals)

    # 1) forward
    inputs, logits = prepare_batch(model, [caption], [image], device=device, debug_tokens=False, amp_if_cuda=amp_if_cuda)
    out: Dict[str, Any] = {
        "inputs": inputs,
        "logit": float(logits[0, 0]),
        "image": image,
        "text": caption,
    }
    if not explain_enabled:
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
    eval_budget = max(min_evals, max_evals or 512)

    explainer = shap.explainers.Permutation(
        predict_fn,
        masker,
        max_evals=eval_budget,
        silent=True,
    )

    call_kwargs = {"max_evals": eval_budget, "silent": True}
    shap_values = _call_shap_with_budget(explainer, X_clean.cpu(), call_kwargs)

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
    explain: Any = True,
    amp_if_cuda: bool = True,
    max_evals: Optional[int] = None,
) -> Dict[str, Any]:
    """Pipeline de ISA para batch (lista de PILs y textos).

    Args:
        explain: Puede ser un booleano o un diccionario con opciones (ver
            ``run_isa_one``).
        max_evals: Presupuesto opcional de evaluaciones para el explicador de SHAP.
            Si no se especifica, se calculará automáticamente para satisfacer el
            requisito ``2 * num_features + 1``.
    """
    explain_enabled, _ = _resolve_explain_options(explain, plot=False)
    max_evals = _resolve_max_evals(explain, max_evals)

    inputs, logits = prepare_batch(model, captions, images, device=device, debug_tokens=False, amp_if_cuda=amp_if_cuda)
    result: Dict[str, Any] = {
        "inputs": inputs,
        "logits": logits.detach().cpu().numpy(),
    }
    if not explain_enabled:
        return result

    nb_text_tokens_tensor, _ = compute_text_token_lengths(inputs, model.tokenizer)
    image_token_ids_expanded, imginfo = make_image_token_ids(inputs, model)
    X_clean, _ = concat_text_image_tokens(inputs, image_token_ids_expanded, device=device)

    masker = build_masker(nb_text_tokens_tensor, tokenizer=model.tokenizer)
    predict_fn = Predictor(model, inputs, patch_size=imginfo["patch_size"], device=device, use_amp=amp_if_cuda)

    num_features = int(X_clean.shape[1])
    min_evals = 2 * num_features + 1
    eval_budget = max(min_evals, max_evals or 512)

    explainer = shap.explainers.Permutation(
        predict_fn,
        masker,
        max_evals=eval_budget,
        silent=True,
    )
    call_kwargs = {"max_evals": eval_budget, "silent": True}
    shap_values = _call_shap_with_budget(explainer, X_clean.cpu(), call_kwargs)

    mm_scores = [compute_mm_score(shap_values, model.tokenizer, inputs, i=i) for i in range(len(captions))]
    iscores   = [compute_iscore(shap_values, inputs, i=i) for i in range(len(captions))]

    result.update({
        "shap_values": shap_values,
        "mm_scores": mm_scores,
        "iscores": iscores,
    })
    return result


def _resolve_explain_options(explain: Any, plot: bool) -> Tuple[bool, bool]:
    """Return a tuple ``(enabled, plot)`` based on the ``explain`` argument."""

    if isinstance(explain, dict):
        enabled = explain.get("enabled", True)
        plot_flag = explain.get("plot", plot)
        return bool(enabled), bool(plot_flag)

    return bool(explain), bool(plot)


def _resolve_max_evals(explain: Any, max_evals: Optional[int]) -> Optional[int]:
    """Resolve the SHAP evaluation budget from ``max_evals`` or ``explain`` options."""

    if max_evals is not None:
        return max_evals

    if isinstance(explain, dict):
        return explain.get("max_evals")

    return None


def _call_shap_with_budget(explainer, X, call_kwargs):
    """Call a SHAP explainer ensuring the evaluation budget satisfies tokenizer demands."""

    try:
        return explainer(X, **call_kwargs)
    except ValueError as exc:
        message = str(exc)
        if "max_evals" not in message or "too low" not in message:
            raise

        required = _extract_required_evals(message)
        if required is None:
            raise

        current = call_kwargs.get("max_evals", 0) or 0
        updated = max(current, required)
        call_kwargs["max_evals"] = updated

        call_defaults = getattr(getattr(explainer, "__call__", None), "__kwdefaults__", None)
        if isinstance(call_defaults, dict):
            call_defaults["max_evals"] = updated

        return explainer(X, **call_kwargs)


_REQUIRED_EVALS_RE = re.compile(r"2 \* num_features \+ 1 = (?P<required>\d+)")


def _extract_required_evals(message: str) -> Optional[int]:
    match = _REQUIRED_EVALS_RE.search(message)
    if not match:
        return None
    try:
        return int(match.group("required"))
    except (ValueError, TypeError):
        return None
