# src/mmshap_medclip/tasks/isa.py
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
    explain: bool = True,
    plot: bool = False,
    amp_if_cuda: bool = True,
) -> Dict[str, Any]:
    """Pipeline mínimo de ISA para 1 (imagen, texto)."""
    inputs, logits = prepare_batch(
        model, [caption], [image], device=device, debug_tokens=False, amp_if_cuda=amp_if_cuda
    )

    out: Dict[str, Any] = {
        "inputs": inputs,
        "logits": logits,
        "logit": float(logits.squeeze().item()),
        "image": image,
        "text": caption,
        "model_wrapper": model,
    }

    if not (explain or plot):
        return out

    explanation = explain_isa(
        model,
        inputs,
        device=device,
        amp_if_cuda=amp_if_cuda,
    )
    out.update(explanation)

    if plot and "shap_values" in out:
        fig = plot_isa(
            image=image,
            caption=caption,
            isa_output=out,
            model_wrapper=model,
            display_plot=True,
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
    inputs, logits = prepare_batch(
        model, captions, images, device=device, debug_tokens=False, amp_if_cuda=amp_if_cuda
    )
    result: Dict[str, Any] = {
        "inputs": inputs,
        "logits": logits.detach().cpu().numpy(),
        "model_wrapper": model,
    }
    if not explain:
        return result

    shap_values, mm_scores, iscores = _compute_isa_shap(
        model,
        inputs,
        device=device,
        amp_if_cuda=amp_if_cuda,
    )

    result.update({
        "shap_values": shap_values,
        "mm_scores": mm_scores,
        "iscores": [float(s) for s in iscores],
    })
    return result


def explain_isa(
    model,
    inputs: Dict[str, Any],
    device,
    amp_if_cuda: bool = True,
) -> Dict[str, Any]:
    """Calcula explicaciones ISA para un batch preparado."""
    shap_values, mm_scores, iscores = _compute_isa_shap(
        model,
        inputs,
        device=device,
        amp_if_cuda=amp_if_cuda,
    )

    out: Dict[str, Any] = {
        "shap_values": shap_values,
        "mm_scores": mm_scores,
        "iscores": [float(s) for s in iscores],
    }

    if len(mm_scores) == 1:
        tscore, _ = mm_scores[0]
        out.update({
            "tscore": float(tscore),
            "iscore": float(iscores[0]),
        })

    return out


def plot_isa(
    image,
    caption: str,
    isa_output: Dict[str, Any],
    model_wrapper=None,
    display_plot: bool = True,
):
    """Genera la figura de heatmaps para un resultado ISA."""
    from mmshap_medclip.vis.heatmaps import plot_text_image_heatmaps

    if model_wrapper is None:
        model_wrapper = isa_output.get("model_wrapper")
    if model_wrapper is None:
        raise ValueError("Se requiere el wrapper del modelo para plot_isa.")

    shap_values = isa_output.get("shap_values")
    mm_scores = isa_output.get("mm_scores")
    inputs = isa_output.get("inputs")
    if shap_values is None or mm_scores is None or inputs is None:
        raise ValueError("plot_isa necesita 'shap_values', 'mm_scores' e 'inputs' en isa_output.")

    fig = plot_text_image_heatmaps(
        shap_values=shap_values,
        inputs=inputs,
        tokenizer=model_wrapper.tokenizer,
        images=image,
        texts=[caption],
        mm_scores=mm_scores,
        model_wrapper=model_wrapper,
        return_fig=True,
    )

    if display_plot:
        fig.show()

    return fig


def _compute_isa_shap(
    model,
    inputs: Dict[str, Any],
    device,
    amp_if_cuda: bool = True,
) -> Tuple[Any, List[Tuple[float, Dict[str, float]]], List[float]]:
    """Aplica SHAP al batch dado y retorna valores por muestra."""
    nb_text_tokens_tensor, _ = compute_text_token_lengths(inputs, model.tokenizer)
    image_token_ids_expanded, imginfo = make_image_token_ids(inputs, model)
    X_clean, _ = concat_text_image_tokens(inputs, image_token_ids_expanded, device=device)

    masker = build_masker(nb_text_tokens_tensor, tokenizer=model.tokenizer)
    predict_fn = Predictor(
        model,
        inputs,
        patch_size=imginfo["patch_size"],
        device=device,
        use_amp=amp_if_cuda,
    )

    # --- Ajuste automático del presupuesto para el Permutation explainer ---
    def _as_hw_tuple(value):
        if value is None:
            return None, None
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None, None
            if len(value) == 1:
                val = int(value[0])
                return val, val
            return int(value[0]), int(value[1])
        val = int(value)
        return val, val

    n_tokens = int(inputs["input_ids"].shape[1])

    n_patches = int(imginfo.get("num_patches") or 0)
    if n_patches <= 0:
        patch = getattr(model, "vision_patch_size", None)
        if patch is None:
            patch = getattr(model, "patch_size", None)
        if patch is None and "patch_size" in imginfo:
            patch = imginfo["patch_size"]

        img_sz = getattr(model, "vision_input_size", None)
        if img_sz is None:
            img_sz = getattr(model, "image_size", None)
        if img_sz is None and hasattr(model, "config") and hasattr(model.config, "vision_config"):
            img_sz = getattr(model.config.vision_config, "image_size", None)

        patch_h, patch_w = _as_hw_tuple(patch)
        img_h, img_w = _as_hw_tuple(img_sz)

        if patch_h is None or patch_w is None:
            patch_h = patch_w = 14
        if img_h is None or img_w is None:
            _, _, img_h, img_w = inputs["pixel_values"].shape

        if patch_h <= 0 or patch_w <= 0:
            patch_h = patch_w = 14

        n_patches = max(1, (img_h // patch_h) * (img_w // patch_w))

    n_features = n_tokens + n_patches
    min_needed = 2 * n_features + 1

    call_kwargs = {}
    maybe_call_kwargs = getattr(model, "shap_call_kwargs", None)
    if isinstance(maybe_call_kwargs, dict):
        call_kwargs.update(maybe_call_kwargs)
    maybe_call_kwargs = inputs.get("shap_call_kwargs") if isinstance(inputs, dict) else None
    if isinstance(maybe_call_kwargs, dict):
        call_kwargs.update(maybe_call_kwargs)

    desired = call_kwargs.get("max_evals", 0) or 0
    max_evals = max(int(desired), int(min_needed))
    call_kwargs["max_evals"] = max_evals

    explainer = shap.Explainer(predict_fn, masker, silent=True)
    shap_values = explainer(X_clean.cpu(), **call_kwargs)

    batch_size = inputs["input_ids"].shape[0]
    mm_scores = [compute_mm_score(shap_values, model.tokenizer, inputs, i=i) for i in range(batch_size)]
    iscores = [compute_iscore(shap_values, inputs, i=i) for i in range(batch_size)]

    return shap_values, mm_scores, iscores
