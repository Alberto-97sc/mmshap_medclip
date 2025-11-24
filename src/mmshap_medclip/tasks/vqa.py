# src/mmshap_medclip/tasks/vqa.py
from typing import List, Dict, Any, Optional, Tuple
import math
import torch
import numpy as np
import shap

from mmshap_medclip.tasks.utils import (
    prepare_batch, compute_text_token_lengths, make_image_token_ids, concat_text_image_tokens
)
from mmshap_medclip.shap_tools.masker import build_masker
from mmshap_medclip.shap_tools.vqa_predictor import VQAPredictor
from mmshap_medclip.metrics import compute_mm_score, compute_iscore


VQA_PATCH_TARGET_GRID = 7


def _extract_text_feature_from_inputs(model_wrapper, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    model = getattr(model_wrapper, "model", model_wrapper)
    text_kwargs = {"input_ids": inputs["input_ids"]}
    if "attention_mask" in inputs:
        text_kwargs["attention_mask"] = inputs["attention_mask"]

    with torch.inference_mode():
        if hasattr(model, "get_text_features"):
            feat = model.get_text_features(**text_kwargs)
        elif hasattr(model, "encode_text"):
            feat = model.encode_text(text_kwargs["input_ids"])
        else:
            raise ValueError("El modelo no expone get_text_features/encode_text.")

    ref_device = next(model.parameters()).device if isinstance(model, torch.nn.Module) else inputs["input_ids"].device
    feat = feat.to(ref_device)
    feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return feat.detach()

def run_vqa_one(
    model,
    image,
    question: str,
    candidates: List[str],
    device,
    answer: Optional[str] = None,
    explain: bool = True,
    plot: bool = False,
    target_logit: str = "correct",
    amp_if_cuda: bool = True,
) -> Dict[str, Any]:
    """
    Pipeline de VQA multiple-choice para 1 (imagen, pregunta, candidatos).
    
    Lógica:
    1. Construir prompts tipo "Question: <pregunta> Answer: <candidate>"
    2. Codificar imagen y cada texto candidato con el modelo
    3. Calcular similitud imagen-texto para cada candidato
    4. Elegir como predicción el candidato con mayor similitud
    
    Args:
        model: Wrapper del modelo CLIP (debe tener processor, tokenizer, model)
        image: PIL.Image
        question: Texto de la pregunta
        candidates: Lista de candidatos (respuestas posibles)
        device: Dispositivo (cuda/cpu)
        answer: Respuesta correcta (opcional, para evaluación)
        explain: Si True, calcula explicaciones SHAP
        plot: Si True, genera visualización (requiere explain=True)
        target_logit: "correct" o "predicted" - qué logit explicar con SHAP
        amp_if_cuda: Usar mixed precision si está en CUDA
        
    Returns:
        Dict con:
        - prediction: Candidato predicho (el de mayor similitud)
        - similarities: Lista de similitudes para cada candidato
        - candidate_scores: Dict {candidato: similitud}
        - correct: bool (si answer fue proporcionado)
        - logits: Tensor con logits de similitud
        - shap_values, mm_scores, iscores (si explain=True)
    """
    if not candidates:
        raise ValueError("La lista de candidatos no puede estar vacía")
    
    # Construir prompts para cada candidato
    prompts = [
        f"Question: {question} Answer: {candidate}"
        for candidate in candidates
    ]
    
    # Preparar batch: imagen repetida para cada candidato
    images = [image] * len(candidates)
    
    # Codificar y calcular similitudes
    inputs, logits = prepare_batch(
        model,
        prompts,
        images,
        device=device,
        debug_tokens=False,
        amp_if_cuda=amp_if_cuda
    )
    
    # Extraer similitudes (logits_per_image es [n_imgs, n_texts])
    logits_cpu = logits.detach().cpu()
    if logits_cpu.ndim == 0:
        similarities_tensor = logits_cpu.view(1)
    elif logits_cpu.ndim == 1:
        similarities_tensor = logits_cpu
    elif logits_cpu.ndim == 2:
        if logits_cpu.shape[0] == logits_cpu.shape[1]:
            similarities_tensor = torch.diag(logits_cpu)
        elif logits_cpu.shape[1] == 1:
            similarities_tensor = logits_cpu.squeeze(1)
        elif logits_cpu.shape[0] == 1:
            similarities_tensor = logits_cpu.squeeze(0)
        else:
            raise ValueError(
                f"Forma de logits inesperada {tuple(logits_cpu.shape)}; "
                "se esperaba matriz cuadrada o vector por candidato."
            )
    else:
        raise ValueError(
            f"Dimensiones de logits no soportadas: {logits_cpu.ndim}"
        )

    similarities = similarities_tensor.numpy()
    if similarities.ndim == 0:
        similarities = np.array([similarities])
    
    # Encontrar el candidato con mayor similitud
    pred_idx = int(np.argmax(similarities))
    prediction = candidates[pred_idx]
    
    # Construir dict de scores
    candidate_scores = {
        candidate: float(sim)
        for candidate, sim in zip(candidates, similarities)
    }
    
    # Verificar si la predicción es correcta (si se proporcionó answer)
    correct = None
    if answer is not None:
        correct = (prediction.lower().strip() == answer.lower().strip())
    
    out: Dict[str, Any] = {
        "prediction": prediction,
        "prediction_idx": pred_idx,
        "similarities": similarities.tolist(),
        "candidate_scores": candidate_scores,
        "correct": correct,
        "logits": logits,
        "question": question,
        "candidates": candidates,
        "answer": answer,
        "model_wrapper": model,
        "image": image,
    }
    
    base_inputs = None
    answer_text = None
    if explain or plot:
        target_candidate = answer if (target_logit == "correct" and answer) else prediction
        if not target_candidate:
            target_candidate = prediction
        answer_text = f"Answer: {target_candidate}"
        target_prompt = f"Question: {question}"
        base_inputs, _ = prepare_batch(
            model,
            [target_prompt],
            [image],
            device=device,
            debug_tokens=False,
            amp_if_cuda=amp_if_cuda
        )
        out["inputs"] = base_inputs

    if not (explain or plot):
        return out
    
    # Calcular explicaciones SHAP
    explanation = explain_vqa(
        model,
        base_inputs,
        question,
        candidates,
        device=device,
        answer=answer,
        target_logit=target_logit,
        amp_if_cuda=amp_if_cuda,
        original_text=question,
        answer_text=answer_text,
    )
    out.update(explanation)
    
    if plot and "shap_values" in out:
        fig = plot_vqa(
            image=image,
            question=question,
            vqa_output=out,
            model_wrapper=model,
            display_plot=True,
        )
        out["fig"] = fig
    
    return out


def run_vqa_batch(
    model,
    images: List,
    questions: List[str],
    candidates_list: List[List[str]],
    device,
    answers: Optional[List[str]] = None,
    amp_if_cuda: bool = True,
) -> Dict[str, Any]:
    """
    Pipeline de VQA multiple-choice para batch.
    
    Args:
        model: Wrapper del modelo CLIP
        images: Lista de PIL.Images
        questions: Lista de preguntas
        candidates_list: Lista de listas de candidatos (una por pregunta)
        device: Dispositivo
        answers: Lista de respuestas correctas (opcional)
        amp_if_cuda: Usar mixed precision si está en CUDA
        
    Returns:
        Dict con resultados por muestra
    """
    if len(images) != len(questions) or len(images) != len(candidates_list):
        raise ValueError(
            f"Longitudes inconsistentes: images={len(images)}, "
            f"questions={len(questions)}, candidates_list={len(candidates_list)}"
        )
    
    if answers is not None and len(answers) != len(images):
        raise ValueError(f"answers debe tener la misma longitud que images")
    
    results = []
    for i in range(len(images)):
        result = run_vqa_one(
            model=model,
            image=images[i],
            question=questions[i],
            candidates=candidates_list[i],
            device=device,
            answer=answers[i] if answers is not None else None,
            amp_if_cuda=amp_if_cuda,
        )
        results.append(result)
    
    # Agregar métricas agregadas si hay respuestas
    if answers is not None:
        correct_count = sum(1 for r in results if r["correct"])
        accuracy = correct_count / len(results)
        
        return {
            "results": results,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total": len(results),
        }
    
    return {"results": results}


def evaluate_vqa_dataset(
    model,
    dataset,
    device,
    max_samples: Optional[int] = None,
    amp_if_cuda: bool = True,
) -> Dict[str, Any]:
    """
    Evalúa un dataset VQA completo.
    
    Args:
        model: Wrapper del modelo CLIP
        dataset: Dataset que devuelve dicts con "image", "question", "answer", "candidates"
        device: Dispositivo
        max_samples: Número máximo de muestras a evaluar (None = todas)
        amp_if_cuda: Usar mixed precision si está en CUDA
        
    Returns:
        Dict con métricas agregadas y resultados por muestra
    """
    total = len(dataset)
    if max_samples is not None:
        total = min(total, max_samples)
    
    all_results = []
    correct_count = 0
    
    for i in range(total):
        sample = dataset[i]
        
        result = run_vqa_one(
            model=model,
            image=sample["image"],
            question=sample["question"],
            candidates=sample["candidates"],
            device=device,
            answer=sample.get("answer"),
            amp_if_cuda=amp_if_cuda,
        )
        
        all_results.append(result)
        if result["correct"]:
            correct_count += 1
    
    accuracy = correct_count / total if total > 0 else 0.0
    
    # Calcular métricas por categoría si está disponible
    category_metrics = {}
    if "category" in dataset[0]:
        category_results = {}
        for i, sample in enumerate(dataset[:total]):
            category = sample.get("category", "unknown")
            if category not in category_results:
                category_results[category] = {"correct": 0, "total": 0}
            category_results[category]["total"] += 1
            if all_results[i]["correct"]:
                category_results[category]["correct"] += 1
        
        category_metrics = {
            cat: {
                "accuracy": res["correct"] / res["total"] if res["total"] > 0 else 0.0,
                "correct": res["correct"],
                "total": res["total"]
            }
            for cat, res in category_results.items()
        }
    
    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total": total,
        "category_metrics": category_metrics,
        "results": all_results,
    }


def explain_vqa(
    model,
    inputs: Dict[str, Any],
    question: str,
    candidates: List[str],
    device,
    answer: Optional[str] = None,
    target_logit: str = "correct",
    amp_if_cuda: bool = True,
    original_text: Optional[str] = None,
    answer_text: Optional[str] = None,
    text_feature: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Calcula explicaciones VQA para un batch preparado."""
    shap_values, mm_scores, iscores, text_len = _compute_vqa_shap(
        model,
        inputs,
        question,
        candidates,
        device=device,
        answer=answer,
        target_logit=target_logit,
        amp_if_cuda=amp_if_cuda,
        original_text=original_text,
        answer_text=answer_text,
        text_feature=text_feature,
    )

    out: Dict[str, Any] = {
        "shap_values": shap_values,
        "mm_scores": mm_scores,
        "iscores": [float(s) for s in iscores],
        "text_len": text_len,
    }

    if len(mm_scores) == 1:
        tscore, _ = mm_scores[0]
        out.update({
            "tscore": float(tscore),
            "iscore": float(iscores[0]),
        })

    return out


def plot_vqa(
    image,
    question: str,
    vqa_output: Dict[str, Any],
    model_wrapper=None,
    display_plot: bool = True,
):
    """Genera la figura de heatmaps para un resultado VQA."""
    from mmshap_medclip.vis.heatmaps import plot_text_image_heatmaps

    if model_wrapper is None:
        model_wrapper = vqa_output.get("model_wrapper")
    if model_wrapper is None:
        raise ValueError("Se requiere el wrapper del modelo para plot_vqa.")

    shap_values = vqa_output.get("shap_values")
    mm_scores = vqa_output.get("mm_scores")
    inputs = vqa_output.get("inputs")
    text_len = vqa_output.get("text_len")
    if shap_values is None or mm_scores is None or inputs is None:
        raise ValueError("plot_vqa necesita 'shap_values', 'mm_scores' e 'inputs' en vqa_output.")

    fig = plot_text_image_heatmaps(
        shap_values=shap_values,
        inputs=inputs,
        tokenizer=model_wrapper.tokenizer,
        images=image,
        texts=[question],
        mm_scores=mm_scores,
        model_wrapper=model_wrapper,
        return_fig=True,
        text_len=text_len,
        target_grid_size=7,
    )

    if display_plot:
        fig.show()

    return fig


def _build_patch_groups(grid_h: int, grid_w: int, target_grid: Optional[int]) -> Optional[List[List[int]]]:
    if not target_grid or grid_h <= target_grid or grid_w <= target_grid:
        return None

    block_h = math.ceil(grid_h / target_grid)
    block_w = math.ceil(grid_w / target_grid)

    groups: List[List[int]] = []
    for gh in range(0, grid_h, block_h):
        for gw in range(0, grid_w, block_w):
            indices: List[int] = []
            for r in range(gh, min(gh + block_h, grid_h)):
                for c in range(gw, min(gw + block_w, grid_w)):
                    indices.append(r * grid_w + c)
            if indices:
                groups.append(indices)
    return groups if groups else None


def _make_group_token_ids(original_ids: torch.Tensor, num_groups: int) -> torch.Tensor:
    device = original_ids.device
    dtype = original_ids.dtype
    base = torch.arange(1, num_groups + 1, device=device, dtype=dtype)
    return base.unsqueeze(0).expand(original_ids.shape[0], -1).clone()


def _compute_vqa_shap(
    model,
    inputs: Dict[str, Any],
    question: str,
    candidates: List[str],
    device,
    answer: Optional[str] = None,
    target_logit: str = "correct",
    amp_if_cuda: bool = True,
    original_text: Optional[str] = None,
    answer_text: Optional[str] = None,
    text_feature: Optional[torch.Tensor] = None,
) -> Tuple[Any, List[Tuple[float, Dict[str, float]]], List[float], int]:
    """Aplica SHAP al batch dado para VQA y retorna valores por muestra y text_len."""
    nb_text_tokens_tensor, _ = compute_text_token_lengths(inputs, model.tokenizer)
    image_token_ids_expanded, imginfo = make_image_token_ids(inputs, model)
    patch_groups = None
    target_grid = VQA_PATCH_TARGET_GRID
    grid_h = int(imginfo.get("grid_h") or 0)
    grid_w = int(imginfo.get("grid_w") or 0)
    if target_grid and grid_h > 0 and grid_w > 0:
        patch_groups = _build_patch_groups(grid_h, grid_w, target_grid)
        if patch_groups:
            image_token_ids_expanded = _make_group_token_ids(image_token_ids_expanded, len(patch_groups))

    if text_feature is None and answer_text:
        tokenizer = getattr(model.tokenizer if hasattr(model, "tokenizer") else model, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("No se pudo obtener tokenizer para encodear la respuesta.")
        text_tokens = tokenizer(
            [answer_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
        text_feature = _extract_text_feature_from_inputs(model, text_tokens)

    # Pasar nb_text_tokens para usar solo tokens reales (sin padding)
    X_clean, text_len = concat_text_image_tokens(
        inputs, image_token_ids_expanded, device=device,
        nb_text_tokens=nb_text_tokens_tensor, tokenizer=model.tokenizer
    )

    masker = build_masker(nb_text_tokens_tensor, tokenizer=model.tokenizer)
    predict_fn = VQAPredictor(
        model,
        inputs,
        question=question,
        candidates=candidates,
        answer_correct=answer,
        target_logit=target_logit,
        patch_size=imginfo["patch_size"],
        device=device,
        use_amp=amp_if_cuda,
        text_len=text_len,
        patch_groups=patch_groups,
        answer_text=answer_text or "",
        text_feature=text_feature,
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

    n_patches = len(patch_groups) if patch_groups else int(imginfo.get("num_patches") or 0)
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
    # Pasar text_len y original_texts para que compute_mm_score use el texto original cuando esté disponible
    mm_scores = [
        compute_mm_score(
            shap_values,
            model.tokenizer,
            inputs,
            i=i,
            text_length=text_len,
            original_text=original_text if original_text else None
        )
        for i in range(batch_size)
    ]
    iscores = [compute_iscore(shap_values, inputs, i=i, text_length=text_len) for i in range(batch_size)]

    return shap_values, mm_scores, iscores, text_len

