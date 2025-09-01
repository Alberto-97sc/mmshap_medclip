# src/mmshap_medclip/tasks/classification.py
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import shap

from mmshap_medclip.tasks.utils import (
    prepare_batch, compute_text_token_lengths, make_image_token_ids, concat_text_image_tokens
)
from mmshap_medclip.shap_tools.masker import build_masker
from mmshap_medclip.shap_tools.predictor import ClassificationPredictor
from mmshap_medclip.metrics import compute_mm_score, compute_iscore

def run_classification_one(
    model,
    image,
    class_names: List[str],
    device,
    explain: bool = True,
    plot: bool = False,
    amp_if_cuda: bool = True,
) -> Dict[str, Any]:
    """Pipeline de clasificación para 1 imagen con múltiples clases posibles."""
    # 1) forward - procesamos la imagen con todas las clases
    inputs, logits = prepare_classification_batch(model, class_names, [image], device=device, amp_if_cuda=amp_if_cuda)
    
    # Para clasificación, usamos logits_per_image que ahora tiene shape [num_classes, num_classes]
    # Tomamos la diagonal que representa la similitud entre cada imagen y su clase correspondiente
    if logits.dim() == 2:
        # Si es [num_classes, num_classes], tomamos la diagonal
        logits_diagonal = torch.diag(logits)
    else:
        # Si es [1, num_classes], lo usamos directamente
        logits_diagonal = logits.squeeze()
    
    # Calculamos probabilidades
    probs = torch.softmax(logits_diagonal, dim=0)
    predicted_class_idx = torch.argmax(probs).item()
    predicted_class = class_names[predicted_class_idx]
    
    out: Dict[str, Any] = {
        "inputs": inputs,
        "logits": logits.detach().cpu().numpy(),
        "probabilities": probs.detach().cpu().numpy(),
        "predicted_class": predicted_class,
        "predicted_class_idx": predicted_class_idx,
        "class_names": class_names,
        "image": image,
    }
    
    if not explain:
        return out

    # 2) tokens y X_clean para SHAP
    nb_text_tokens_tensor, _ = compute_text_token_lengths(inputs, model.tokenizer)
    image_token_ids_expanded, imginfo = make_image_token_ids(inputs, model)
    X_clean, _ = concat_text_image_tokens(inputs, image_token_ids_expanded, device=device)

    # 3) SHAP - explicamos la predicción para la clase predicha
    masker = build_masker(nb_text_tokens_tensor, tokenizer=model.tokenizer)
    predict_fn = ClassificationPredictor(
        model, inputs, class_names, predicted_class_idx, 
        patch_size=imginfo["patch_size"], device=device, use_amp=amp_if_cuda
    )
    explainer = shap.Explainer(predict_fn, masker, silent=True)
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
            texts=class_names,
            mm_scores=[(tscore, word_shap)],
            model_wrapper=model,
            return_fig=True,
        )
        out["fig"] = fig

    return out


def prepare_classification_batch(
    model_wrapper,
    class_names: List[str],
    images,
    device: torch.device = None,
    padding: bool = True,
    to_rgb: bool = True,
    amp_if_cuda: bool = True,
) -> tuple:
    """
    Prepara el batch para clasificación con RClip.
    Procesa una imagen con múltiples nombres de clases.
    
    Nota: Para mantener compatibilidad con el modelo, expandimos la imagen
    para que coincida con el número de clases, creando un batch balanceado.
    """
    # normaliza inputs a listas
    if not isinstance(images, (list, tuple)):
        images = [images]

    # asegúrate de RGB si se solicita
    if to_rgb:
        images = [im.convert("RGB") for im in images]

    # tokenización con el processor del wrapper
    processor = model_wrapper.processor
    
    # Para clasificación, necesitamos que imagen y texto tengan el mismo batch size
    # Expandimos la imagen para que coincida con el número de clases
    num_classes = len(class_names)
    expanded_images = images * num_classes  # Repetir la imagen para cada clase
    
    inputs = processor(text=class_names, images=expanded_images, return_tensors="pt", padding=padding)

    # mover al device del wrapper si no se pasó explícito
    if device is None:
        device = next(model_wrapper.parameters()).device
    
    # mover inputs al device
    from mmshap_medclip.devices import move_to_device
    inputs = move_to_device(inputs, device)

    # forward sin gradientes (con AMP si hay CUDA)
    with torch.inference_mode():
        if amp_if_cuda and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model_wrapper(**inputs)
        else:
            outputs = model_wrapper(**inputs)

    # Para clasificación, usamos logits_per_image
    logits = outputs.logits_per_image  # [1, num_classes]
    return inputs, logits


def run_classification_batch(
    model,
    images,
    class_names: List[str],
    device,
    explain: bool = True,
    amp_if_cuda: bool = True,
) -> Dict[str, Any]:
    """Pipeline de clasificación para batch de imágenes."""
    inputs, logits = prepare_classification_batch(model, class_names, images, device=device, amp_if_cuda=amp_if_cuda)
    
    # Calculamos probabilidades para cada imagen
    probs = torch.softmax(logits, dim=1)  # [batch_size, num_classes]
    predicted_classes_idx = torch.argmax(probs, dim=1)  # [batch_size]
    predicted_classes = [class_names[idx.item()] for idx in predicted_classes_idx]
    
    result: Dict[str, Any] = {
        "inputs": inputs,
        "logits": logits.detach().cpu().numpy(),
        "probabilities": probs.detach().cpu().numpy(),
        "predicted_classes": predicted_classes,
        "predicted_classes_idx": predicted_classes_idx.tolist(),
        "class_names": class_names,
    }
    
    if not explain:
        return result

    # Para batch, explicamos cada imagen individualmente
    nb_text_tokens_tensor, _ = compute_text_token_lengths(inputs, model.tokenizer)
    image_token_ids_expanded, imginfo = make_image_token_ids(inputs, model)
    X_clean, _ = concat_text_image_tokens(inputs, image_token_ids_expanded, device=device)

    # SHAP para cada imagen
    all_shap_values = []
    mm_scores = []
    iscores = []
    
    for i in range(len(images)):
        masker = build_masker(nb_text_tokens_tensor[i:i+1], tokenizer=model.tokenizer)
        predict_fn = ClassificationPredictor(
            model, 
            {k: v[i:i+1] for k, v in inputs.items()},  # slice del batch
            class_names, 
            predicted_classes_idx[i].item(),
            patch_size=imginfo["patch_size"], 
            device=device, 
            use_amp=amp_if_cuda
        )
        explainer = shap.Explainer(predict_fn, masker, silent=True)
        shap_vals = explainer(X_clean[i:i+1].cpu())
        all_shap_values.append(shap_vals)
        
        # métricas para esta imagen
        tscore, word_shap = compute_mm_score(shap_vals, model.tokenizer, {k: v[i:i+1] for k, v in inputs.items()}, i=0)
        iscore = compute_iscore(shap_vals, {k: v[i:i+1] for k, v in inputs.items()}, i=0)
        
        mm_scores.append((tscore, word_shap))
        iscores.append(iscore)

    result.update({
        "shap_values": all_shap_values,
        "mm_scores": mm_scores,
        "iscores": iscores,
    })
    
    return result
