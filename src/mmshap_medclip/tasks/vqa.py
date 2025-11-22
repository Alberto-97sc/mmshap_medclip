# src/mmshap_medclip/tasks/vqa.py
from typing import List, Dict, Any, Optional
import torch
import numpy as np

from mmshap_medclip.tasks.utils import prepare_batch

def run_vqa_one(
    model,
    image,
    question: str,
    candidates: List[str],
    device,
    answer: Optional[str] = None,
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
        amp_if_cuda: Usar mixed precision si está en CUDA
        
    Returns:
        Dict con:
        - prediction: Candidato predicho (el de mayor similitud)
        - similarities: Lista de similitudes para cada candidato
        - candidate_scores: Dict {candidato: similitud}
        - correct: bool (si answer fue proporcionado)
        - logits: Tensor con logits de similitud
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
    
    # Extraer similitudes (logits son las similitudes imagen-texto)
    similarities = logits.squeeze().cpu().detach().numpy()
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
    
    return {
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
    }


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

