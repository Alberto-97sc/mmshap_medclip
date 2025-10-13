"""Utilidades específicas para experimentos con WhyXrayCLIP sobre ROCO."""

from __future__ import annotations

import random
import re
from contextlib import nullcontext
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def clean_roco_caption(text: str) -> str:
    """Normaliza captions de ROCO removiendo prefijos y espacios repetidos."""
    cleaned = re.sub(r"^\s*ROCO_\d+\s*[\t,|;]\s*", "", str(text)).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _build_keyword_pattern(keywords: Sequence[str]) -> str:
    escaped = [re.escape(k.lower()) for k in keywords if k]
    return "|".join(escaped) if escaped else ""


def pick_chestxray_sample(
    dataset,
    keywords: Sequence[str] = ("chest x-ray", "lung"),
    reproducible: bool = False,
    seed: int = 42,
) -> Dict[str, object]:
    """Selecciona una muestra del dataset ROCO filtrando por keywords médicas."""
    if not hasattr(dataset, "df"):
        raise AttributeError("Se requiere un dataset ROCO con atributo 'df'.")

    caption_col = getattr(dataset, "caption_key", None)
    if caption_col is None:
        raise AttributeError("El dataset debe exponer 'caption_key'.")

    df = dataset.df
    pattern = _build_keyword_pattern(keywords)
    if not pattern:
        raise ValueError("Proporciona al menos una keyword para filtrar.")

    captions = df[caption_col].astype(str)
    mask = captions.str.lower().str.contains(pattern, na=False)
    positions = np.flatnonzero(mask.to_numpy())
    if positions.size == 0:
        raise RuntimeError("No se encontraron captions que coincidieran con las keywords solicitadas.")

    rng = np.random.default_rng(seed if reproducible else None)
    pos = int(rng.choice(positions))

    sample = dataset[pos]
    row = df.iloc[pos]
    caption_clean = clean_roco_caption(row[caption_col])

    return {
        "index": pos,
        "image": sample["image"],
        "caption_clean": caption_clean,
        "caption_raw": row[caption_col],
        "meta": {**sample.get("meta", {}), "row_index": pos},
    }


def sample_negative_captions(
    dataset,
    positive_caption: str,
    k: int = 15,
    reproducible: bool = False,
    seed: int = 42,
) -> List[str]:
    """Toma captions alternas del dataset, excluyendo la positiva, para zero-shot."""
    if k <= 0:
        return []

    caption_col = getattr(dataset, "caption_key", None)
    if caption_col is None:
        raise AttributeError("El dataset debe exponer 'caption_key'.")

    captions = (
        dataset.df[caption_col]
        .astype(str)
        .map(clean_roco_caption)
        .dropna()
        .drop_duplicates()
    )
    captions = captions[captions.str.len() > 0]

    positive_norm = clean_roco_caption(positive_caption).lower()
    candidates = captions[captions.str.lower() != positive_norm].tolist()
    if not candidates:
        return []

    size = min(k, len(candidates))
    if reproducible:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=size, replace=False)
        selected = [candidates[int(i)] for i in idx]
    else:
        selected = random.sample(candidates, size)
        random.shuffle(selected)
    return selected[:k]


def score_alignment(
    model_wrapper,
    image,
    caption: str,
    device: Optional[torch.device] = None,
    negatives: Optional[Iterable[str]] = None,
    temperature: float = 100.0,
    amp_if_cuda: bool = True,
) -> Dict[str, object]:
    """Evalúa la alineación imagen-texto con WhyXrayCLIP, con o sin negativos."""
    negatives = list(negatives or [])
    device = device or next(model_wrapper.parameters()).device

    processor = getattr(model_wrapper, "processor", None)
    if processor is None:
        raise AttributeError("El wrapper del modelo debe exponer un 'processor'.")

    # Prepara tensores
    pixel_values = processor.process_images(image).to(device)
    text_tokens = processor.tokenizer([caption] + negatives)
    if not isinstance(text_tokens, torch.Tensor):
        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
    text_tokens = text_tokens.to(device)

    use_amp = amp_if_cuda and device.type == "cuda"
    amp_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()

    with torch.inference_mode(), amp_ctx:
        image_features = model_wrapper.model.encode_image(pixel_values)
        text_features = model_wrapper.model.encode_text(text_tokens)

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    sims = (image_features @ text_features.T)[0]

    sim_pos = float(sims[0].item())
    score_01 = (sim_pos + 1.0) / 2.0

    if not negatives:
        return {
            "similarity": sim_pos,
            "score_01": score_01,
            "negatives": [],
        }

    logits = temperature * sims.unsqueeze(0)
    probs = torch.softmax(logits, dim=-1)[0]

    candidates = [caption] + negatives
    ranking = sorted(zip(candidates, probs.tolist()), key=lambda x: x[1], reverse=True)
    rank_true = 1 + next(i for i, (txt, _) in enumerate(ranking) if txt == caption)

    return {
        "similarity": sim_pos,
        "score_01": score_01,
        "candidates": candidates,
        "logits": logits.detach().cpu(),
        "probs": probs.detach().cpu(),
        "ranking": ranking,
        "rank_true": rank_true,
        "temperature": temperature,
    }
