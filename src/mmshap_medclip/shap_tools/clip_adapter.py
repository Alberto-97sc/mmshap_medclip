"""Adapter unificado para modelos tipo CLIP usados en SHAP."""
import inspect
from typing import Optional

import numpy as np
import torch


def _accepts_kwarg(fn, name: str) -> bool:
    """Return True if ``fn`` declares a parameter named ``name``."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # callable sin signature introspectable
        return False

    for param in sig.parameters.values():
        if param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD) and param.name == name:
            return True
    return False


def _is_transformers_clip(model) -> bool:
    """Heuristic to detect 🤗 Transformers CLIP-style models."""
    if _accepts_kwarg(getattr(model, "forward", lambda *_, **__: None), "pixel_values"):
        return True
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None) if config is not None else None
    return hasattr(vision_config, "patch_size")


def _normalize_logits_output(logits: torch.Tensor) -> torch.Tensor:
    """Ensure logits have shape ``[B, 1]`` selecting diagonal elements if needed."""
    if logits.ndim == 2 and logits.shape[1] != 1:
        if logits.shape[0] == logits.shape[1]:
            logits = torch.diag(logits).unsqueeze(1)
        else:
            idx = torch.arange(min(logits.shape[0], logits.shape[1]), device=logits.device)
            logits = logits[idx, idx].unsqueeze(1)
    elif logits.ndim == 1:
        logits = logits.unsqueeze(1)
    return logits


class ClipAdapter:
    """Interfaz unificada para invocar modelos CLIP/OpenCLIP.

    ``__call__`` siempre acepta ``pixel_values`` e ``input_ids`` y devuelve un tensor
    ``[B, 1]`` con logits por imagen.
    """

    def __init__(self, model, device: torch.device, use_amp: bool = False):
        self.model = model
        self.device = device
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.is_hf = _is_transformers_clip(model)

    @torch.no_grad()
    def __call__(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pixel_values = pixel_values.to(self.device)
        input_ids = input_ids.to(self.device)
        if attention_mask is not None and hasattr(attention_mask, "to"):
            attention_mask = attention_mask.to(self.device)

        if self.is_hf:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = getattr(outputs, "logits_per_image", None)
                if logits is None:
                    raise ValueError("El modelo Transformers CLIP no devolvió logits_per_image.")
                logits = _normalize_logits_output(logits)
            return logits

        # Ruta OpenCLIP/WhyXRayCLIP
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            img = self.model.encode_image(pixel_values)      # [B, D]
            txt = self.model.encode_text(input_ids)          # [B, D]
            img = img / img.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            txt = txt / txt.norm(dim=-1, keepdim=True).clamp_min(1e-8)

            logit_scale = getattr(self.model, "logit_scale", None)
            if logit_scale is None:
                scale = 1.0
            else:
                if torch.is_tensor(logit_scale):
                    scale = torch.exp(logit_scale).item()
                else:
                    scale = float(np.exp(float(logit_scale)))

            logits = scale * (img * txt).sum(dim=-1, keepdim=True)
        return logits
