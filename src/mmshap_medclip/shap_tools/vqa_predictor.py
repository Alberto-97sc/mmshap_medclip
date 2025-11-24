# src/mmshap_medclip/shap_tools/vqa_predictor.py
from typing import Dict, Optional, Union, Tuple, List
from contextlib import nullcontext
import numpy as np
import torch

class VQAPredictor:
    """
    Callable para SHAP en VQA: aplica máscaras de parches sobre pixel_values y tokens de texto,
    calcula similitudes con todos los candidatos, y retorna el logit del candidato target.
    
    - Acepta x como np.ndarray o torch.Tensor (1D o 2D) con [texto | parches].
    - Para cada máscara, construye prompts "Question: <q> Answer: <candidate>" y calcula similitudes.
    - Retorna el logit del candidato correcto o predicho según target_logit.
    """

    def __init__(
        self,
        model_wrapper,                         # p.ej., CLIPWrapper (expone .model)
        base_inputs: Dict[str, torch.Tensor],  # dict del processor para imagen+pregunta base
        question: str,                        # Texto de la pregunta
        candidates: List[str],                 # Lista de candidatos (respuestas posibles)
        answer_correct: Optional[str] = None,  # Respuesta correcta (opcional)
        target_logit: str = "correct",         # "correct" o "predicted"
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        text_len: Optional[int] = None,
        patch_groups: Optional[List[List[int]]] = None,
        text_feature: Optional[torch.Tensor] = None,
    ):
        self.wrapper = model_wrapper
        self.model = getattr(model_wrapper, "model", model_wrapper).eval()
        reference_module = getattr(model_wrapper, "model", model_wrapper)
        self.device = device or next(reference_module.parameters()).device

        # Copia base_inputs al device del modelo
        self.base_inputs = {k: v.to(self.device) for k, v in base_inputs.items()}
        
        self.question = question
        self.candidates = candidates
        self.answer_correct = answer_correct
        self.target_logit = target_logit
        self.tokenizer = getattr(model_wrapper, "tokenizer", None)
        if self.tokenizer is None and hasattr(model_wrapper, "processor"):
            self.tokenizer = getattr(model_wrapper.processor, "tokenizer", None)

        # Inferir patch_size si no viene
        ps = patch_size
        if ps is None:
            ps = getattr(model_wrapper, "patch_size", None)
            if ps is None:
                ps = getattr(getattr(getattr(self.model, "config", None), "vision_config", None), "patch_size", None)
            if ps is None:
                ps = getattr(getattr(getattr(self.model, "vision_model", None), "config", None), "patch_size", None)
            if ps is None:
                ps = getattr(getattr(self.model, "visual", None), "patch_size", None)
            if ps is None:
                raise ValueError("No pude inferir patch_size del modelo. Pásalo explícitamente.")

        if isinstance(ps, (list, tuple)):
            if len(ps) != 2:
                raise ValueError(f"patch_size iterable inesperado: {ps}")
            patch_h, patch_w = int(ps[0]), int(ps[1])
        else:
            patch_h = patch_w = int(ps)

        self.patch_size = (patch_h, patch_w)
        self.patch_h = patch_h
        self.patch_w = patch_w

        # Geometría de la imagen
        _, _, H, W = self.base_inputs["pixel_values"].shape
        if (H % self.patch_h != 0) or (W % self.patch_w != 0):
            raise AssertionError(
                f"Imagen {H}×{W} no divisible por patch_size={self.patch_size}"
            )
        self.grid_h = H // self.patch_h
        self.grid_w = W // self.patch_w
        self.original_num_patches = self.grid_h * self.grid_w
        self.patch_groups = patch_groups if patch_groups else None
        if self.patch_groups:
            self.num_patches = len(self.patch_groups)
        else:
            self.num_patches = self.original_num_patches

        # Longitud de texto (para el split)
        self.text_len = text_len if text_len is not None else self.base_inputs["input_ids"].shape[1]

        # Obtener vocab_size del tokenizer
        tokenizer = self.tokenizer
        self.vocab_size = None

        if tokenizer is not None:
            self.vocab_size = getattr(tokenizer, "vocab_size", None)
            if self.vocab_size is None:
                try:
                    if hasattr(tokenizer, "__len__"):
                        self.vocab_size = len(tokenizer)
                except (TypeError, AttributeError):
                    pass

            if self.vocab_size is None:
                model_config = getattr(self.model, "config", None)
                if model_config is not None:
                    text_config = getattr(model_config, "text_config", None)
                    if text_config is not None:
                        self.vocab_size = getattr(text_config, "vocab_size", None)
                    if self.vocab_size is None:
                        self.vocab_size = getattr(model_config, "vocab_size", None)

        if self.vocab_size is None:
            max_id_in_base = self.base_inputs["input_ids"].max().item()
            self.vocab_size = max(max_id_in_base + 100, 30522)

        if self.vocab_size is None or self.vocab_size <= 0:
            self.vocab_size = 50257

        self.pad_token_id = 0
        if tokenizer is not None:
            self.pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0

        self.use_amp = bool(use_amp and self.device.type == "cuda")

        # Precomputar coordenadas de cada parche original para masking rápido
        self._patch_coords: List[Tuple[int, int, int, int]] = []
        for k in range(self.original_num_patches):
            r = k // self.grid_w
            c = k % self.grid_w
            r0, r1 = r * self.patch_h, (r + 1) * self.patch_h
            c0, c1 = c * self.patch_w, (c + 1) * self.patch_w
            self._patch_coords.append((r0, r1, c0, c1))

        if text_feature is None:
            raise ValueError("Se requiere un embedding de texto precomputado para VQAPredictor.")
        self.text_feature = text_feature.to(self.device)
        self.text_feature = self.text_feature / self.text_feature.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.logit_scale = self._resolve_logit_scale()

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Aplica máscaras SHAP, calcula similitudes con todos los candidatos,
        y retorna el logit del candidato target.
        """
        # Normalizar x → tensor long en device
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(dtype=torch.long, device=self.device)
        else:
            x = x.to(dtype=torch.long, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        B, L = x.shape
        exp_L = self.text_len + self.num_patches
        if L != exp_L:
            raise AssertionError(f"L esperado={exp_L}, recibido={L}")

        input_ids = x[:, :self.text_len]      # [B, L_txt]
        patch_mask_ids = x[:, self.text_len:] # [B, N]

        # Validar y clamp input_ids
        input_ids = input_ids.clamp(min=0, max=self.vocab_size - 1)

        out = torch.empty(B, dtype=torch.float32, device=self.device)

        # Contexto AMP
        amp_ctx = torch.amp.autocast("cuda") if self.use_amp else nullcontext()

        with torch.inference_mode(), amp_ctx:
            for i in range(B):
                # Clonar tensores base
                masked = {k: v.clone() for k, v in self.base_inputs.items()}
                masked["input_ids"] = input_ids[i].unsqueeze(0)  # [1, L_txt]

                # Atender attention_mask si existe
                if "attention_mask" in masked:
                    am = masked["attention_mask"]
                    masked["attention_mask"] = (am[i] if am.shape[0] > i else am[0]).unsqueeze(0)

                mid = patch_mask_ids[i]              # [N]
                pix = masked["pixel_values"].clone()         # [1, 3, H, W]
                self._apply_mask(pix, mid)

                image_feature = self._encode_image_feature(pix)
                sim = torch.sum(image_feature * self.text_feature, dim=-1)
                logit_value = (self.logit_scale * sim).reshape(-1)[0]
                out[i] = logit_value

        return out.detach().cpu().numpy()

    def _apply_mask(self, pix: torch.Tensor, mask_vector: torch.Tensor) -> None:
        mask_value = torch.zeros((1, 3, 1, 1), dtype=pix.dtype, device=pix.device)
        mask_value[:, :, :, :] = 0.0
        if self.patch_groups:
            for group_idx, patch_list in enumerate(self.patch_groups):
                if mask_vector[group_idx].item() == 0:
                    for patch_idx in patch_list:
                        r0, r1, c0, c1 = self._patch_coords[patch_idx]
                        pix[:, :, r0:r1, c0:c1] = mask_value
        else:
            for k in range(self.original_num_patches):
                if mask_vector[k].item() == 0:
                    r0, r1, c0, c1 = self._patch_coords[k]
                    pix[:, :, r0:r1, c0:c1] = mask_value

    def _encode_image_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        model = self.model
        with torch.inference_mode():
            if hasattr(model, "get_image_features"):
                feats = model.get_image_features(pixel_values=pixel_values)
            elif hasattr(model, "encode_image"):
                feats = model.encode_image(pixel_values)
            else:
                raise ValueError("El modelo no expone get_image_features/encode_image.")
        feats = feats.to(self.device)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return feats

    def _resolve_logit_scale(self) -> torch.Tensor:
        scale = getattr(self.model, "logit_scale", None)
        if scale is None and hasattr(self.wrapper, "logit_scale"):
            scale = getattr(self.wrapper, "logit_scale", None)
        if scale is None:
            return torch.tensor(1.0, device=self.device, dtype=torch.float32)
        if isinstance(scale, torch.Tensor):
            scale = scale.to(self.device)
            if hasattr(scale, "exp"):
                return scale.exp()
            if scale.numel() == 1:
                return scale
            return torch.tensor(float(scale.mean().item()), device=self.device, dtype=torch.float32)
        try:
            return torch.tensor(float(scale), device=self.device, dtype=torch.float32)
        except Exception:
            return torch.tensor(1.0, device=self.device, dtype=torch.float32)

