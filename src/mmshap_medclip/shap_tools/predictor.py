# src/mmshap_medclip/shap_tools/predictor.py
from typing import Dict, Optional, Union, Tuple
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F


class Predictor:
    """
    Callable para SHAP: aplica máscaras de parches sobre pixel_values y ejecuta el modelo.
    - Acepta x como np.ndarray o torch.Tensor (1D o 2D) con [texto | parches].
    - No usa variables globales: todo viene del constructor.
    """

    def __init__(
        self,
        model_wrapper,                         # p.ej., CLIPWrapper (expone .model)
        base_inputs: Dict[str, torch.Tensor],  # dict del processor para este batch
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,  # si None se infiere del modelo
        device: Optional[torch.device] = None,
        use_amp: bool = True,                  # AMP en CUDA
        text_len: Optional[int] = None,        # longitud real de texto (sin padding)
    ):
        self.wrapper = model_wrapper
        self.model = getattr(model_wrapper, "model", model_wrapper).eval()
        reference_module = getattr(model_wrapper, "model", model_wrapper)
        self.device = device or next(reference_module.parameters()).device

        # Copia base_inputs al device del modelo
        self.base_inputs = {k: v.to(self.device) for k, v in base_inputs.items()}
        self._pixel_values = self.base_inputs["pixel_values"]

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
        _, _, H, W = self._pixel_values.shape
        if (H % self.patch_h != 0) or (W % self.patch_w != 0):
            raise AssertionError(
                f"Imagen {H}×{W} no divisible por patch_size={self.patch_size}"
            )
        self.grid_h = H // self.patch_h
        self.grid_w = W // self.patch_w
        self.num_patches = self.grid_h * self.grid_w
        self.image_height = H
        self.image_width = W

        # Longitud de texto (para el split)
        # Usar text_len proporcionado (tokens reales) o fallback al tamaño completo
        self.text_len = text_len if text_len is not None else self.base_inputs["input_ids"].shape[1]

        # Obtener vocab_size del tokenizer para validar input_ids
        tokenizer = getattr(model_wrapper, "tokenizer", None)
        self.vocab_size = None

        if tokenizer is not None:
            # Intentar obtener vocab_size de diferentes formas
            self.vocab_size = getattr(tokenizer, "vocab_size", None)

            # Si el tokenizer tiene un atributo len() o __len__, usarlo
            if self.vocab_size is None:
                try:
                    if hasattr(tokenizer, "__len__"):
                        self.vocab_size = len(tokenizer)
                except (TypeError, AttributeError):
                    pass

            # Intentar desde el modelo/config
            if self.vocab_size is None:
                model_config = getattr(self.model, "config", None)
                if model_config is not None:
                    # Para VisionTextDualEncoderModel, buscar en text_config
                    text_config = getattr(model_config, "text_config", None)
                    if text_config is not None:
                        self.vocab_size = getattr(text_config, "vocab_size", None)
                    # Fallback a vocab_size directo en config
                    if self.vocab_size is None:
                        self.vocab_size = getattr(model_config, "vocab_size", None)

        # Si aún no lo encontramos, calcular desde los input_ids base
        if self.vocab_size is None:
            max_id_in_base = self.base_inputs["input_ids"].max().item()
            # Usar el máximo encontrado + margen seguro, pero con un límite razonable
            self.vocab_size = max(max_id_in_base + 100, 30522)  # BERT base usa 30522

        # Fallback final seguro
        if self.vocab_size is None or self.vocab_size <= 0:
            self.vocab_size = 50257  # Tamaño común de vocabularios grandes

        # Obtener pad_token_id para usarlo en lugar de valores inválidos
        self.pad_token_id = 0
        if tokenizer is not None:
            self.pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0

        self.use_amp = bool(use_amp and self.device.type == "cuda")

        # Valor al que se sustituyen los parches apagados (en el espacio normalizado CLIP)
        self.mask_fill_value = torch.tensor(
            [0.0, 0.0, 0.0],
            dtype=self._pixel_values.dtype,
            device=self.device,
        ).view(1, 3, 1, 1)
        self._mask_fill_is_zero = bool(
            torch.allclose(self.mask_fill_value, torch.zeros_like(self.mask_fill_value))
        )

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
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

        # Validar y clamp input_ids al rango válido del vocabulario
        # Esto previene errores de CUDA "device-side assert triggered"
        input_ids = input_ids.clamp(min=0, max=self.vocab_size - 1)

        out = torch.empty(B, dtype=torch.float32, device=self.device)

        # Contexto AMP (API nueva de PyTorch)
        amp_ctx = torch.amp.autocast("cuda") if self.use_amp else nullcontext()

        with torch.inference_mode(), amp_ctx:
            base_inputs = self.base_inputs
            base_attention = base_inputs.get("attention_mask")

            for i in range(B):
                # Clonar únicamente lo necesario para evitar trabajo extra en GPU
                masked = dict(base_inputs)
                masked["pixel_values"] = self._pixel_values.clone()
                masked["input_ids"] = input_ids[i].unsqueeze(0)  # [1, L_txt]

                # Atender attention_mask si existe
                if base_attention is not None:
                    if base_attention.shape[0] > 1 and base_attention.shape[0] >= (i + 1):
                        masked["attention_mask"] = base_attention[i].unsqueeze(0)
                    else:
                        masked["attention_mask"] = base_attention[0].unsqueeze(0)

                # Construir máscara espacial de parches directamente en GPU
                patch_mask = patch_mask_ids[i].view(1, 1, self.grid_h, self.grid_w)
                patch_mask = patch_mask.to(dtype=masked["pixel_values"].dtype, device=self.device)
                patch_mask = F.interpolate(
                    patch_mask,
                    size=(self.image_height, self.image_width),
                    mode="nearest",
                )

                pix = masked["pixel_values"]  # [1, 3, H, W]
                pix.mul_(patch_mask)
                if not self._mask_fill_is_zero:
                    pix.add_((1.0 - patch_mask) * self.mask_fill_value)

                outputs = self.wrapper(**masked)     # logits_per_image: [1,1]
                out[i] = outputs.logits_per_image.squeeze()

        return out.detach().cpu().numpy()
