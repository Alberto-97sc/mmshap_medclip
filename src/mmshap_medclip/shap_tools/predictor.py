# src/mmshap_medclip/shap_tools/predictor.py
from typing import Dict, Optional, Union
from contextlib import nullcontext
import numpy as np
import torch


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
        patch_size: Optional[int] = None,      # si None se infiere del modelo
        device: Optional[torch.device] = None,
        use_amp: bool = True,                  # AMP en CUDA
    ):
        self.wrapper = model_wrapper
        self.model = getattr(model_wrapper, "model", model_wrapper).eval()
        self.device = device or next(self.model.parameters()).device

        # Copia base_inputs al device del modelo
        self.base_inputs = {k: v.to(self.device) for k, v in base_inputs.items()}

        # Inferir patch_size si no viene
        if patch_size is None:
            ps = getattr(getattr(getattr(self.model, "config", None), "vision_config", None), "patch_size", None)
            if ps is None:
                ps = getattr(getattr(getattr(self.model, "vision_model", None), "config", None), "patch_size", None)
            if ps is None:
                raise ValueError("No pude inferir patch_size del modelo. Pásalo explícitamente.")
            patch_size = int(ps)
        self.patch_size = int(patch_size)

        # Geometría de la imagen
        _, _, H, W = self.base_inputs["pixel_values"].shape
        if (H % self.patch_size != 0) or (W % self.patch_size != 0):
            raise AssertionError(f"Imagen {H}×{W} no divisible por patch_size={self.patch_size}")
        self.grid_h = H // self.patch_size
        self.grid_w = W // self.patch_size
        self.num_patches = self.grid_h * self.grid_w

        # Longitud de texto (para el split)
        self.text_len = self.base_inputs["input_ids"].shape[1]

        self.use_amp = bool(use_amp and self.device.type == "cuda")

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

        out = torch.empty(B, dtype=torch.float32, device=self.device)

        # Contexto AMP (API nueva de PyTorch)
        amp_ctx = torch.amp.autocast("cuda") if self.use_amp else nullcontext()

        with torch.inference_mode(), amp_ctx:
            for i in range(B):
                # Clonar tensores base para no mutar el estado
                masked = {k: v.clone() for k, v in self.base_inputs.items()}
                masked["input_ids"] = input_ids[i].unsqueeze(0)  # [1, L_txt]

                # Atender attention_mask si existe
                if "attention_mask" in masked:
                    am = masked["attention_mask"]
                    masked["attention_mask"] = (am[i] if am.shape[0] > i else am[0]).unsqueeze(0)

                # Poner en cero los parches donde patch_mask_ids == 0
                mid = patch_mask_ids[i]              # [N]
                pix = masked["pixel_values"]         # [1, 3, H, W]
                for k in range(self.num_patches):
                    if mid[k].item() == 0:
                        r = k // self.grid_w
                        c = k %  self.grid_w
                        r0, r1 = r * self.patch_size, (r + 1) * self.patch_size
                        c0, c1 = c * self.patch_size, (c + 1) * self.patch_size
                        pix[:, :, r0:r1, c0:c1] = 0

                outputs = self.model(**masked)       # logits_per_image: [1,1]
                out[i] = outputs.logits_per_image.squeeze()

        return out.detach().cpu().numpy()


class ClassificationPredictor:
    """
    Predictor para tareas de clasificación con RClip.
    Similar al Predictor original pero adaptado para múltiples clases.
    """

    def __init__(
        self,
        model_wrapper,
        base_inputs: Dict[str, torch.Tensor],
        class_names: list,
        target_class_idx: int,
        patch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
    ):
        self.wrapper = model_wrapper
        self.model = getattr(model_wrapper, "model", model_wrapper).eval()
        self.device = device or next(self.model.parameters()).device
        self.class_names = class_names
        self.target_class_idx = target_class_idx

        # Copia base_inputs al device del modelo
        self.base_inputs = {k: v.to(self.device) for k, v in base_inputs.items()}

        # Inferir patch_size si no viene
        if patch_size is None:
            ps = getattr(getattr(getattr(self.model, "config", None), "vision_config", None), "patch_size", None)
            if ps is None:
                ps = getattr(getattr(getattr(self.model, "vision_model", None), "config", None), "patch_size", None)
            if ps is None:
                raise ValueError("No pude inferir patch_size del modelo. Pásalo explícitamente.")
            patch_size = int(ps)
        self.patch_size = int(patch_size)

        # Geometría de la imagen
        _, _, H, W = self.base_inputs["pixel_values"].shape
        if (H % self.patch_size != 0) or (W % self.patch_size != 0):
            raise AssertionError(f"Imagen {H}×{W} no divisible por patch_size={self.patch_size}")
        self.grid_h = H // self.patch_size
        self.grid_w = W // self.patch_size
        self.num_patches = self.grid_h * self.grid_w

        # Longitud de texto (para el split)
        self.text_len = self.base_inputs["input_ids"].shape[1]

        self.use_amp = bool(use_amp and self.device.type == "cuda")

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

        out = torch.empty(B, dtype=torch.float32, device=self.device)

        # Contexto AMP
        amp_ctx = torch.amp.autocast("cuda") if self.use_amp else nullcontext()

        with torch.inference_mode(), amp_ctx:
            for i in range(B):
                # Clonar tensores base para no mutar el estado
                masked = {k: v.clone() for k, v in self.base_inputs.items()}
                masked["input_ids"] = input_ids[i].unsqueeze(0)  # [1, L_txt]

                # Atender attention_mask si existe
                if "attention_mask" in masked:
                    am = masked["attention_mask"]
                    masked["attention_mask"] = (am[i] if am.shape[0] > i else am[0]).unsqueeze(0)

                # Poner en cero los parches donde patch_mask_ids == 0
                mid = patch_mask_ids[i]              # [N]
                pix = masked["pixel_values"]         # [1, 3, H, W]
                for k in range(self.num_patches):
                    if mid[k].item() == 0:
                        r = k // self.grid_w
                        c = k %  self.grid_w
                        r0, r1 = r * self.patch_size, (r + 1) * self.patch_size
                        c0, c1 = c * self.patch_size, (c + 1) * self.patch_size
                        pix[:, :, r0:r1, c0:c1] = 0

                outputs = self.model(**masked)
                # Para clasificación, tomamos el logit de la clase objetivo
                logits_per_image = outputs.logits_per_image  # [1, num_classes]
                out[i] = logits_per_image[0, self.target_class_idx]

        return out.detach().cpu().numpy()