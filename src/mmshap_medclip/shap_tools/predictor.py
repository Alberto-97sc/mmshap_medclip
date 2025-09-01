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
        patch_size: Optional[int] = None,      # si None se infere del modelo
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

        # Validar target_class_idx
        if target_class_idx < 0 or target_class_idx >= len(class_names):
            raise ValueError(f"target_class_idx={target_class_idx} fuera de rango [0, {len(class_names)-1}]")

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

        # Obtener información del vocabulario del modelo
        self.vocab_size = getattr(self.model.config, "vocab_size", None)
        if self.vocab_size is None:
            # Intentar obtener del text model
            text_model = getattr(self.model, "text_model", None)
            if text_model:
                self.vocab_size = getattr(text_model.config, "vocab_size", None)
        
        if self.vocab_size is None:
            # Fallback: usar un valor seguro
            self.vocab_size = 50000
            print(f"⚠️ No se pudo obtener vocab_size, usando fallback: {self.vocab_size}")
        
        # Asegurar que vocab_size sea un entero válido
        try:
            self.vocab_size = int(self.vocab_size)
        except (TypeError, ValueError):
            self.vocab_size = 50000
            print(f"⚠️ Error convirtiendo vocab_size, usando fallback: {self.vocab_size}")
        
        # Validar que vocab_size sea positivo
        if self.vocab_size <= 0:
            self.vocab_size = 50000
            print(f"⚠️ vocab_size no válido, usando fallback: {self.vocab_size}")

        # Debug info
        print(f"🔍 ClassificationPredictor inicializado:")
        print(f"  - target_class_idx: {target_class_idx}")
        print(f"  - num_classes: {len(class_names)}")
        print(f"  - text_len: {self.text_len}")
        print(f"  - num_patches: {self.num_patches}")
        print(f"  - vocab_size: {self.vocab_size}")

    def _sanitize_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Sanitiza input_ids para evitar IndexError en el embedding layer.
        - Reemplaza IDs fuera de rango con [PAD] token (ID 0)
        - Mantiene BOS/EOS tokens si están en rango
        """
        # Clonar para no modificar el original
        sanitized = input_ids.clone()
        
        # Obtener tokens especiales del modelo si es posible
        pad_token_id = 0  # Default
        bos_token_id = 1  # Default
        eos_token_id = 2  # Default
        
        # Intentar obtener del config del modelo
        if hasattr(self.model, 'config'):
            config = self.model.config
            # Obtener pad_token_id
            temp_pad = getattr(config, 'pad_token_id', None)
            if temp_pad is not None:
                pad_token_id = temp_pad
                
            # Obtener bos_token_id
            temp_bos = getattr(config, 'bos_token_id', None)
            if temp_bos is not None:
                bos_token_id = temp_bos
                
            # Obtener eos_token_id
            temp_eos = getattr(config, 'eos_token_id', None)
            if temp_eos is not None:
                eos_token_id = temp_eos
            
        # Asegurar que los tokens especiales no sean None
        if pad_token_id is None:
            pad_token_id = 0
        if bos_token_id is None:
            bos_token_id = 1
        if eos_token_id is None:
            eos_token_id = 2
            
        # Validar que los tokens especiales estén en rango
        if pad_token_id >= self.vocab_size:
            pad_token_id = 0
        if bos_token_id >= self.vocab_size:
            bos_token_id = 0
        if eos_token_id >= self.vocab_size:
            eos_token_id = 0
        
        # Validar y sanitizar
        invalid_mask = (sanitized < 0) | (sanitized >= self.vocab_size)
        
        if invalid_mask.any():
            print(f"⚠️ Encontrados {invalid_mask.sum().item()} tokens inválidos")
            print(f"  - Rango válido: [0, {self.vocab_size-1}]")
            print(f"  - Tokens inválidos: {sanitized[invalid_mask].tolist()}")
            print(f"  - Tokens especiales: PAD={pad_token_id}, BOS={bos_token_id}, EOS={eos_token_id}")
            
            # Reemplazar tokens inválidos con PAD
            sanitized[invalid_mask] = pad_token_id
            
            # Preservar BOS y EOS si están en rango
            if bos_token_id < self.vocab_size:
                sanitized[0] = bos_token_id  # Primer token
            if eos_token_id < self.vocab_size and self.text_len > 1:
                sanitized[self.text_len - 1] = eos_token_id  # Último token de texto
                
            print(f"✅ Tokens sanitizados, usando PAD_ID={pad_token_id}")
        
        return sanitized

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
                
                # Sanitizar input_ids antes de pasarlos al modelo
                current_input_ids = input_ids[i].unsqueeze(0)  # [1, L_txt]
                sanitized_input_ids = self._sanitize_input_ids(current_input_ids)
                masked["input_ids"] = sanitized_input_ids

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

                try:
                    print(f"🔍 Ejecutando forward pass {i+1}/{B}...")
                    print(f"🔍 Input IDs sanitizados: {sanitized_input_ids[0].tolist()}")
                    
                    outputs = self.model(**masked)
                    
                    # Debug: verificar la estructura completa de outputs
                    print(f"🔍 Tipo de outputs: {type(outputs)}")
                    print(f"🔍 Atributos de outputs: {dir(outputs)}")
                    
                    # Debug: verificar la forma de outputs si es un tensor
                    if hasattr(outputs, 'shape'):
                        print(f"🔍 Outputs shape: {outputs.shape}")
                    
                    # Debug: verificar si tiene logits_per_image
                    if hasattr(outputs, 'logits_per_image'):
                        logits_shape = outputs.logits_per_image.shape
                        print(f"🔍 Outputs logits_per_image shape: {logits_shape}")
                        print(f"🔍 logits_per_image dtype: {outputs.logits_per_image.dtype}")
                        
                        # Validar que target_class_idx esté en rango
                        if self.target_class_idx >= logits_shape[1]:
                            print(f"⚠️ target_class_idx={self.target_class_idx} >= logits_shape[1]={logits_shape[1]}")
                            # Usar el primer logit como fallback
                            out[i] = outputs.logits_per_image[0, 0]
                        else:
                            print(f"✅ Accediendo a logits_per_image[0, {self.target_class_idx}]")
                            out[i] = outputs.logits_per_image[0, self.target_class_idx]
                    else:
                        print(f"⚠️ No se encontró logits_per_image en outputs")
                        # Fallback: usar el primer logit disponible
                        if hasattr(outputs, 'logits'):
                            print(f"🔍 Usando outputs.logits como fallback")
                            if hasattr(outputs.logits, 'shape'):
                                print(f"🔍 outputs.logits shape: {outputs.logits.shape}")
                            out[i] = outputs.logits[0, 0]
                        else:
                            print(f"⚠️ No se encontró ningún atributo de logits")
                            # Último recurso: usar 0.0
                            out[i] = torch.tensor(0.0, device=self.device)
                            
                except Exception as e:
                    print(f"❌ Error en forward pass {i+1}/{B}: {e}")
                    print(f"  - target_class_idx: {self.target_class_idx}")
                    print(f"  - class_names: {self.class_names}")
                    print(f"  - Tipo de error: {type(e).__name__}")
                    import traceback
                    print(f"  - Traceback: {traceback.format_exc()}")
                    # Fallback: usar 0.0
                    out[i] = torch.tensor(0.0, device=self.device)

        return out.detach().cpu().numpy()