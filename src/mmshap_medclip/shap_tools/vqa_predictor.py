# src/mmshap_medclip/shap_tools/vqa_predictor.py
from typing import Dict, Optional, Union, Tuple, List
from contextlib import nullcontext
import numpy as np
import torch

from mmshap_medclip.tasks.utils import prepare_batch


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
        self.num_patches = self.grid_h * self.grid_w

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

                # Enmascarar los parches donde patch_mask_ids == 0
                mid = patch_mask_ids[i]              # [N]
                pix = masked["pixel_values"]         # [1, 3, H, W]

                mask_value = torch.tensor([0.0, 0.0, 0.0],
                                         dtype=pix.dtype,
                                         device=pix.device).view(1, 3, 1, 1)

                for k in range(self.num_patches):
                    if mid[k].item() == 0:
                        r = k // self.grid_w
                        c = k %  self.grid_w
                        r0, r1 = r * self.patch_h, (r + 1) * self.patch_h
                        c0, c1 = c * self.patch_w, (c + 1) * self.patch_w
                        pix[:, :, r0:r1, c0:c1] = mask_value

                # Reconstruir el texto de la pregunta (solo imagen + pregunta deben influir en SHAP)
                question_text = self._tokens_to_question_text(masked["input_ids"][0])

                # Construir prompts para todos los candidatos
                prompts = [
                    f"Question: {question_text} Answer: {candidate}"
                    for candidate in self.candidates
                ]
                
                # Convertir imagen enmascarada de tensor normalizado a PIL
                # La imagen está normalizada con CLIP_MEAN y CLIP_STD
                from PIL import Image
                CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], 
                                       dtype=pix.dtype, device=pix.device).view(1, 3, 1, 1)
                CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                                       dtype=pix.dtype, device=pix.device).view(1, 3, 1, 1)
                
                # Desnormalizar
                img_denorm = pix * CLIP_STD + CLIP_MEAN
                img_denorm = torch.clamp(img_denorm, 0, 1)
                
                # Convertir a numpy y luego a PIL
                img_np = img_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                
                # Calcular similitudes con todos los candidatos usando prepare_batch
                # Imagen repetida para cada candidato
                images_pil = [img_pil] * len(self.candidates)
                
                # Usar prepare_batch para procesar todos los candidatos a la vez
                inputs_batch, logits_batch = prepare_batch(
                    self.wrapper,
                    prompts,
                    images_pil,
                    device=self.device,
                    debug_tokens=False,
                    amp_if_cuda=self.use_amp
                )
                
                # Extraer similitudes
                # logits_batch tiene forma [B, 1] donde B es el número de candidatos
                similarities = logits_batch.squeeze(-1).cpu()  # [B] - quitar la dimensión 1
                
                # Asegurar que similarities sea 1D
                if similarities.ndim == 0:
                    similarities = similarities.unsqueeze(0)
                elif similarities.ndim > 1:
                    # Si por alguna razón es 2D, aplanarlo
                    similarities = similarities.flatten()
                
                # Convertir a numpy para evitar problemas con .item()
                similarities_np = similarities.numpy()
                
                # Determinar qué logit retornar
                if self.target_logit == "correct" and self.answer_correct is not None:
                    # Buscar índice del candidato correcto
                    try:
                        correct_idx = next(
                            i for i, cand in enumerate(self.candidates)
                            if cand.lower().strip() == self.answer_correct.lower().strip()
                        )
                        # Asegurar que correct_idx esté dentro del rango
                        if 0 <= correct_idx < len(similarities_np):
                            target_logit_value = float(similarities_np[correct_idx])
                        else:
                            # Si el índice está fuera de rango, usar el predicho
                            pred_idx = int(np.argmax(similarities_np))
                            target_logit_value = float(similarities_np[pred_idx])
                    except (StopIteration, IndexError):
                        # Si no se encuentra, usar el predicho
                        pred_idx = int(np.argmax(similarities_np))
                        target_logit_value = float(similarities_np[pred_idx])
                else:
                    # Usar el predicho (mayor similitud)
                    pred_idx = int(np.argmax(similarities_np))
                    target_logit_value = float(similarities_np[pred_idx])
                
                out[i] = target_logit_value

        return out.detach().cpu().numpy()

    def _tokens_to_question_text(self, token_ids: torch.Tensor) -> str:
        """
        Convierte los input_ids (con máscaras aplicadas) en un texto legible que
        solo contenga la pregunta. Si falla la decodificación, se usa la pregunta original.
        """
        ids_list = token_ids.detach().cpu().tolist()
        question_text = None
        tokenizer = self.tokenizer

        if tokenizer is not None:
            decode_fn = getattr(tokenizer, "decode", None)
            if callable(decode_fn):
                try:
                    question_text = decode_fn(ids_list, skip_special_tokens=True)
                    if isinstance(question_text, (list, tuple)):
                        question_text = question_text[0]
                except Exception:
                    question_text = None

            if (question_text is None or not question_text.strip()) and hasattr(tokenizer, "convert_ids_to_tokens"):
                try:
                    tokens = tokenizer.convert_ids_to_tokens(ids_list)
                    specials = set(getattr(tokenizer, "all_special_tokens", []) or [])
                    filtered_tokens = [t for t in tokens if t and t not in specials]
                    if hasattr(tokenizer, "convert_tokens_to_string"):
                        question_text = tokenizer.convert_tokens_to_string(filtered_tokens)
                    else:
                        question_text = " ".join(filtered_tokens)
                except Exception:
                    question_text = None

        if question_text is None or not question_text.strip():
            fallback_tokens = [str(i) for i in ids_list if i not in (0, self.pad_token_id)]
            question_text = " ".join(fallback_tokens).strip()

        if not question_text:
            question_text = self.question

        return question_text.strip()

